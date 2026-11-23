"""队列消费与目录监听。"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .db import DB
from .pipeline import run_job, sha256_file
from .settings import Config

log = logging.getLogger("tingji.worker")

AUDIO_EXT = {
    ".wav", ".mp3", ".aac", ".m4a", ".mp4", ".mov",
    ".webm", ".weba", ".opus", ".ogg", ".oga", ".flac",
    ".wma", ".amr", ".3gp", ".caf", ".aiff", ".aif", ".m4b", ".mkv",
}


def iter_audio(cfg: Config):
    """递归枚举 input/ 下的音频文件（跳过隐藏文件与 ._ 资源叉）。

    之前只 glob("*") 扫第一层，子文件夹里的录音会被忽略。
    """
    for p in sorted(cfg.input_path.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in AUDIO_EXT:
            continue
        if any(part.startswith(".") for part in p.relative_to(cfg.input_path).parts):
            continue
        yield p


def display_name(cfg: Config, path: Path) -> str:
    """嵌套文件用相对路径做名字，避免不同子目录里的同名文件互相覆盖输出目录。"""
    try:
        rel = path.resolve().relative_to(cfg.input_path.resolve())
    except ValueError:
        return path.name
    if len(rel.parts) == 1:
        return path.name
    return "／".join(rel.parts)      # 全角斜杠：可读且能当文件名


def ingest(db: DB, cfg: Config, path: Path) -> str | None:
    from .engines import FFmpeg
    path = path.resolve()
    if path.suffix.lower() not in AUDIO_EXT:
        return None
    sha = sha256_file(path)
    if db.find_by_sha(sha):
        log.info("重复文件，跳过: %s", path.name)
        return None
    info = FFmpeg.probe(path)
    if info["duration"] <= 0:
        raise ValueError("无法解码或无时长")
    return db.create_job(display_name(cfg, path), str(path), sha, info["size"], info["duration"])


def run_once(db: DB, cfg: Config) -> bool:
    job = db.next_queued()
    if not job:
        return False
    log.info("开始处理 %s (%s)", job["id"], job["source_name"])
    try:
        run_job(db, cfg, job["id"])
    except Exception as e:
        log.exception("任务失败")
        db.update_job(job["id"], status="failed", error_code=type(e).__name__,
                      message=str(e)[:500], finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        db.log(job["id"], "error", "failed", f"{type(e).__name__}: {e}")
        try:
            from . import notify
            notify.send(cfg, dict(job), "failed", str(e))
        except Exception:
            pass
    return True


def _lock_busy(cfg: Config, jid: str) -> bool:
    """判断某个任务的 job 目录是否正被别的进程占用。"""
    import fcntl
    lock = cfg.work_path / jid / ".lock"
    if not lock.exists():
        return False
    try:
        with open(lock, "w") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return False
    except OSError:
        return True


def worker_loop(cfg: Config, stop: threading.Event | None = None) -> None:
    db = DB(cfg.db_path)
    log.info("worker 启动，db=%s", cfg.db_path)
    try:
        n = db.requeue_stale_running(lambda jid: _lock_busy(cfg, jid))
        if n:
            log.warning("重置 %d 个中断的任务为排队（服务重启导致）", n)
    except Exception as e:
        log.warning("重置中断任务失败（忽略）: %s", e)
    while not (stop and stop.is_set()):
        if not run_once(db, cfg):
            time.sleep(2)


def scan_input(cfg: Config) -> int:
    """把 input/ 里未处理过的文件入队。"""
    db = DB(cfg.db_path)
    n = 0
    for p in iter_audio(cfg):
        try:
            if ingest(db, cfg, p):
                n += 1
                log.info("入队 %s", display_name(cfg, p))
        except Exception as e:
            log.warning("跳过 %s: %s", display_name(cfg, p), e)
    return n


def watch_input(cfg: Config, stop: threading.Event) -> None:
    """轮询监听 input/，文件大小稳定后自动入队。

    非阻塞实现：每次循环只比较"上次看到的大小"，不再为每个文件 sleep，
    否则目录里已有几个文件时，新文件要排队等好几秒。
    """
    seen: set[str] = set()
    pending: dict[str, tuple[int, float]] = {}
    db = DB(cfg.db_path)
    while not stop.is_set():
        try:
            files = list(iter_audio(cfg))
        except Exception:
            files = []
        now = time.monotonic()
        for p in files:
            if not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
                continue
            key = str(p.resolve())
            if key in seen:
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            prev = pending.get(key)
            if prev and prev[0] == size and size > 0 and now - prev[1] >= 2.0:
                seen.add(key)
                pending.pop(key, None)
                try:
                    jid = ingest(db, cfg, p)
                    if jid:
                        log.info("自动入队 %s → %s", p.name, jid)
                    else:
                        log.info("重复文件，跳过 %s", p.name)
                except Exception as e:
                    log.warning("入队失败 %s: %s", p.name, e)
            elif not prev or prev[0] != size:
                pending[key] = (size, now)
        stop.wait(1.5)


def _stable(p: Path, checks: int = 2, interval: float = 2.0) -> bool:
    last = -1
    for _ in range(checks + 1):
        size = p.stat().st_size
        if size == last and size > 0:
            return True
        last = size
        time.sleep(interval)
    return False
