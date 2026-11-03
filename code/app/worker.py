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
    return db.create_job(path.name, str(path), sha, info["size"], info["duration"])


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


def worker_loop(cfg: Config, stop: threading.Event | None = None) -> None:
    db = DB(cfg.db_path)
    log.info("worker 启动，db=%s", cfg.db_path)
    while not (stop and stop.is_set()):
        if not run_once(db, cfg):
            time.sleep(2)


def scan_input(cfg: Config) -> int:
    """把 input/ 里未处理过的文件入队。"""
    db = DB(cfg.db_path)
    n = 0
    for p in sorted(cfg.input_path.glob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            try:
                if ingest(db, cfg, p):
                    n += 1
            except Exception as e:
                log.warning("跳过 %s: %s", p.name, e)
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
            files = sorted(cfg.input_path.glob("*"))
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
