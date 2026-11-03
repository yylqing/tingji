"""命令行入口：run / ingest / worker / serve / hotword / selfcheck。"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

from .pipeline import run_job, sha256_file
from .settings import CODE_ROOT, ROOT, load_config


def _setup_log(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _local_ips() -> list[str]:
    """本机可用于局域网访问的 IPv4，主网卡优先（过滤虚拟网卡的网络地址）。"""
    import socket
    import subprocess
    ips: list[str] = []
    # macOS：en0 通常是主网卡（Wi-Fi/有线），最可能是你要的那个
    for dev in ("en0", "en1"):
        try:
            ip = subprocess.check_output(["ipconfig", "getifaddr", dev],
                                         text=True, stderr=subprocess.DEVNULL).strip()
            if ip and ip not in ips:
                ips.append(ip)
        except Exception:
            pass
    try:
        sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sk.connect(("8.8.8.8", 80))
        ip = sk.getsockname()[0]
        sk.close()
        if ip not in ips:
            ips.append(ip)
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    # 过滤回环、链路本地、以 .0 结尾的网络地址
    return [ip for ip in ips
            if not ip.startswith(("127.", "169.254.")) and not ip.endswith(".0")]


def _prepare_server(cfg, host: str | None, port: int | None) -> tuple[str, int, str]:
    """统一处理监听地址与访问令牌；返回 (bind, port, token)。"""
    import secrets
    bind = host or cfg.server.bind
    p = port or cfg.server.port
    loopback = bind in ("127.0.0.1", "localhost", "::1")
    token = getattr(cfg.server, "token", "") or ""
    if not loopback and not token:
        tok_file = CODE_ROOT / "config" / "access_token.txt"
        if tok_file.exists():
            token = tok_file.read_text().strip()
        if not token:
            token = secrets.token_urlsafe(18)
            tok_file.write_text(token + "\n")
            try:
                tok_file.chmod(0o600)
            except OSError:
                pass
            print(f"🔑 已生成访问令牌并写入 {tok_file}")
    cfg.server.token = token
    return bind, p, token


def _print_urls(bind: str, port: int, token: str) -> None:
    loopback = bind in ("127.0.0.1", "localhost", "::1")
    if loopback:
        print(f"🖥  仅本机：http://127.0.0.1:{port}/")
    else:
        print(f"🖥  本机：  http://127.0.0.1:{port}/?token={token}")
        ips = _local_ips()
        if ips:
            print(f"🌐 局域网：http://{ips[0]}:{port}/?token={token}")
            if len(ips) > 1:
                print(f"   其他网卡：{', '.join(ips[1:])}")
        else:
            print("🌐 局域网：未探测到网卡地址（用 ifconfig 自查）")
        print("（浏览器首次带 token 访问后会写入 cookie，之后可直接用 /）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tingji", description="本地离线会议转写")
    ap.add_argument("-c", "--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="直接处理一个音频文件（同步）")
    p_run.add_argument("audio")
    p_run.add_argument("--force", action="store_true", help="忽略 sha256 去重")
    p_run.add_argument("--speakers", type=int, default=0, help="已知说话人数（0=自动）")
    p_run.add_argument("--correct", action="store_true", help="启用 LLM 纠错")
    p_run.add_argument("--asr", choices=["funasr", "sensevoice", "funasr_nano"], default=None, help="识别引擎")

    p_ing = sub.add_parser("ingest", help="把文件/目录放入队列")
    p_ing.add_argument("path")

    sub.add_parser("worker", help="消费队列")
    sub.add_parser("scan", help="扫描 input/ 并入队")
    sub.add_parser("watch", help="监听 input/ 并自动入队处理（常驻）")

    p_all = sub.add_parser("service", help="统一入口：监听文件夹 + 队列处理 + Web（常驻）")
    p_all.add_argument("--host", default=None)
    p_all.add_argument("--port", type=int, default=None)
    p_all.add_argument("--no-web", action="store_true", help="只跑监听与队列，不起 Web")

    p_auto = sub.add_parser("autostart", help="开机自启（macOS launchd）")
    p_auto.add_argument("action", choices=["install", "uninstall", "status", "restart"])
    p_auto.add_argument("--port", type=int, default=None)

    p_srv = sub.add_parser("serve", help="启动 Web 服务")
    p_srv.add_argument("--host", default=None)
    p_srv.add_argument("--port", type=int, default=None)
    p_srv.add_argument("--with-worker", action="store_true", help="同进程内跑 worker")

    p_hw = sub.add_parser("hotword", help="热词管理")
    p_hw.add_argument("action", choices=["list", "add", "del", "import"])
    p_hw.add_argument("args", nargs="*")

    sub.add_parser("selfcheck", help="自检模型与依赖")

    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    _setup_log(cfg.app.log_level)

    from .db import DB
    db = DB(cfg.db_path)

    if a.cmd == "run":
        from .engines import FFmpeg
        src = Path(a.audio).resolve()
        sha = sha256_file(src)
        existing = db.find_by_sha(sha)
        info = FFmpeg.probe(src)
        if existing and not a.force:
            print("该文件已处理过（sha256 重复），已跳过。加 --force 可强制重跑。")
            return 0
        if a.speakers:
            cfg.diarization.max_speakers = a.speakers
        if a.correct:
            cfg.correction.enabled = True
        if a.asr:
            cfg.asr.provider = a.asr
        if existing:
            jid = existing["id"]
            db.update_job(jid, status="queued", progress=0, stage=None, message="强制重跑",
                          error_code=None, degraded=0, finished_at=None)
        else:
            jid = db.create_job(src.name, str(src), sha, info["size"], info["duration"])
        print(f"任务 {jid}：{src.name}  {info['duration'] / 60:.1f} 分钟")
        run_job(db, cfg, jid)
        job = db.get_job(jid)
        print(f"完成：{job['status']}  用时 {job['message']}")
        return 0 if job["status"] == "done" else 1

    if a.cmd == "ingest":
        from .worker import ingest
        p = Path(a.path).resolve()
        targets = sorted(p.glob("*")) if p.is_dir() else [p]
        n = 0
        for t in targets:
            if t.is_file():
                jid = ingest(db, cfg, t)
                if jid:
                    print(f"入队 {jid}  {t.name}")
                    n += 1
        print(f"共入队 {n} 个")
        return 0

    if a.cmd == "watch":
        from .worker import watch_input
        import threading
        stop = threading.Event()
        print(f"👀 监听 {cfg.input_path}（Ctrl+C 退出）")
        try:
            watch_input(cfg, stop)
        except KeyboardInterrupt:
            stop.set()
            print("已停止")
        return 0

    if a.cmd == "scan":
        from .worker import scan_input
        print(f"入队 {scan_input(cfg)} 个")
        return 0

    if a.cmd == "worker":
        from .worker import worker_loop
        worker_loop(cfg)
        return 0

    if a.cmd == "service":
        import threading
        import uvicorn
        from .worker import watch_input, worker_loop
        stop = threading.Event()
        threading.Thread(target=worker_loop, args=(cfg, stop), daemon=True, name="worker").start()
        threading.Thread(target=watch_input, args=(cfg, stop), daemon=True, name="watch").start()
        bind, port, token = _prepare_server(cfg, a.host, a.port)
        print(f"👀 监听 {cfg.input_path}｜⚙️ 队列处理｜🌐 Web :{port}")
        if not a.no_web:
            _print_urls(bind, port, token)
        if a.no_web:
            try:
                stop.wait()
            except KeyboardInterrupt:
                stop.set()
            return 0
        from .web import create_app
        uvicorn.run(create_app(cfg), host=bind, port=port, log_level="info")
        return 0

    if a.cmd == "autostart":
        from . import service as svc
        if a.action == "install":
            r = svc.install(config=a.config or "config/config.yaml",
                            port=a.port or cfg.server.port)
            if not r["ok"] and r.get("permission"):
                print("⚠️  无权写入 ~/Library/LaunchAgents（当前沙箱/权限）")
                print(f"   plist 已生成：{r['local']}")
                print("   请在你自己的终端执行一次（只需一次）：")
                print("     bash scripts/install-autostart.sh")
                return 3
            if not r["ok"]:
                print("❌ 安装失败:", r["error"] or "未知错误")
                return 2
            print(f"✅ 已注册开机自启：{r['path']}")
            print(f"   日志：{cfg.work_path.parent}/logs/service.log")
            print("   立即启动：python -m app.cli autostart restart")
        elif a.action == "uninstall":
            r = svc.uninstall()
            print("✅ 已移除开机自启" if r["removed"] else "（本来就没安装）")
        elif a.action == "restart":
            print(svc.kickstart())
        else:
            st = svc.status()
            print(f"已安装：{'是' if st['installed'] else '否'}｜运行中：{'是' if st['running'] else '否'}")
            if st["detail"]:
                print("  ", st["detail"])
            if st["installed"]:
                print("   plist:", st["path"])
        return 0

    if a.cmd == "serve":
        import uvicorn
        bind, port, token = _prepare_server(cfg, a.host, a.port)
        _print_urls(bind, port, token)
        from .web import create_app
        if a.with_worker:
            from .worker import worker_loop
            threading.Thread(target=worker_loop, args=(cfg,), daemon=True).start()
        uvicorn.run(create_app(cfg), host=bind, port=port, log_level="info")
        return 0

    if a.cmd == "hotword":
        if a.action == "list":
            for r in db.list_hotwords():
                print(f"  {r['term']:<16} {r['type']:<8} w={r['weight']}  别名={','.join(r['aliases']) or '-'}")
        elif a.action == "add":
            term = a.args[0]
            aliases = a.args[1].split(",") if len(a.args) > 1 and a.args[1] else []
            db.upsert_hotword(term, aliases=aliases)
            print(f"已添加 {term}")
        elif a.action == "del":
            db.delete_hotword(a.args[0])
            print(f"已删除 {a.args[0]}")
        elif a.action == "import":
            import yaml
            rows = yaml.safe_load(Path(a.args[0]).read_text(encoding="utf-8")) or []
            for r in rows:
                db.upsert_hotword(r["term"], r.get("type", "term"), r.get("aliases") or [],
                                  float(r.get("weight", 1.0)), r.get("note", ""))
            print(f"导入 {len(rows)} 条")
        return 0

    if a.cmd == "selfcheck":
        ok = True
        import shutil
        for name, cmd in [("ffmpeg", "ffmpeg"), ("ffprobe", "ffprobe")]:
            p = shutil.which(cmd)
            print(f"  {'✅' if p else '❌'} {name}: {p or '未找到'}")
            ok &= bool(p)
        for label, path in [("CAM++ ONNX", cfg.diarization.campplus_model)]:
            fp = Path(path) if Path(path).is_absolute() else CODE_ROOT / path
            print(f"  {'✅' if fp.exists() else '❌'} {label}: {fp}")
            ok &= fp.exists()
        try:
            from .engines import AsrEngine
            AsrEngine(cfg.asr)._load()
            print("  ✅ FunASR 模型可加载")
        except Exception as e:
            print(f"  ❌ FunASR: {e}")
            ok = False
        from .engines import LlmEngine
        llm = LlmEngine(cfg.llm, cfg.correction.model)
        print(f"  {'✅' if llm.alive() else '⚠️'} Ollama: {cfg.llm.base_url}")
        print("自检结果:", "通过" if ok else "存在问题")
        return 0 if ok else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
