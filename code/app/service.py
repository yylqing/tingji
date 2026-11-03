"""把听记注册成 macOS 登录服务（launchd LaunchAgent），开机自启。

统一入口是 `python -m app.cli service`：一个进程里同时跑
文件夹监听 + 队列处理 + Web 服务。launchd 只负责把它拉起来并保活。
"""
from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = CODE_ROOT.parent
LABEL = "com.tingji.service"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def _uid() -> int:
    return os.getuid()


def build_plist(python: str, root: Path, config: str, log_dir: Path,
                host: str = "", port: int = 0, variant: str = "full") -> dict:
    """variant：full（带 KeepAlive/ProcessType）/ minimal（最简）/ no_workdir（不设工作目录）。

    有些 macOS 版本对本机 Documents 下的 LaunchAgent 会在 spawn 阶段报 78 EX_CONFIG，
    给安装脚本留了逐级降级的余地。
    """
    # 注意：-c/--config 是 argparse 的全局参数，必须放在子命令 service 之前
    args = [python, "-m", "app.cli"]
    if config:
        args += ["-c", config]
    args += ["service"]
    if host:
        args += ["--host", host]
    if port:
        args += ["--port", str(port)]
    plist = {
        "Label": LABEL,
        "ProgramArguments": args,
        "EnvironmentVariables": {
            "PYTHONPATH": str(root),
            "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin",
            "LANG": "zh_CN.UTF-8",
        },
        "RunAtLoad": True,
        "StandardOutPath": str(log_dir / "service.log"),
        "StandardErrorPath": str(log_dir / "service.err.log"),
    }
    if variant != "no_workdir":
        plist["WorkingDirectory"] = str(root)
    if variant == "full":
        plist["KeepAlive"] = {"SuccessfulExit": False}
        plist["ProcessType"] = "Background"
        plist["ThrottleInterval"] = 10
    return plist


def install(python: str | None = None, config: str = "config/config.yaml",
            host: str = "", port: int = 0) -> dict:
    python = python or sys.executable
    # launchd 打开 ~/Documents 下的日志文件会触发 TCC 限制（EX_CONFIG 78），
    # 所以自启日志统一放 ~/Library/Logs（不受保护），应用自身日志仍在 output/.runtime/logs
    log_dir = Path.home() / "Library" / "Logs" / "TingJi"
    log_dir.mkdir(exist_ok=True)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    cfg_path = str(CODE_ROOT / config) if not Path(config).is_absolute() else config
    plist = build_plist(python, CODE_ROOT, cfg_path, log_dir, host, port)
    # 预检：先把要写进 plist 的命令跑一次 --help，参数顺序错会在这里暴露，
    # 而不是等 launchd 反复崩溃（踩过：--config 放在子命令后面 → argparse 报错 → 退出）
    try:
        probe = subprocess.run(plist["ProgramArguments"] + ["--help"],
                               capture_output=True, text=True, timeout=20)
        if probe.returncode != 0:
            return {"ok": False, "error": f"命令行预检失败（exit {probe.returncode}）："
                                          f"{(probe.stderr or probe.stdout).strip()[:300]}",
                    "plist": plist}
    except Exception as e:
        return {"ok": False, "error": f"命令行预检异常：{e}", "plist": plist}
    try:
        PLIST_PATH.write_bytes(plistlib.dumps(plist))
    except PermissionError as e:
        # 沙箱/权限不允许写 ~/Library/LaunchAgents：把 plist 落到项目内，
        # 让用户用 scripts/install-autostart.sh 执行一次（那条命令在沙箱外跑）
        local = CODE_ROOT / "config" / f"{LABEL}.plist"
        local.write_bytes(plistlib.dumps(plist))
        return {"ok": False, "permission": True, "path": str(PLIST_PATH),
                "local": str(local), "error": str(e), "plist": plist}
    # 先卸载旧的（若存在），再加载
    subprocess.run(["launchctl", "bootout", f"gui/{_uid()}", str(PLIST_PATH)],
                   capture_output=True)
    r = subprocess.run(["launchctl", "bootstrap", f"gui/{_uid()}", str(PLIST_PATH)],
                       capture_output=True, text=True)
    ok = r.returncode == 0
    return {"ok": ok, "path": str(PLIST_PATH), "error": (r.stderr or "").strip()[:300],
            "plist": plist}


def uninstall() -> dict:
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "bootout", f"gui/{_uid()}", str(PLIST_PATH)],
                       capture_output=True)
        PLIST_PATH.unlink(missing_ok=True)
        return {"ok": True, "removed": True}
    return {"ok": True, "removed": False}


def status() -> dict:
    r = subprocess.run(["launchctl", "print", f"gui/{_uid()}/{LABEL}"],
                       capture_output=True, text=True)
    running = r.returncode == 0
    info = ""
    if running:
        for line in r.stdout.splitlines():
            s = line.strip()
            if s.startswith(("state =", "pid =", "last exit")):
                info += s + " "
    return {"installed": PLIST_PATH.exists(), "running": running, "detail": info.strip(),
            "path": str(PLIST_PATH)}


def kickstart() -> dict:
    r = subprocess.run(["launchctl", "kickstart", "-k", f"gui/{_uid()}/{LABEL}"],
                       capture_output=True, text=True)
    return {"ok": r.returncode == 0, "error": (r.stderr or "").strip()[:300]}
