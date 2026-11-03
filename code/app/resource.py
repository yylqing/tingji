"""资源守卫：任务开始前检查磁盘与内存，避免跑到一半 OOM / 磁盘满。

16GB 机器上 5 小时音频 + 中间文件会同时占用几 GB；磁盘满会让归一化失败，
内存不足会让 ASR 进程被系统杀掉（表现为"任务莫名失败"）。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# 经验值：16k 单声道 wav ≈ 115MB/小时；中间文件（切片/预览/缓存）≈ 2 倍
BYTES_PER_SEC = 32_000 * 3


def available_memory_gb() -> float | None:
    """返回可用内存（GB）；取不到时返回 None。"""
    try:
        import psutil
        return round(psutil.virtual_memory().available / 1024 ** 3, 1)
    except Exception:
        pass
    try:
        # macOS: vm_stat 的 free + inactive 页
        out = subprocess.check_output(["vm_stat"], text=True, timeout=2)
        page = 16384
        free = inactive = 0
        for line in out.splitlines():
            if "page size of" in line:
                page = int(line.split("page size of")[1].split()[0])
            elif line.startswith("Pages free"):
                free = int(line.split(":")[1].strip(" ."))
            elif line.startswith("Pages inactive"):
                inactive = int(line.split(":")[1].strip(" ."))
        if page and (free or inactive):
            return round((free + inactive) * page / 1024 ** 3, 1)
    except Exception:
        pass
    return None


def check(cfg, duration_sec: float, free_gb: float | None = None,
          mem_gb: float | None = None) -> tuple[bool, str]:
    """返回 (是否放行, 说明)。free_gb/mem_gb 可注入，便于测试。"""
    need_bytes = max(1_500_000_000, int(max(duration_sec, 0) * BYTES_PER_SEC))
    if free_gb is None:
        try:
            free_gb = shutil.disk_usage(_existing_parent(cfg.work_path)).free / 1024 ** 3
        except Exception:
            free_gb = None
    if free_gb is not None and free_gb * 1024 ** 3 < need_bytes:
        return False, f"磁盘可用 {free_gb:.1f} GB，本任务约需 {need_bytes / 1024 ** 3:.1f} GB"

    if mem_gb is None:
        mem_gb = available_memory_gb()
    if mem_gb is not None and mem_gb < 1.5:
        return False, f"可用内存仅 {mem_gb:.1f} GB（建议 ≥1.5 GB）"
    return True, ""


def _existing_parent(p: Path) -> Path:
    p = Path(p)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p
