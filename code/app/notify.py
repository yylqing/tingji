"""任务完成/失败通知（webhook）。

配置 `notify.webhook_url` 后，任务结束会 POST 一条 JSON：
{"event": "done"|"failed", "job_id", "source", "duration_sec", "message", "note"}
失败不重试、不影响主流程（通知永远不能把任务搞挂）。
"""
from __future__ import annotations

import json
import logging
import urllib.request

log = logging.getLogger("tingji.notify")


def send(cfg, job: dict, event: str, message: str = "", note: str = "") -> bool:
    url = getattr(getattr(cfg, "notify", None), "webhook_url", "") or ""
    if not url or not getattr(cfg.notify, "enabled", False):
        return False
    payload = {
        "event": event,
        "job_id": job.get("id"),
        "source": job.get("source_name"),
        "duration_sec": job.get("duration_sec"),
        "message": (message or "")[:300],
        "note": note,
    }
    try:
        req = urllib.request.Request(
            url, data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            ok = 200 <= r.status < 300
        log.info("通知已发送 %s → %s", event, url)
        return ok
    except Exception as e:
        log.warning("通知发送失败（忽略）: %s", e)
        return False
