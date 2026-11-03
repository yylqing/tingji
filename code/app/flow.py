"""任务流程图：把事件流折算成「节点 + 状态 + 耗时」，供 Web 展示。

节点顺序固定（管线顺序），状态从 `events` 表推断：
- done：该阶段有事件且不是"跳过"；
- skipped：该阶段事件文案含"跳过"/"命中…缓存"；
- running：是任务当前 stage 且任务未结束；
- failed：任务失败且该阶段是失败前最后一个阶段；
- pending：还没有事件。
"""
from __future__ import annotations

from datetime import datetime

# 顺序 + 中文名 + 一句说明
FLOW: list[tuple[str, str, str]] = [
    ("normalized", "归一化", "解码为 16k 单声道、响度归一"),
    ("planned", "分窗", "按时长切分处理窗口"),
    ("transcribed", "语音识别", "VAD + ASR + 标点"),
    ("emotion", "情绪/事件", "SenseVoice 情绪与声学事件"),
    ("diarized", "说话人分离", "帧级分离 / 声纹聚类"),
    ("identified", "声纹认名", "与声纹库比对"),
    ("cleaned", "文本清洗", "去口水词、重复、ITN"),
    ("hotworded", "热词修正", "拼音模糊替换"),
    ("corrected", "LLM 纠错", "逐句/按块纠错（可选）"),
    ("refined", "整理稿", "段落化 + 润色"),
    ("roles", "角色推断", "从内容判断身份"),
    ("summarized", "智能纪要", "摘要 + 章节 + 待办"),
    ("archived", "归档", "写出文件 / Obsidian"),
]

SKIP_MARKS = ("跳过", "命中", "无有效语音")
DONE_STATES = ("done", "failed", "archived", "cancelled")


def _ts(s: str):
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


def _last_run(events: list[dict]) -> list[dict]:
    """只保留最近一次运行的事件（以最后一个 normalized 为界）。"""
    start = 0
    for i, e in enumerate(events):
        if (e.get("stage") or "") == "normalized":
            start = i
    return events[start:]


def build(job: dict, events: list[dict]) -> list[dict]:
    events = _last_run(events)
    by: dict[str, list[dict]] = {}
    for e in events:
        by.setdefault(e.get("stage") or "", []).append(e)

    cur = (job.get("stage") or "").strip()
    finished = (job.get("status") in DONE_STATES)
    failed = job.get("status") == "failed"

    nodes = []
    for i, (key, label, desc) in enumerate(FLOW):
        evs = by.get(key, [])
        node = {"key": key, "label": label, "desc": desc, "status": "pending",
                "detail": "", "seconds": None}
        t_start = t_end = None
        if evs:
            last = evs[-1]
            msg = last.get("message") or ""
            node["detail"] = msg[:80]
            node["status"] = "skipped" if any(m in msg for m in SKIP_MARKS) else "done"
            t_start = _ts(evs[0].get("ts", ""))
            t_end = _ts(last.get("ts", ""))
        elif finished:
            # 任务已结束但没有该阶段事件 → 该阶段被跳过（未启用/无内容）
            node["status"] = "skipped"
        node["_t0"], node["_t1"] = t_start, t_end
        if key == cur and not finished:
            node["status"] = "running"
        if failed and key == cur:
            node["status"] = "failed"
        nodes.append(node)

    # 耗时 = 本阶段结束时刻 − 上一阶段结束时刻（顺序管线里就是本阶段的实际耗时）
    prev_end = None
    for n in nodes:
        t0 = n.pop("_t0", None)
        t1 = n.pop("_t1", None)
        if t1:
            start = prev_end or t0
            if start and t1 >= start:
                n["seconds"] = round((t1 - start).total_seconds(), 1)
            prev_end = t1
    return nodes


def summary(nodes: list[dict]) -> dict:
    done = sum(1 for n in nodes if n["status"] in ("done", "skipped"))
    total = len(nodes)
    return {"done": done, "total": total,
            "percent": round(done / total * 100) if total else 0}
