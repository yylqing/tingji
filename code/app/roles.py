"""说话人角色消歧：从整理稿里推断"说话人N 是谁"（姓名/角色）。

只在有明确证据时赋值（自我介绍、他人称呼、职位上下文），否则留空。
声纹库已命名的说话人优先，不被覆盖。
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger("tingji.roles")

SYS = "你是会议纪要编辑，擅长从对话内容判断说话人身份。"
PROMPT = """下面是一场会议的分段整理稿，每段标注了说话人编号（S0、S1…）。

请判断每个说话人最可能的**姓名**或**角色**：
- 只有稿件里有明确依据时才填（自我介绍、别人称呼、职位描述等）；
- 没有依据就留空字符串，**不要猜**；
- 姓名用稿件中出现的写法；角色用简短词（如"客户方""技术负责人""主持人"）。

输出 JSON：
{{"speakers": [{{"spk": 0, "name": "", "role": "", "evidence": "依据原文片段（≤20字）"}}]}}

只输出 JSON，不要解释。

整理稿：
{body}"""


def _parse(s: str) -> dict | None:
    s = re.sub(r"^```(?:json)?|```$", "", (s or "").strip(), flags=re.M).strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < 0:
        return None
    try:
        v = json.loads(s[a:b + 1])
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def detect(db, cfg, job_id: str, paragraphs: list[dict],
           existing: dict[int, str] | None = None) -> dict[int, dict]:
    """返回 {spk: {"name":..., "role":..., "evidence":...}}，仅含有依据的项。"""
    from .engines import LlmEngine

    existing = existing or {}
    if not paragraphs:
        return {}
    lines = [f"S{p.get('spk', -1)}: {p['text']}" for p in paragraphs]
    body = "\n".join(lines)
    if len(body) > cfg.roles.max_input_chars:
        step = max(2, len(lines) // 80)
        body = "\n".join(ln for i, ln in enumerate(lines) if i % step == 0)
    llm = LlmEngine(cfg.llm, cfg.roles.model)
    try:
        raw = llm.chat(PROMPT.replace("{body}", body), system=SYS, max_tokens=600, temperature=0.1)
    except Exception as e:
        db.log(job_id, "warning", "roles", f"角色识别失败：{e}")
        return {}
    data = _parse(raw)
    if not data:
        db.log(job_id, "warning", "roles", "角色识别 JSON 解析失败")
        return {}
    # 防幻觉：evidence 必须是稿件原文的子串，否则视为臆测丢弃
    corpus = "".join(p.get("text", "") for p in paragraphs)
    def _norm(x: str) -> str:
        return re.sub(r"[\s，。！？、,.!?;：:]", "", x or "")

    corpus_n = _norm(corpus)
    out: dict[int, dict] = {}
    for item in data.get("speakers") or []:
        if not isinstance(item, dict):
            continue
        try:
            k = int(item.get("spk", -1))
        except Exception:
            continue
        if k < 0 or k in existing:          # 声纹已命名，跳过
            continue
        name = str(item.get("name") or "").strip()[:20]
        role = str(item.get("role") or "").strip()[:20]
        ev = str(item.get("evidence") or "").strip()[:40]
        if not name and not role:
            continue
        ev_n = _norm(ev)
        if len(ev_n) >= 4 and ev_n not in corpus_n:
            db.log(job_id, "info", "roles", f"丢弃无依据的角色：{name or role}（{ev}）")
            continue
        out[k] = {"name": name, "role": role, "evidence": ev, "applied": False}
    db.log(job_id, "info", "roles", f"候选 {len(out)} 人", out)
    return out


def display_names(roles: dict[int, dict], existing: dict[int, str]) -> dict[int, str]:
    """把角色结果合成为显示名：声纹姓名 > 稿件姓名 > 角色 > 空。"""
    out = dict(existing)
    for k, r in roles.items():
        if k in out and out[k]:
            continue
        # 只有"姓名"才替换显示名；角色属于推断，不顶替"说话人N"
        if r.get("name"):
            out[k] = r["name"]
    return out
