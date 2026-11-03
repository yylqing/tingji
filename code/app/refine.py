"""段落化润色：把碎片化的 ASR 句子合并成可读段落，再交给本地 LLM 顺滑。

设计要点：
- **段落保留真实时间范围**（首句 start → 末句 end），时间戳仍来自 ASR；
- 逐字稿（segments）不被覆盖，两套文本并存；
- 批量润色（多段一次调用 + JSON 对齐），降低长音频的 LLM 成本；
- 严格校验：不许造数字/人名、不许输出时间戳、不许加解释性括号。
"""
from __future__ import annotations

import json
import logging
import re

from . import textproc as tp

log = logging.getLogger("tingji.refine")

MEANING = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
META_START = ("（注", "(注", "由于", "抱歉", "作为AI", "我无法", "很抱歉")
META_HARD = ("请补充完整", "无法进行校对", "暂无可输出", "我将立即为您", "无法提取有效", "根据您提供")
# 模型爱加的"编辑说明"，一律剥掉
EDITORIAL = re.compile(r"[（(][^（）()]{0,20}(原文|原话|应为|疑似|误作|校对|注[:：])[^（）()]{0,20}[）)]")

SYS = (
    "你是中文语音转写的整理编辑。输入是口语转写的原始输出，标点混乱、有口水词、可能被停顿打断。"
    "请把它整理成通顺连贯的文字。"
)

SYS_NARRATIVE = (
    "你是中文个人叙述录音的整理编辑。说话人在情绪化地讲述自己的感受，中间可能因哽咽停顿被打断。"
    "请把被打断的话接起来，整理成通顺、保留第一人称与情绪的文字。"
)

RULES = """要求：
1. 去掉口水词（嗯/呃/啊/哦/那个/这个/就是说/然后就是）与无意义重复，保留所有实质信息；
2. 重新断句，使用自然的中文标点；**不要每两个词就加逗号**，按语义停顿断句；
3. 修正明显的同音错字与英文术语，优先采用下方术语表；不确定的**保持原样**，不要猜；
4. 输出连续文字，禁止分点、编号、小标题、总结；
5. **不要添加任何解释性括号**（如"（原文误作X）""（应为X）"）、不要加注、不要评论；
6. 不增加原文没有的事实、数字、人名；不要输出时间戳；
7. 尽量保留原句结构与说话人语气，只做必要的顺滑与纠错。"""

RULES_NARRATIVE = """要求：
1. 去掉口水词（嗯/呃/啊/哦/那个/这个/就是说）与无意义重复，保留所有实质内容与情绪表达；
2. **把被打断的碎片整理成连贯的 1–3 句话**：被哽咽/停顿切断的半句要接起来，重复与改口要合并，
   语义跳跃处按上下文补足连接词（如"所以""但是"），但**不得添加原文没有的事实**；
3. 重新断句，标点自然；不要每两个词就加逗号；
4. 保留第一人称口吻与原本的情绪，**不要改成会议书面语、不要添加心理描写或评价**；
5. 修正明显的同音错字；不确定的保持原样；
6. 不增加原文没有的事实；不要输出时间戳、不要加解释性括号。
只输出整理后的正文。"""

BATCH_RULES = """下面是同一场会议的若干段落，按 [序号] 标注。请逐段整理，输出 JSON 数组：
[{"i": 序号, "text": "整理后的段落"}]
要求：段数必须与输入一致，不得合并或拆分段落；其余规则同上；只输出 JSON。"""


SENT_END = "。！？!?"


def _ends_sentence(text: str) -> bool:
    return bool(text) and text.rstrip()[-1:] in SENT_END


def _is_emotional(seg: dict | None) -> bool:
    if not seg:
        return False
    return seg.get("event") == "Cry" or seg.get("emotion") in ("SAD", "FEARFUL")


def build_blocks(segs: list[dict], max_gap_ms: int = 2500, max_chars: int = 320,
                 sad_gap_ms: int = 9000, narrative: bool = False) -> list[dict]:
    blocks: list[dict] = []
    for s in segs:
        if s.get("quality") == "low_quality" or s.get("speaker_kind") in ("nav", "noise"):
            continue
        text = s.get("clean_text") or s.get("raw_text") or ""
        if not MEANING.search(text):
            continue
        # 未到句末就超长时，允许再并入一句（最多 1.35 倍），避免把句子拦腰切断
        soft = blocks and not _ends_sentence("".join(
            (x.get("clean_text") or x.get("raw_text") or "") for x in blocks[-1]["items"]))
        limit = max_chars * 1.35 if soft else max_chars
        prev_items = blocks[-1]["items"] if blocks else []
        gap_limit = sad_gap_ms if (_is_emotional(prev_items[-1] if prev_items else None)
                                   or _is_emotional(s)) else max_gap_ms
        if narrative:
            gap_limit = max(gap_limit, 12000)
            max_chars = max(max_chars, 600)
        if blocks and (blocks[-1]["spk"] == s.get("spk")
                       and s["start_ms"] - blocks[-1]["end_ms"] <= gap_limit
                       and blocks[-1]["chars"] + len(text) <= limit):
            b = blocks[-1]
            b["items"].append(s)
            b["end_ms"] = s["end_ms"]
            b["chars"] += len(text)
        else:
            blocks.append({"spk": s.get("spk"), "start_ms": s["start_ms"], "end_ms": s["end_ms"],
                           "items": [s], "chars": len(text)})
    return blocks


def _sanitize(fixed: str, raw: str) -> str:
    t = (fixed or "").strip()
    if not t or t.startswith(META_START) or any(h in t for h in META_HARD):
        return ""
    t = re.sub(r"^```\w*\s*|\s*```$", "", t).strip()
    t = re.sub(r"^[#\-\*\d.、\s]+", "", t)
    t = EDITORIAL.sub("", t).strip()
    t, _ = tp.strip_repeats(t)
    return t


def _numbers(s: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", s))


def _ok_single(fixed: str, raw: str) -> tuple[bool, str]:
    """单段重试的宽松校验：长度上限放宽到 1.35，其余不变。"""
    if not fixed or TIME_RE.search(fixed):
        return False, "empty_or_time"
    if len(fixed) / max(len(raw), 1) > 1.35:
        return False, "too_long"
    if _numbers(fixed) - _numbers(raw):
        return False, "new_numbers"
    return True, ""


def _ok(fixed: str, raw: str) -> tuple[bool, str]:
    if not fixed or TIME_RE.search(fixed):
        return False, "empty_or_time"
    if len(fixed) / max(len(raw), 1) > 1.25:
        return False, "too_long"
    if _numbers(fixed) - _numbers(raw):
        return False, "new_numbers"
    return True, ""


def _raw_text(block: dict) -> str:
    return "".join((s.get("clean_text") or s.get("raw_text") or "") for s in block["items"])


def _glossary(hot_rows, limit: int = 40) -> str:
    if not hot_rows:
        return ""
    terms = [r["term"] for r in sorted(hot_rows, key=lambda r: -r.get("weight", 1))][:limit]
    return "\n术语表（优先采用标准写法）：" + "、".join(terms)


def _parse_json_array(s: str) -> list | None:
    s = re.sub(r"^```(?:json)?|```$", "", (s or "").strip(), flags=re.M).strip()
    a, b = s.find("["), s.rfind("]")
    if a < 0 or b < 0:
        return None
    try:
        v = json.loads(s[a:b + 1])
        return v if isinstance(v, list) else None
    except Exception:
        return None


def refine(db, cfg, job_id: str, segs: list[dict], matcher=None,
           hot_rows=None, max_blocks: int = 0) -> list[dict]:
    """生成段落稿；失败块回退到规则清理后的原文。"""
    from .engines import LlmEngine

    style = _pick_style(cfg, segs)
    blocks = build_blocks(segs, cfg.refine.max_gap_ms, cfg.refine.max_chars,
                          getattr(cfg.refine, "sad_gap_ms", 9000),
                          narrative=(style == "narrative"))
    if max_blocks:
        blocks = blocks[:max_blocks]
    llm = LlmEngine(cfg.llm, cfg.refine.model)
    gloss = _glossary(hot_rows)
    texts: list[str] = [""] * len(blocks)
    polished = [False] * len(blocks)
    fails: dict[str, int] = {}

    i = 0
    while i < len(blocks):
        j, size = i, 0
        while j < len(blocks) and (j - i) < 4 and size < 900:
            size += len(_raw_text(blocks[j]))
            j += 1
        batch = blocks[i:j]
        results = _polish_batch(llm, batch, cfg, gloss, fails, style)
        for k, b in enumerate(batch):
            raw = _raw_text(b)
            fixed = results.get(k, "")
            if not fixed:
                fails["empty"] = fails.get("empty", 0) + 1
                continue
            ok, reason = _ok(fixed, raw)
            if ok:
                texts[i + k] = _sanitize(fixed, raw)
                polished[i + k] = bool(texts[i + k])
            else:
                fails[reason] = fails.get(reason, 0) + 1
        i = j
        if (i // 4) % 5 == 0:
            db.log(job_id, "info", "refined", f"段落 {i}/{len(blocks)}")

    # 第二遍：失败的块逐块重试（换个更宽松的单段路径）
    retry_idx = [i for i, t in enumerate(texts) if not t
                 and len(MEANING.findall(_raw_text(blocks[i]))) >= 10]
    for i in retry_idx:
        got = _polish_batch(llm, [blocks[i]], cfg, gloss, {}, style)
        fixed = got.get(0, "")
        if fixed:
            ok, _ = _ok_single(fixed, _raw_text(blocks[i]))
            if ok:
                texts[i] = _sanitize(fixed, _raw_text(blocks[i]))
                polished[i] = bool(texts[i])
    if retry_idx:
        db.log(job_id, "info", "refined", f"重试 {len(retry_idx)} 块，成功 {sum(1 for i in retry_idx if texts[i])}")

    out = []
    for idx, (b, text) in enumerate(zip(blocks, texts)):
        if not text:
            text = _raw_text(b)
        # 兜底：润色稿再过一遍规则清理 + ITN（回退段落里常有残留口水词）
        text, _ = tp.clean_text(text, True, True)
        if getattr(cfg, "text", None) and cfg.text.itn:
            from . import itn as itn_mod
            text = itn_mod.normalize(text)
        if matcher:
            text, _ = matcher.apply(text)
        emos = [x.get("emotion") for x in b["items"] if x.get("emotion")]
        events = [x.get("event") for x in b["items"] if x.get("event")]
        dom = None
        for e_ in ("SAD", "FEARFUL", "ANGRY", "SURPRISED", "HAPPY", "NEUTRAL"):
            if emos.count(e_) >= max(1, len(emos) // 2):
                dom = e_
                break
        out.append({"idx": idx, "start_ms": b["start_ms"], "end_ms": b["end_ms"],
                    "spk": b["spk"], "text": text, "polished": int(polished[idx]),
                    "emotion": dom, "event": ("Cry" if events.count("Cry") >= 1 else (events[0] if events else None))})
    # 极短碎片并入上一段，避免"哦，"这类孤立段落
    merged: list[dict] = []
    for p_ in out:
        if (len(MEANING.findall(p_["text"])) < 3 and merged
                and merged[-1]["spk"] == p_["spk"]):
            merged[-1]["text"] += p_["text"].strip()
            merged[-1]["end_ms"] = p_["end_ms"]
            merged[-1]["polished"] = int(merged[-1]["polished"] or p_["polished"])
        else:
            merged.append(p_)
    # 清洗后变空的段落（如整段只有"嗯"）直接丢弃
    cleaned = []
    for p_ in merged:
        if len(MEANING.findall(p_["text"])) < 1:
            continue
        cleaned.append(p_)
    merged = cleaned
    for i, p_ in enumerate(merged):
        p_["idx"] = i
    db.log(job_id, "info", "refined",
           f"段落 {len(merged)}，润色 {sum(p_['polished'] for p_ in merged)}，失败原因 {fails or '无'}")
    return merged


def _pick_style(cfg, segs: list[dict]) -> str:
    st = getattr(cfg.refine, "style", "auto")
    if st in ("meeting", "narrative"):
        return st
    spk = {s.get("spk") for s in segs if s.get("spk") is not None}
    dur = [(s["end_ms"] - s["start_ms"]) / 1000 for s in segs]
    med = sorted(dur)[len(dur) // 2] if dur else 0
    cries = sum(1 for s in segs if s.get("event") == "Cry" or s.get("emotion") in ("SAD", "FEARFUL"))
    ratio = cries / max(len(segs), 1)
    return "narrative" if (len(spk) <= 2 and (ratio >= 0.03 or med < 3)) else "meeting"


def _polish_batch(llm, batch: list[dict], cfg, gloss: str, fails: dict,
                  style: str = "meeting") -> dict[int, str]:
    rules = RULES_NARRATIVE if style == "narrative" else RULES
    system = SYS_NARRATIVE if style == "narrative" else SYS
    if len(batch) == 1:
        b = batch[0]
        raw = _raw_text(b)
        if len(MEANING.findall(raw)) < 6:
            fails["too_short"] = fails.get("too_short", 0) + 1
            return {}
        prompt = f"{rules}{gloss}\n\n原文：\n{raw}"
        try:
            out = llm.chat(prompt, system=system, max_tokens=900, temperature=0.2)
        except Exception as e:
            fails["llm_error"] = fails.get("llm_error", 0) + 1
            return {}
        fixed = _sanitize(out, raw)
        ok, reason = _ok(fixed, raw)
        if not ok:
            fails[reason] = fails.get(reason, 0) + 1
            return {}
        return {0: fixed}

    # 多段批处理：标号 + JSON 对齐
    parts = []
    for k, b in enumerate(batch):
        parts.append(f"[{k}] {_raw_text(b)}")
    prompt = f"{rules}\n{BATCH_RULES}{gloss}\n\n输入：\n" + "\n".join(parts)
    try:
        out = llm.chat(prompt, system=system, max_tokens=1800, temperature=0.2)
    except Exception:
        fails["llm_error"] = fails.get("llm_error", 0) + 1
        return {}
    arr = _parse_json_array(out)
    if not arr:
        fails["json_parse"] = fails.get("json_parse", 0) + 1
        return {}
    got: dict[int, str] = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        try:
            k = int(item.get("i", -1))
        except Exception:
            continue
        if 0 <= k < len(batch):
            got[k] = _sanitize(str(item.get("text", "")), _raw_text(batch[k]))
    if len(got) != len(batch):
        fails["json_mismatch"] = fails.get("json_mismatch", 0) + 1
        # 已拿到的照样用，缺的回退原文
    return got
