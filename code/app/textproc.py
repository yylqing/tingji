"""文本处理：规则清理、热词替换、质量标记、时间戳守卫、发言块合并。"""
from __future__ import annotations

import re
from typing import Iterable

HAN_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

# 独立成句的填充词
LEAD_FILLERS = ("嗯", "呃", "啊", "哦", "唔", "哎", "唉", "诶", "那个", "这个", "就是说", "然后就是", "怎么说呢")
# 句末语气词（啊/吧/呢/哦）承载语气，按设计保留；只清理真正的填充音
TAIL_FILLERS = ("嗯", "呃")
# 只在这些边界上删，避免误伤词内字
BOUNDARY = "，。！？；、,.!?;：: \n"


def strip_fillers(text: str) -> tuple[str, int]:
    """删除句首/句尾独立填充词与标记词。返回 (新文本, 改动数)。"""
    n = 0
    changed = True
    while changed:
        changed = False
        t = text.lstrip()
        for w in LEAD_FILLERS:
            if t.startswith(w):
                rest = t[len(w):]
                # 必须后接标点/空白，才算独立成分
                if not rest or rest[0] in BOUNDARY or rest[0] in LEAD_FILLERS:
                    text = rest.lstrip(BOUNDARY + " ")
                    n += 1
                    changed = True
                    break
        t = text.rstrip()
        for w in TAIL_FILLERS:
            if t.endswith(w):
                text = t[: -len(w)].rstrip(BOUNDARY + " ")
                n += 1
                changed = True
                break
    return text, n


_REPEAT_CHAR = re.compile(r"([\u4e00-\u9fff])\1{1,}")
_REPEAT_PHRASE = re.compile(r"([\u4e00-\u9fff]{2,4})\1+")


def strip_repeats(text: str) -> tuple[str, int]:
    """合并叠字与短句重复：我我觉得→我觉得，对对对→对。"""
    n = 0
    text, k = _REPEAT_CHAR.subn(r"\1", text); n += k
    text, k = _REPEAT_PHRASE.subn(r"\1", text); n += k
    return text, n


def clean_text(text: str, do_fillers: bool = True, do_repeats: bool = True) -> tuple[str, dict]:
    stat = {"fillers": 0, "repeats": 0}
    if do_fillers:
        text, stat["fillers"] = strip_fillers(text)
    if do_repeats:
        text, stat["repeats"] = strip_repeats(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, stat


# ---------------------------------------------------------------- 热词
class HotwordMatcher:
    """L2 文本级热词：拼音模糊 + 显式别名（复用 FunASR 实现）。"""

    def __init__(self, rows: Iterable[dict], threshold: float = 0.85):
        from funasr.utils.postprocess_hotwords import build_postprocess_hotword_matcher
        mapping: dict[str, str] = {}
        for r in rows:
            if not r.get("enabled", 1):
                continue
            term = r["term"]
            mapping[term] = term
            for a in r.get("aliases") or []:
                if a:
                    mapping[a] = term
        self.rows = list(rows)
        self._m = build_postprocess_hotword_matcher(mapping or {"__none__": "__none__"},
                                                    postprocess_hotword_threshold=threshold) if mapping else None

    def apply(self, text: str) -> tuple[str, list[dict]]:
        if not self._m:
            return text, []
        out, matches = self._m.apply_text(text)
        return out, [{"original": m.original, "replacement": m.replacement, "score": round(m.score, 3)} for m in matches]


# ---------------------------------------------------------------- 质量
def quality_flag(text: str, dur_sec: float) -> tuple[str, float]:
    rate = len(text) / dur_sec if dur_sec > 0 else 0.0
    if dur_sec > 10 and rate < 1.0:
        return "low_quality", rate
    if dur_sec > 10 and rate < 3.0:
        return "suspect", rate
    return "ok", rate


# ---------------------------------------------------------------- 时间戳守卫
def check_timeline(segs: list[dict]) -> list[str]:
    errs = []
    for i, s in enumerate(segs):
        if s["end_ms"] <= s["start_ms"]:
            errs.append(f"seg{i} 时长为非正: {s['start_ms']}-{s['end_ms']}")
        if i and s["start_ms"] < segs[i - 1]["start_ms"]:
            errs.append(f"seg{i} 起点早于前一段")
        if i and s["start_ms"] < segs[i - 1]["end_ms"]:
            errs.append(f"seg{i} 与上一段重叠")
    return errs


def blocks_of(segs: list[dict], max_gap_ms: int = 2000) -> list[dict]:
    """合并同一说话人的连续句子为发言块（用于展示与纠错分批）。"""
    out: list[dict] = []
    for s in segs:
        if out and out[-1]["spk"] == s.get("spk") and s["start_ms"] - out[-1]["end_ms"] <= max_gap_ms:
            out[-1]["items"].append(s)
            out[-1]["end_ms"] = s["end_ms"]
        else:
            out.append({"spk": s.get("spk"), "start_ms": s["start_ms"], "end_ms": s["end_ms"], "items": [s]})
    return out


_TAIL_PUNCT = "。！？!?；;，,"


def trim_uncertain_tail(text: str, boundary: str | None) -> tuple[str, bool]:
    """切点不是真正的句末时，去掉末尾那个"补出来的"标点。

    对应 Brevia 的 boundary 语义：只有 endpoint 可信；pause / cut 的句末标点是
    模型对被截断音频的补全。返回 (处理后的文本, 是否改动)。
    """
    if not text or boundary not in ("pause", "cut"):
        return text, False
    t = text.rstrip()
    if t and t[-1] in _TAIL_PUNCT:
        return t[:-1].rstrip(), True
    return text, False


def split_by_time(words: list[dict] | None, start_ms: int, end_ms: int,
                  at_ms: int, text: str = "") -> tuple[str, str]:
    """把一段话按时间点切成左右两半。

    优先用词级时间戳（Paraformer 精确、SenseVoice 估算也可用）；没有词级时间就按
    时长比例切文字。时间点必须落在段内，否则抛 ValueError。
    """
    if not (start_ms < at_ms < end_ms):
        raise ValueError("拆分点必须在该段内部")
    if words:
        left = [w for w in words if int(w.get("e", w.get("s", 0))) <= at_ms]
        right = [w for w in words if int(w.get("e", w.get("s", 0))) > at_ms]
        if left and right:
            return ("".join(w.get("t", "") for w in left),
                    "".join(w.get("t", "") for w in right))
    # 没有词级时间：按时间比例切
    t = text or ""
    if not t:
        return "", ""
    ratio = (at_ms - start_ms) / max(1, end_ms - start_ms)
    cut = max(1, min(len(t) - 1, round(len(t) * ratio)))
    # 尽量落在标点上，切得自然一点
    for i in range(cut, min(len(t), cut + 6)):
        if t[i - 1] in "，。！？；、,.!?; ":
            return t[:i], t[i:]
    return t[:cut], t[cut:]
