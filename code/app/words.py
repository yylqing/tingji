"""词级时间戳：把 ASR 的 token 时间或整句时间，折算成"每个词/字什么时候说的"。

两种来源：
1. **精确**：Paraformer 的 `raw_text`（空格分隔 token）+ `timestamp`（每个 token 的 [s,e]），
   一一对应，直接用；
2. **插值**：SenseVoice 不返回 token 时间（实测 timestamp 为空），按字符数在句内均摊，
   标注为 `approx`，UI 上与精确时间区分显示。

注意：只对**原始转写**（raw_text）有意义。清洗稿删了口水词、整理稿是重写的，
词与音频不再一一对应——这是行业普遍做法（WhisperX 也是对齐原始文本）。
"""
from __future__ import annotations

import re

_MEAN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
# 英文单词、数字、单个汉字各算一个单位；标点并入前一个单位
_UNIT = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|[\u4e00-\u9fff]")


def from_tokens(tokens: list[str], timestamps: list[list[int]],
                offset_ms: int = 0) -> list[dict]:
    """精确词级：token 与 timestamp 一一对应。"""
    out = []
    for tok, ts in zip(tokens, timestamps or []):
        tok = (tok or "").strip()
        if not tok:
            continue
        try:
            s, e = int(ts[0]) + offset_ms, int(ts[1]) + offset_ms
        except Exception:
            continue
        out.append({"t": tok, "s": s, "e": max(e, s), "approx": 0})
    return out


def interpolate(text: str, start_ms: int, end_ms: int) -> list[dict]:
    """按字符数在句内均摊时间（SenseVoice 无 token 时间时使用）。"""
    text = (text or "").strip()
    if not text or end_ms <= start_ms:
        return []
    units = _UNIT.findall(text)
    if not units:
        return []
    span = end_ms - start_ms
    step = span / len(units)
    out = []
    for i, u in enumerate(units):
        s = start_ms + int(i * step)
        e = start_ms + int((i + 1) * step)
        out.append({"t": u, "s": s, "e": e, "approx": 1})
    return out


def join(words: list[dict]) -> str:
    """把词拼回文本；英文/数字之间补空格。"""
    out = []
    for i, w in enumerate(words):
        t = w.get("t", "")
        if i and re.match(r"^[A-Za-z0-9]", t) and out and re.search(r"[A-Za-z0-9]$", str(out[-1])):
            out.append(" ")
        out.append(t)
    return "".join(out)


def is_approx(words: list[dict]) -> bool:
    return bool(words) and all(w.get("approx") for w in words)
