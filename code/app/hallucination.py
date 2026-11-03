"""ASR 幻觉过滤：识别"模型在没听到人话时编出来的内容"。

灵感来自 WhisprRT 的 `HALLUCINATION_PATTERNS`：Whisper 在静音/噪声/音乐上会稳定地
吐出固定套话（"优优独播剧场""请不吝点赞 订阅 转发"）。FunASR 系模型也有类似行为，
表现为：整段重复、纯标点、与音频长度严重不匹配的短文本。

这里做三件事：
1. 黑名单匹配（中英文常见 ASR 幻觉套话）；
2. 段内异常重复检测（同一 3 字片段出现 >3 次）；
3. 纯标点/纯符号段。
"""
from __future__ import annotations

import re

# 常见 ASR 幻觉套话（Whisper 系最典型；FunASR 偶发）
PATTERNS = [
    r"优优独播剧场", r"YoYo\s*Television", r"请不吝点赞", r"订阅.*转发.*打赏",
    r"明镜与点点", r"支持明镜", r"独播剧场", r"点赞.*订阅", r"转发.*打赏",
    r"感谢.{0,4}观看", r"关注.{0,4}频道", r"网友们.{0,6}支持",
    r"字幕由.{0,8}提供", r"本字幕由", r"amara\.org", r"字幕组",
    r"感谢您的收看", r"下期再见", r"谢谢观看",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in PATTERNS]
_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)
_TANDEM = re.compile(r"(.{1,6}?)\1{2,}")
_MEANING = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


def has_pattern(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    return any(p.search(t) for p in _COMPILED)


def is_repetitive(text: str, min_len: int = 3, max_count: int = 3) -> bool:
    """异常重复：单字长串（≥6）或同一片段重叠出现 >max_count 次。

    注意 str.count 是非重叠计数，"对对对对对对对对对"（9 字）只能数出 3 组，
    所以必须自己按重叠位置统计，否则漏判。
    """
    t = (text or "").strip()
    if len(t) < 6:
        return False
    run = best = 1
    for i in range(1, len(t)):
        run = run + 1 if t[i] == t[i - 1] else 1
        best = max(best, run)
    if best >= 6:
        return True
    # 只抓"连续重复"（tandem repeat）——ASR 幻觉的典型形态；
    # 分散重复（"没有不喜欢他…没有不喜欢他"）是真实语气，不能误杀。
    m = _TANDEM.search(t)
    return bool(m and len(m.group(0)) >= 4)


def is_punct_only(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and not _MEANING.search(t) and bool(_PUNCT_ONLY.match(t))


def is_hallucination(text: str, min_chars: int = 1) -> tuple[bool, str]:
    """返回 (是否幻觉, 原因)。

    注意：**短句不算幻觉**。"嗯。""是。""好。"是真实应答，误杀会丢对话内容；
    短而空的段由 `textproc.quality_flag`（字符率）负责。
    """
    t = (text or "").strip()
    if not t:
        return True, "empty"
    if len(_MEANING.findall(t)) < min_chars:
        return True, "empty"
    if is_punct_only(t):
        return True, "punct_only"
    if has_pattern(t):
        return True, "boilerplate"
    # 重复**不整段丢弃**：真口吃也是内容，交给 textproc.strip_repeats 清理即可
    return False, ""
