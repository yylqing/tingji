"""中文 ITN（逆文本规范化）：把口语数字写成阿拉伯数字。

保守策略：**只在有明确单位/上下文时转换**，避免把"三心二意""一个"这类改坏。
覆盖：技术量词（毫秒/核/卡/台/GB…）、金额、百分比、年份、时刻、万/亿。
"""
from __future__ import annotations

import re

CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "壹": 1, "二": 2, "两": 2, "贰": 2, "三": 3, "叁": 3,
            "四": 4, "肆": 4, "五": 5, "伍": 5, "六": 6, "陆": 6, "七": 7, "柒": 7,
            "八": 8, "捌": 8, "九": 9, "玖": 9}
CN_UNIT = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
CN_BIG = {"万": 10000, "亿": 100000000}
NUM_CHARS = "".join(CN_DIGIT) + "".join(CN_UNIT) + "".join(CN_BIG)

# 只有这些单位才允许把"小数字"（<10）也转成阿拉伯数字
TECH_UNITS = ("毫秒", "微秒", "纳秒", "秒", "分钟", "小时", "天", "周", "个月", "年",
              "GB", "TB", "MB", "KB", "G", "T", "M", "兆", "核", "卡", "台", "倍",
              "元", "块", "%", "度", "瓦", "千瓦", "米", "公里",
              "月", "日", "号", "张", "个百分点", "倍频", "万", "亿", "美元", "美金", "块钱")


def cn2int(s: str) -> int | None:
    """解析中文数字：三百二十一 → 321；两千零一 → 2001；十五 → 15。"""
    if not s or any(c not in NUM_CHARS for c in s):
        return None
    # 纯数字串（如 二零二六）
    if all(c in CN_DIGIT for c in s):
        return int("".join(str(CN_DIGIT[c]) for c in s))
    total, section, num = 0, 0, 0
    for ch in s:
        if ch in CN_DIGIT:
            num = CN_DIGIT[ch]
        elif ch in CN_UNIT:
            unit = CN_UNIT[ch]
            section += (num if num else 1) * unit
            num = 0
        elif ch in CN_BIG:
            big = CN_BIG[ch]
            section = (section + (num if num else 0)) * big
            total += section
            section, num = 0, 0
        else:
            return None
    return total + section + num


_NUM = f"[{NUM_CHARS}]+"
_RE_PERCENT = re.compile(rf"百分之({_NUM})")
_RE_POINT = re.compile(rf"(?<!点)({_NUM})个?百分点")
_RE_YEAR = re.compile(r"([零〇一二三四五六七八九]{4})年")
_RE_CLOCK = re.compile(r"(上午|下午|晚上|中午|凌晨)?([一二三四五六七八九十]{1,2})点(半|[一二三四五六七八九十]{1,2}分?)?")
_RE_DECIMAL = re.compile(rf"({_NUM})点([{''.join(CN_DIGIT)}])(?=倍|个|成|万|亿|元|米|秒|公斤|吨)")
_GENERIC_UNITS = ("个", "只", "位", "条", "件", "家", "套", "瓶", "杯", "包",
                  "页", "行", "篇", "次", "天", "年", "月", "日", "号", "岁", "层", "份")
_ALL_UNITS = tuple(dict.fromkeys(list(TECH_UNITS) + list(_GENERIC_UNITS)))
_RE_UNIT = re.compile(rf"(?<!点)({_NUM})(?=(?:{'|'.join(sorted(_ALL_UNITS, key=len, reverse=True))}))")
_RE_DATE = re.compile(rf"(?<!点)({_NUM})(?=[月日号])")

# 约数/范围：十五六 → 15~16；三五百 → 300~500；二三十 → 20~30；三四 → 3~4
_UNIT = ("个|只|人|位|次|天|年|月|日|号|台|张|核|卡|条|件|倍|成|元|块|"
         "米|公里|分钟|小时|毫秒|秒|周|岁|层|斤|公斤|页|行|篇|家|套|瓶|杯|包")
_D = "一二两三四五六七八九"
_RE_NUM_UNIT = re.compile(rf"([{_D}十百千万]{{2,4}})(?=(?:{_UNIT}))")


def _range_of(word: str) -> str | None:
    """判断一个中文数字串是不是"约数/范围"，是则返回 `lo~hi`。"""
    d = CN_DIGIT
    n = len(word)
    if n == 2 and word[0] in d and word[1] in d:
        a, b = d[word[0]], d[word[1]]
        return f"{a}~{b}" if 0 < a < b else None            # 三四 → 3~4
    if n == 3 and word[0] in d and word[1] in d and word[2] == "百":
        return f"{d[word[0]] * 100}~{d[word[1]] * 100}"    # 三五百 → 300~500
    if n == 3 and word[0] == "十" and word[1] in d and word[2] in d:
        lo = 10 + d[word[1]]
        return f"{lo}~{lo + 1}"                            # 十五六 → 15~16
    if n == 3 and word[0] in d and word[1] in d and word[2] == "十":
        return f"{d[word[0]] * 10}~{d[word[1]] * 10}"      # 二三十 → 20~30（字形是 二/三/十）
    if n == 3 and word[0] in d and word[1] == "千" and word[2] in d:
        base = d[word[0]] * 1000
        return f"{base}~{base + d[word[2]] * 100}"         # 一千二 → 1000~1200
    return None


def _range_sub(text: str) -> str:
    def repl(m):
        r = _range_of(m.group(1))
        return r if r else m.group(0)
    return _RE_NUM_UNIT.sub(repl, text)


def _fmt(n: int) -> str:
    if n >= 100000000 and n % 100000000 == 0:
        return f"{n // 100000000}亿"
    if n >= 10000 and n % 10000 == 0:
        return f"{n // 10000}万"
    return str(n)


def normalize(text: str) -> str:
    if not text:
        return text

    def pct(m):
        n = cn2int(m.group(1))
        return f"{_fmt(n)}%" if n is not None else m.group(0)

    def point(m):
        n = cn2int(m.group(1))
        return f"{_fmt(n)}个百分点" if n is not None else m.group(0)

    def year(m):
        n = cn2int(m.group(1))
        return f"{_fmt(n)}年" if n is not None else m.group(0)

    def clock(m):
        ap, hh, mm = m.group(1) or "", m.group(2), m.group(3) or ""
        h = cn2int(hh)
        if h is None or h > 24:
            return m.group(0)
        if not (ap or mm):
            return m.group(0)          # 光"三点"多半不是时刻
        if not ap and not (mm == "半" or mm.endswith("分")):
            return m.group(0)          # 无时段标记且无"分/半"，多半是小数（三点五倍）
        suffix = "30" if mm == "半" else ""
        if mm and mm != "半":
            mi = cn2int(mm.rstrip("分"))
            suffix = f"{mi:02d}" if mi is not None else ""
        if not suffix:
            return m.group(0)
        return f"{ap}{_fmt(h)}:{suffix}"

    def unit(m):
        raw = m.group(1)
        n = cn2int(raw)
        if n is None:
            return m.group(0)
        tail = text[m.end():m.end() + 3]
        # <10 的数字只在技术/计量单位前转换，避免"一个""三个"这类口语被改硬
        if n < 10 and not any(tail.startswith(u) for u in TECH_UNITS):
            return m.group(0)
        return _fmt(n)

    out = _range_sub(text)
    out = _RE_PERCENT.sub(pct, out)
    out = _RE_POINT.sub(point, out)
    out = _RE_YEAR.sub(year, out)
    def decimal(m):
        a, b = cn2int(m.group(1)), CN_DIGIT.get(m.group(2))
        return f"{a}.{b}" if a is not None and b is not None else m.group(0)
    out = _RE_DECIMAL.sub(decimal, out)
    out = _RE_CLOCK.sub(clock, out)
    out = _RE_UNIT.sub(unit, out)
    # 日期：十月三十一号 → 10月31号
    def date(m):
        n = cn2int(m.group(1))
        return _fmt(n) if n is not None else m.group(0)
    out = _RE_DATE.sub(date, out)
    return out
