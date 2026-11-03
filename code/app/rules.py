"""正则纠错规则（借鉴 CapsWriter-Offline 的 hot-rule.txt）。

格式：每行 `查找模式 = 替换式`，`#` 开头为注释；两侧空格会去掉。
- 查找模式是正则，替换式支持 \\1 / \\2 反向引用；
- 用来处理"固定格式"的错误：单位、邮箱、网址、重复出现的同音错字。

为什么需要它：热词表只能替换"词"，处理不了
`(艾特)\\s*(\\w+)\\s*(点)\\s*(\\w+) = @$2.$4` 这种结构化的错法。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

log = logging.getLogger("tingji.rules")


def parse(text: str) -> tuple[list[tuple[re.Pattern, str]], list[str]]:
    """把规则文本解析成 (编译好的规则, 错误列表)。坏规则只跳过，不抛异常。"""
    rules: list[tuple[re.Pattern, str]] = []
    errors: list[str] = []
    for i, line in enumerate((text or "").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(.*?)\s=\s(.*)$", line)   # 优先按「 = 」分
        if m:
            lhs, rhs = m.group(1), m.group(2)
        elif "=" in line:
            lhs, rhs = line.split("=", 1)
        else:
            errors.append(f"第 {i} 行：缺少 '='")
            continue
        lhs, rhs = lhs.strip(), rhs.strip()
        if not lhs:
            errors.append(f"第 {i} 行：查找模式为空")
            continue
        try:
            rules.append((re.compile(lhs), rhs))
        except re.error as e:
            errors.append(f"第 {i} 行：正则无效（{e}）")
    return rules, errors


class RuleCorrector:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else None
        self.rules: list[tuple[re.Pattern, str]] = []
        self.errors: list[str] = []
        self.reload()

    def reload(self) -> None:
        if not self.path or not self.path.exists():
            self.rules, self.errors = [], []
            return
        self.rules, self.errors = parse(self.path.read_text(encoding="utf-8"))
        if self.errors:
            log.warning("纠错规则有 %d 条无效：%s", len(self.errors), self.errors[:3])

    def apply(self, text: str) -> tuple[str, int]:
        """返回 (纠错后文本, 命中条数)。"""
        if not text or not self.rules:
            return text, 0
        hits = 0
        out = text
        for pat, rep in self.rules:
            try:
                new, n = pat.subn(rep, out)
            except re.error as e:
                log.warning("规则 %s 替换失败：%s", pat.pattern[:30], e)
                continue
            if n and new != out:
                hits += n
                out = new
        return out, hits
