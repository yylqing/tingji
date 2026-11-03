"""纪要生成：从整理稿（段落）产出摘要 + 章节，供 Obsidian/Web 使用。

参考商业产品做法：摘要吃"润色稿"，但每条结论可回溯到段落时间。
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger("tingji.summarize")

SYS = "你是资深内容整理助手。你会收到一段录音的分段整理稿，请先判断它是什么类型的记录，再输出结构化纪要。"

DOC_TYPES = ["会议纪要", "日常对话", "重要时刻", "访谈", "讲座", "备忘", "其他"]

PROMPT = """请根据下面的整理稿，先判断这段录音是什么内容，再输出 JSON 对象：
{
  "doc_type": "会议纪要|日常对话|重要时刻|访谈|讲座|备忘|其他",
  "title": "整篇内容的标题（≤20字，概括实际谈了什么，不要写成文件名）",
  "topic": "主题（≤20字）",
  "summary": "全文摘要（120–200字，覆盖主要议题与结论）",
  "chapters": [{"title": "章节标题（≤12字）", "para": 起始段落序号, "summary": "该章节一句话（≤40字）"}],
  "decisions": ["明确达成的决定（没有就空数组）"],
  "todos": [{"task": "要做的事", "owner": "负责人（稿中出现的名字，没有写未知）"}],
  "keywords": ["关键词", "3-6 个"]
}

要求：
0. **doc_type 判断依据（先看内容主题，再看人数）**：
   - 围绕工作/业务/客户/项目/技术/决策/数据 → **会议纪要**（哪怕只有一个人讲、哪怕是两个人一对一沟通）
   - 生活、家庭、情感、朋友闲聊、购物出行 → **日常对话**
   - 有明显情绪与纪念意义（宝宝胎动/表白/生日/告别/争吵和解）→ **重要时刻**
   - 单人系统讲授知识、课程 → **讲座**；一问一答的采访 → **访谈**
   - 极短的提醒/待办记录 → **备忘**；都不像 → **其他**
   有明显情绪、纪念意义（表白/生日/宝宝胎动/告别）→ 重要时刻；一问一答的采访 → 访谈；
   单人长时间讲述知识 → 讲座；简短记录/备忘 → 备忘；都不像 → 其他。
1. chapters 按时间顺序，3–6 个，覆盖全文；
2. summary 与章节只依据稿件内容，**不要编造**没有的事实、数字、人名；
3. 稿件里没有明确决议/待办时，对应数组留空，不要硬凑；
4. 只输出 JSON，不要解释、不要代码块标记。

整理稿（序号可作 para 引用）：
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


def _fmt_para(p: dict) -> str:
    return f"[{p['idx']}] {p['text']}"


def build_body(paragraphs: list[dict], max_chars: int = 6000) -> tuple[str, int]:
    """拼整理稿；超长时按比例抽样，保证首尾与中段都有。"""
    lines = [_fmt_para(p) for p in paragraphs]
    body = "\n".join(lines)
    if len(body) <= max_chars:
        return body, 1
    # 分层抽样：每 k 段取 1 段，但保证每段摘要足够
    k = max(2, len(lines) // 60)
    keep = [ln for i, ln in enumerate(lines) if i % k == 0 or i >= len(lines) - 3]
    return "\n".join(keep), k


def summarize(db, cfg, job_id: str, paragraphs: list[dict], names: dict[int, str] | None = None) -> dict | None:
    from .engines import LlmEngine

    if not paragraphs:
        return None
    llm = LlmEngine(cfg.llm, cfg.summary.model)
    body, step = build_body(paragraphs, cfg.summary.max_input_chars)
    speakers = {p.get("spk") for p in paragraphs if p.get("spk") is not None}
    dur_min = (int(paragraphs[-1]["end_ms"]) - int(paragraphs[0]["start_ms"])) / 60000 if paragraphs else 0
    body = f"（录音信息：{len(speakers)} 位说话人，约 {dur_min:.0f} 分钟）\n" + body
    try:
        raw = llm.chat(PROMPT.replace("{body}", body), system=SYS, max_tokens=1200, temperature=0.2)
    except Exception as e:
        db.log(job_id, "warning", "summarized", f"纪要生成失败：{e}")
        return None
    data = _parse(raw)
    if not data:
        db.log(job_id, "warning", "summarized", "纪要 JSON 解析失败")
        return None
    chapters = []
    for c in (data.get("chapters") or [])[:6]:
        if not isinstance(c, dict):
            continue
        try:
            pidx = int(c.get("para", 0)) * step
        except Exception:
            pidx = 0
        pidx = max(0, min(pidx, len(paragraphs) - 1))
        chapters.append({
            "title": str(c.get("title", ""))[:24],
            "summary": str(c.get("summary", ""))[:80],
            "para": pidx,
            "start_ms": paragraphs[pidx]["start_ms"],
        })
    doc_type = str(data.get("doc_type") or "").strip()
    if doc_type not in DOC_TYPES:
        doc_type = "其他"
    out = {
        "doc_type": doc_type,
        "title": str(data.get("title") or "").strip()[:40],
        "topic": str(data.get("topic", ""))[:40],
        "summary": str(data.get("summary", ""))[:400],
        "chapters": chapters,
        "decisions": [str(x)[:120] for x in (data.get("decisions") or [])][:6],
        "todos": [{"task": str(t.get("task", ""))[:120],
                   "owner": str(t.get("owner", "未知"))[:20]}
                  for t in (data.get("todos") or []) if isinstance(t, dict)][:8],
        "keywords": [str(x)[:20] for x in (data.get("keywords") or [])][:8],
        "model": cfg.summary.model,
        "sampled_step": step,
    }
    db.log(job_id, "info", "summarized", f"章节 {len(chapters)}，待办 {len(out['todos'])}")
    return out

