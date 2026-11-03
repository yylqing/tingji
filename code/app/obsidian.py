"""Obsidian 归档：纪要笔记 + 逐字稿笔记 + 附件（m4a/srt/txt），原子写、幂等。"""
from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

BAD_NAME = re.compile(r'[\\/:*?"<>|#^\[\]%]')


def safe_name(s: str, maxlen: int = 80) -> str:
    s = BAD_NAME.sub(" ", s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s[:maxlen].strip() or "未命名"


def hms(ms: int) -> str:
    s = int(ms // 1000)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def srt_time(ms: int) -> str:
    h, rem = divmod(int(ms), 3600000)
    m, rem = divmod(rem, 60000)
    s, ms_ = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_:03d}"


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def build_srt(segs: list[dict], speaker_names: dict[int, str]) -> str:
    out = []
    for i, s in enumerate(segs, 1):
        spk = s.get("spk", -1)
        who = speaker_names.get(spk) or (f"说话人{spk + 1}" if spk >= 0 else "未知")
        text = s.get("final_text") or s.get("clean_text") or s.get("raw_text") or ""
        out.append(f"{i}\n{srt_time(s['start_ms'])} --> {srt_time(s['end_ms'])}\n[{who}] {text}\n")
    return "\n".join(out)


DOC_TYPES = ["会议纪要", "日常对话", "重要时刻", "访谈", "讲座", "备忘", "其他"]


def classify(job: dict, segs: list[dict], duration_sec: float,
             summary: dict | None = None) -> str:
    """判断这条录音是什么类型的记录。

    优先用 LLM 给出的 doc_type；没有（未启用纪要/失败）时按结构启发式判断。
    """
    if summary and summary.get("doc_type") in DOC_TYPES:
        return summary["doc_type"]
    spk = {s.get("spk") for s in segs if s.get("spk") is not None and s.get("spk", -1) >= 0}
    n_spk = len(spk)
    cries = sum(1 for s in segs if s.get("event") == "Cry")
    sad = sum(1 for s in segs if s.get("emotion") in ("SAD", "FEARFUL"))
    n = max(len(segs), 1)
    if cries or sad / n >= 0.15:
        return "重要时刻"
    if n_spk >= 3:
        return "会议纪要"
    if n_spk == 1 and duration_sec >= 900:
        return "讲座"
    if n_spk <= 2 and duration_sec < 1800:
        return "日常对话"
    return "其他"


def note_rel_dir(o, date: datetime, doc_type: str, base: str = "") -> str:
    """笔记相对 vault 的目录。

    默认（per_note_folder=true）：`录音笔记/<类型>/<年>/<日期 标题>/`
    —— 每条录音一个文件夹，里面放 md + 音频 + 字幕；
    per_note_folder=false 时退回旧结构：`录音笔记/<类型>/<年>/`。
    """
    parts = [o.notes_dir]
    if getattr(o, "notes_by_type", True):
        parts.append(doc_type)
    parts.append(f"{date:%Y}")
    if base and getattr(o, "per_note_folder", True):
        parts.append(base)
    return "/".join(x for x in parts if x)


def build_tags(job: dict, duration_sec: float, speaker_names: list[str], date: datetime,
               doc_type: str = "会议纪要") -> list[str]:
    tags = [doc_type, f"类型/{doc_type}", "来源/本地录音", f"{date:%Y/%m}"]
    if duration_sec < 1800:
        tags.append("时长/短会")
    elif duration_sec <= 3600:
        tags.append("时长/标准")
    else:
        tags.append("时长/长会")
    for n in speaker_names[:6]:
        if n and not n.startswith("说话人"):
            tags.append(f"说话人/{n}")
    return tags[:12]


def _yaml_list(items: list[str]) -> str:
    return "[" + ", ".join(items) + "]"


def _yaml_links(items: list[str]) -> str:
    """已命名 → 人物双链；未命名 → 纯文本（避免制造死链）。"""
    parts = []
    for i in items:
        if not i:
            continue
        if i.startswith("说话人"):
            parts.append(f'"{i}"')
        else:
            parts.append(f'"[[人物/{safe_name(i)}]]"')
    return "[" + ", ".join(parts) + "]"


def write_vault(cfg, job: dict, segs: list[dict], duration_sec: float,
                speaker_names: dict[int, str], wdir: Path, source: Path,
                paragraphs: list[dict] | None = None,
                summary: dict | None = None) -> dict:
    """把一次转写写入 Obsidian vault，返回写出的路径。未配置 vault 则跳过。"""
    o = cfg.obsidian
    if not o.vault_path:
        return {}
    vault = Path(o.vault_path).expanduser()
    if not vault.exists():
        raise RuntimeError(f"Obsidian vault 不存在：{vault}")

    now = datetime.now()
    date = now
    doc_type = classify(job, segs, duration_sec, summary)
    # 标题优先用 LLM 概括的 title，其次用文件名
    llm_title = (summary or {}).get("title") or ""
    title = safe_name(o.title_prefix + (llm_title or Path(job["source_name"]).stem))
    base = f"{date:%Y-%m-%d %H%M} {title}"
    per_note = bool(getattr(o, "per_note_folder", True))
    notes_rel = note_rel_dir(o, date, doc_type, base)
    notes_dir = vault / notes_rel
    att_dir = notes_dir if per_note else (vault / o.attachment_dir)
    tr_dir = vault / o.transcript_dir / f"{date:%Y}"
    audio_src = wdir / "preview.m4a"
    audio_rel = ""
    if audio_src.exists():
        att_dir.mkdir(parents=True, exist_ok=True)
        dst = att_dir / f"{base}.m4a"
        shutil.copy2(audio_src, dst)
        audio_rel = str(dst.relative_to(vault))

    srt_path = att_dir / f"{base}.srt"
    if audio_rel:
        atomic_write(srt_path, build_srt(segs, speaker_names))
    srt_rel = str(srt_path.relative_to(vault)) if srt_path.exists() else ""

    txt_path = att_dir / f"{base}.txt"
    lines = []
    for s in segs:
        spk = s.get("spk", -1)
        who = speaker_names.get(spk) or (f"说话人{spk + 1}" if spk >= 0 else "未知")
        lines.append(f"[{hms(s['start_ms'])}] {who}: {s.get('final_text') or s.get('raw_text')}")
    atomic_write(txt_path, "\n".join(lines))
    txt_rel = str(txt_path.relative_to(vault))

    # 说话人统计
    per: dict[int, float] = {}
    chars: dict[int, int] = {}
    for s in segs:
        k = s.get("spk", -1)
        per[k] = per.get(k, 0) + (s["end_ms"] - s["start_ms"]) / 1000
        chars[k] = chars.get(k, 0) + len(s.get("final_text") or s.get("raw_text") or "")
    order = sorted(per, key=lambda k: -per[k])
    named = [speaker_names.get(k) or f"说话人{k + 1}" for k in order]

    # 逐字稿笔记
    tr_lines = [f"---", "type: transcript", f'job_id: "{job["id"]}"',
                f'title: "{base}-逐字稿"',
                f'audio: "[[{audio_rel}]]"' if audio_rel else "audio: ",
                f'parent: "[[{notes_rel}/{base}]]"',
                "tags: [转写稿]",
                f"created: {now:%Y-%m-%dT%H:%M:%S%z}",
                "---", "", "## 逐字稿", ""]
    last = None
    for s in segs:
        k = s.get("spk", -1)
        if k != last:
            tr_lines.append(f"\n**{speaker_names.get(k) or f'说话人{k + 1}'}**\n")
            last = k
        tr_lines.append(f"`[{hms(s['start_ms'])}]` {s.get('final_text') or s.get('raw_text')}\n")
    tr_path = tr_dir / f"{base}-逐字稿.md"
    atomic_write(tr_path, "\n".join(tr_lines))

    if paragraphs:
        rf = [f"---", "type: refined-transcript", f'job_id: "{job["id"]}"',
              f'title: "{base}-整理稿"', "tags: [整理稿]", "---", "", "## 整理稿", ""]
        last = None
        for p in paragraphs:
            if p.get("spk") != last:
                rf.append(f"\n**{speaker_names.get(p.get('spk')) or f'说话人{(p.get('spk') or 0) + 1}'}**\n")
                last = p.get("spk")
            rf.append(f"`[{hms(p['start_ms'])}]` {p['text']}\n")
        rp = tr_dir / f"{base}-整理稿.md"
        atomic_write(rp, "\n".join(rf))

    # 纪要笔记
    tags = build_tags(job, duration_sec, named, date, doc_type)
    fm = [
        "---",
        f"type: {doc_type}",
        f'job_id: "{job["id"]}"',
        f'title: "{title}"',
        f"date: {date:%Y-%m-%d}",
        f'start_time: "{date:%H:%M}"',
        f'duration: "{hms(int(duration_sec * 1000))}"',
        f"duration_sec: {int(duration_sec)}",
        f'audio: "[[{audio_rel}]]"' if audio_rel else "audio: ",
        f'transcript: "[[{o.transcript_dir}/{date:%Y}/{base}-逐字稿]]"',
        f'srt: "[[{srt_rel}]]"' if srt_rel else "srt: ",
        f'source: "{safe_name(job["source_name"])}"',
        f"speakers: {_yaml_links(named)}",
        f"speaker_count: {len(per)}",
        f"language: zh-CN",
        f"status: 待校对",
        f"doc_type: {doc_type}",
        f"tags: {_yaml_list(tags)}",
        "engine:",
        "  asr: funasr-paraformer-zh",
        f"  app: 0.1.0",
        f"created: {now:%Y-%m-%dT%H:%M:%S%z}",
        f"updated: {now:%Y-%m-%dT%H:%M:%S%z}",
        "---", "",
        "> [!abstract] 概览",
        f"> 时长 {hms(int(duration_sec * 1000))}，共 {len(segs)} 段，{len(per)} 位说话人。",
    ]
    if summary and summary.get("summary"):
        fm += [f"> {summary['summary']}"]
    else:
        fm += ["> 智能纪要未启用（`summary.enabled` 关闭或 LLM 不可用）。"]
    fm += ["",
        "## 👥 说话人",
        "",
        "| 说话人 | 时长 | 字数 |",
        "|---|---|---|",
    ]
    for k in order:
        fm.append(f"| {speaker_names.get(k) or f'说话人{k + 1}'} | {per[k] / 60:.1f} 分钟 | {chars[k]} 字 |")
    if paragraphs:
        fm += ["", "## ✍️ 整理稿", ""]
        last = None
        for p in paragraphs:
            if p.get("spk") != last:
                fm.append(f"**{speaker_names.get(p.get('spk')) or f'说话人{(p.get('spk') or 0) + 1}'}**")
                last = p.get("spk")
            fm.append(f"`[{hms(p['start_ms'])}]` {p['text']}")
            fm.append("")
    if summary:
        if summary.get("chapters"):
            fm += ["## 🗂 章节", "", "| 时间 | 章节 | 概要 |", "|---|---|---|"]
            for c in summary["chapters"]:
                fm.append(f"| {hms(c.get('start_ms', 0))} | {c.get('title', '')} | {c.get('summary', '')} |")
            fm.append("")
        if summary.get("decisions"):
            fm += ["## 📌 决议", ""] + [f"- {d}" for d in summary["decisions"]] + [""]
        if summary.get("todos"):
            fm += ["## ✅ 待办", ""] + [f"- [ ] {t['task']} — @{t['owner']}" for t in summary["todos"]] + [""]
        if summary.get("keywords"):
            fm += ["## 🏷 关键词", "", "、".join(summary["keywords"]), ""]
    fm += ["", "## 📝 逐字稿", f"[[{o.transcript_dir}/{date:%Y}/{base}-逐字稿|查看完整逐字稿]]", ""]
    if audio_rel:
        fm += ["## 🎧 音频", f"![[{audio_rel}]]", ""]
    fm += ["## 🔗 附件", f"- 字幕：[[{srt_rel}]]" if srt_rel else "", f"- 纯文本：[[{txt_rel}]]", "", ""]
    note_path = notes_dir / f"{base}.md"
    atomic_write(note_path, "\n".join(x for x in fm if x is not None))

    return {
        "note": str(note_path.relative_to(vault)),
        "transcript": str(tr_path.relative_to(vault)),
        "audio": audio_rel, "srt": srt_rel, "txt": txt_rel, "tags": tags,
        "refined": str((tr_dir / f"{base}-整理稿.md").relative_to(vault)) if paragraphs else "",
    }
