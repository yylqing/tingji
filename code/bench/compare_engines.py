"""同一段音频用不同 ASR 引擎各跑一遍，产出可读的对比报告。

用法：
    PYTHONPATH=. .venv-funasr/bin/python bench/compare_engines.py work/xxx.wav

只跑「识别 + 说话人」两层（关掉纠错/整理稿/纪要），把三次的逐字稿放到
output/compare/<engine>/transcript.json，最后生成 output/compare/report.md。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ENGINES = ["funasr", "sensevoice", "funasr_nano"]
LABELS = {"funasr": "Paraformer", "sensevoice": "SenseVoice", "funasr_nano": "Fun-ASR-Nano"}
MEAN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")
FILLERS = ("嗯", "呃", "啊", "那个", "就是", "然后")


def make_config(engine: str) -> Path:
    base = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    base.setdefault("asr", {})["provider"] = engine
    for sec, key in (("correction", "enabled"), ("refine", "enabled"),
                     ("summary", "enabled"), ("roles", "enabled")):
        base.setdefault(sec, {})[key] = False
    tmp = ROOT / "work" / f"cmp-{engine}.yaml"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return tmp


def metrics(segs: list[dict]) -> dict:
    text = "".join(s.get("raw_text") or "" for s in segs)
    chars = len(MEAN.findall(text))
    durs = [(s["end_ms"] - s["start_ms"]) / 1000 for s in segs]
    return {
        "chars": chars,
        "segs": len(segs),
        "median_seg_sec": round(sorted(durs)[len(durs) // 2], 2) if durs else 0,
        "commas": text.count("，"),
        "periods": text.count("。"),
        "fillers": {f: text.count(f) for f in FILLERS},
        "punctuation_anomalies": len(re.findall(r"[，。！？][^，。！？]{1}[，。！？]", text)),
        "sample": text[:120],
    }


def window_text(segs: list[dict], start_s: float, end_s: float, limit: int = 90) -> str:
    out = []
    for s in segs:
        if s["end_ms"] / 1000 < start_s or s["start_ms"] / 1000 > end_s:
            continue
        t = (s.get("raw_text") or "").strip()
        if t:
            out.append(t)
    return "".join(out)[:limit]


def main(wav: str) -> int:
    src = Path(wav)
    if not src.is_absolute():
        src = ROOT / wav
    stem = src.stem
    cmp_dir = ROOT / "output" / "compare"
    cmp_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}

    for eng in ENGINES:
        cfg = make_config(eng)
        # 关键：清掉该音频的工作目录，否则会复用上一次的 ASR 窗口缓存
        import sqlite3, shutil
        try:
            con = sqlite3.connect(ROOT / "data/app.db")
            row = con.execute(
                "SELECT id FROM jobs WHERE source_name=? ORDER BY created_at DESC LIMIT 1",
                (src.name,)).fetchone()
            con.close()
            if row:
                shutil.rmtree(ROOT / "work" / row[0], ignore_errors=True)
        except Exception as e:
            print("清理缓存失败:", e, flush=True)
        print(f"\n=== {LABELS[eng]}（{eng}）===", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-m", "app.cli", "-c", str(cfg), "run", str(src), "--force"],
            cwd=str(ROOT), capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        )
        dt = time.time() - t0
        if r.returncode != 0:
            print("失败:", (r.stderr or r.stdout)[-400:], flush=True)
            continue
        out_json = ROOT / "output" / stem / "transcript.json"
        if not out_json.exists():
            print("没有产出 transcript.json", flush=True)
            continue
        segs = json.loads(out_json.read_text(encoding="utf-8"))
        (cmp_dir / eng).mkdir(parents=True, exist_ok=True)
        (cmp_dir / eng / "transcript.json").write_text(
            json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
        m = metrics(segs)
        m["seconds"] = round(dt, 1)
        m["segments"] = segs
        results[eng] = m
        print(f"完成 {dt:.0f}s：{m['chars']} 字 / {m['segs']} 段 / 标点异常 {m['punctuation_anomalies']}",
              flush=True)

    if not results:
        print("没有成功的结果")
        return 1

    # 报告
    lines = [f"# 引擎对比：{stem}", "",
             f"音频：`{src.relative_to(ROOT)}`　时长：{results[next(iter(results))]['segments'][-1]['end_ms']/60000:.1f} 分钟", ""]
    header = "| 指标 | " + " | ".join(LABELS[e] for e in results) + " |"
    lines += [header, "|" + "---|" * (len(results) + 1)]
    rows = [("总字数", "chars"), ("段数", "segs"), ("段长中位数(s)", "median_seg_sec"),
            ("逗号数", "commas"), ("句号数", "periods"), ("标点异常", "punctuation_anomalies"),
            ("耗时(s)", "seconds")]
    for name, key in rows:
        lines.append(f"| {name} | " + " | ".join(str(results[e][key]) for e in results) + " |")
    lines.append("| 嗯/呃/啊 | " + " | ".join(
        "/".join(str(results[e]["fillers"][f]) for f in ("嗯", "呃", "啊")) for e in results) + " |")
    lines.append("")

    for a, b in ((0, 60), (300, 360), (600, 660), (700, 760)):
        lines += [f"## {a//60:02d}:{a%60:02d} – {b//60:02d}:{b%60:02d}", ""]
        for e in results:
            txt = window_text(results[e]["segments"], a, b)
            if txt:
                lines += [f"**{LABELS[e]}**：{txt}", ""]

    (cmp_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告 → {cmp_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
