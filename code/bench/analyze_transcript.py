"""把转写 JSON 整理成可读逐字稿 + 统计 + 专名候选。"""
import json
import sys
from collections import Counter, defaultdict

import jieba.posseg as pseg

INP = sys.argv[1]
MD_OUT = sys.argv[2]
TXT_OUT = sys.argv[3]

d = json.load(open(INP))
segs = [s for s in d["segments"] if s["text"]]
MERGE_GAP_MS = 2000

# 合并同一说话人的连续短句
blocks = []
for s in segs:
    if blocks and blocks[-1]["spk"] == s["spk"] and s["start_ms"] - blocks[-1]["end_ms"] <= MERGE_GAP_MS:
        blocks[-1]["text"] += s["text"]
        blocks[-1]["end_ms"] = s["end_ms"]
    else:
        blocks.append({"spk": s["spk"], "start_ms": s["start_ms"], "end_ms": s["end_ms"], "text": s["text"]})

speech = sum(s["end_ms"] - s["start_ms"] for s in segs) / 1000
per = defaultdict(lambda: {"sec": 0.0, "turns": 0, "chars": 0})
for b in blocks:
    p = per[b["spk"]]
    p["sec"] += (b["end_ms"] - b["start_ms"]) / 1000
    p["turns"] += 1
    p["chars"] += len(b["text"])

def hms(ms):
    s = int(ms // 1000)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

# 专名候选
cands = Counter()
for s in segs:
    for w in pseg.cut(s["text"]):
        if w.flag in ("nr", "ns", "nt") and len(w.word) >= 2:
            cands[w.word] += 1

stats = {
    "audio_sec": d["audio_sec"],
    "infer_sec": d["infer_sec"],
    "speed_x_realtime": d["speed_x_realtime"],
    "peak_rss_mb": d["peak_rss_mb"],
    "num_segments": len(segs),
    "num_blocks": len(blocks),
    "num_speakers": len(per),
    "speech_sec": round(speech, 1),
    "speech_ratio": round(speech / d["audio_sec"], 3),
    "chars": sum(p["chars"] for p in per.values()),
    "per_speaker": {str(k): {kk: round(vv, 1) for kk, vv in v.items()} for k, v in sorted(per.items())},
    "top_proper_nouns": cands.most_common(25),
}
print(json.dumps(stats, ensure_ascii=False, indent=1))

# 逐字稿 Markdown
num = {k: i + 1 for i, k in enumerate(sorted(per))}
with open(MD_OUT, "w") as f:
    f.write(f"# 会议录音 · 逐字稿\n\n")
    f.write(f"- 时长：{hms(d['audio_sec']*1000)}｜有效语音 {speech/60:.1f} 分钟（{speech/d['audio_sec']*100:.0%}）\n")
    f.write(f"- 说话人：{len(per)} 人｜句子 {len(segs)} 条｜合并后发言块 {len(blocks)} 个\n")
    f.write(f"- 引擎：FunASR Paraformer-zh + FSMN-VAD + CT-Punc + CAM++（CPU）\n")
    f.write(f"- 处理耗时：{d['infer_sec']}s（{d['speed_x_realtime']}× 实时）\n\n")
    for k in sorted(per):
        p = per[k]
        f.write(f"- 说话人{k+1}：{p['sec']/60:.1f} 分钟，{p['turns']} 次发言，{p['chars']} 字\n")
    f.write("\n---\n\n")
    for b in blocks:
        f.write(f"**说话人{b['spk']+1}** `[{hms(b['start_ms'])}]`\n\n{b['text']}\n\n")

with open(TXT_OUT, "w") as f:
    for b in blocks:
        f.write(f"[{hms(b['start_ms'])}] 说话人{b['spk']+1}: {b['text']}\n")

json.dump(stats, open(MD_OUT.replace(".md", "_stats.json"), "w"), ensure_ascii=False, indent=1)
print(f"\n已写出：{MD_OUT} / {TXT_OUT}")
