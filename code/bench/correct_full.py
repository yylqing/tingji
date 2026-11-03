"""全量纠错：按"发言块内连续句子"切批，逐批送本地 LLM 纠错+重断句，保留说话人与时间戳。

用法: python bench/correct_full.py <spkfix.json> <out.md> [batch_chars]
"""
import json
import sys
import time
from collections import defaultdict

import httpx

INP = sys.argv[1]
OUT = sys.argv[2]
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 700
MODEL = "qwen3.5:9b-q4_K_M"
LLM = "http://127.0.0.1:11434/api/generate"

SYSTEM = (
    "你是中文会议转写稿的校对员。输入是语音识别的原始输出，存在标点混乱、口语填充词、"
    "同音字与英文术语错识。请整理成通顺的书面口语。"
)
RULES = """要求：
1. 去掉口语填充词（嗯/呃/啊/那个/就是说/然后就是等）与无意义重复，但不删减实质信息；
2. 重新断句并使用正确标点；
3. 修正明显的同音错字与英文术语（如 sars→SaaS、口顶→Copilot、Infinity Band→InfiniBand），不确定的不要改；
4. 保持第一人称口语叙述，**输出必须是一整段连续文字**：禁止编号列表、禁止小标题、禁止分点、禁止总结；
5. 不增加原文没有的事实、数字或人名；
6. 只输出整理后的正文。"""

d = json.load(open(INP))
segs = [s for s in d["segments"] if s["text"]]


def hms(ms):
    s = int(ms // 1000)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# 1) 合并为发言块（同人 + 间隔 ≤2s）
blocks = []
for s in segs:
    if blocks and blocks[-1]["spk"] == s["spk"] and s["start_ms"] - blocks[-1]["segments"][-1]["end_ms"] <= 2000:
        blocks[-1]["segments"].append(s)
        blocks[-1]["end_ms"] = s["end_ms"]
    else:
        blocks.append({"spk": s["spk"], "start_ms": s["start_ms"], "end_ms": s["end_ms"], "segments": [s]})

# 2) 块内按字数切批（不跨块，保留说话人与起始时间）
chunks = []
for bi, b in enumerate(blocks):
    buf, start = [], None
    for s in b["segments"]:
        if start is None:
            start = s["start_ms"]
        buf.append(s)
        if sum(len(x["text"]) for x in buf) >= BATCH:
            chunks.append({"bi": bi, "start_ms": start, "text": "".join(x["text"] for x in buf)})
            buf, start = [], None
    if buf:
        chunks.append({"bi": bi, "start_ms": start, "text": "".join(x["text"] for x in buf)})

print(f"发言块 {len(blocks)}，待纠错批次 {len(chunks)}", flush=True)

# 3) 逐批调用
t0 = time.time()
out = []
with httpx.Client(timeout=300) as cli:
    for i, ch in enumerate(chunks, 1):
        payload = {
            "model": MODEL,
            "prompt": f"{RULES}\n\n原文：\n{ch['text']}",
            "system": SYSTEM,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.15, "num_predict": 1200},
            "keep_alive": "10m",
        }
        fixed = None
        for attempt in range(2):
            try:
                r = cli.post(LLM, json=payload).json()
                fixed = (r.get("response") or "").strip()
            except Exception as e:
                fixed = None
            if fixed:
                break
        if not fixed:
            fixed = ch["text"]  # 失败则保留原文
        out.append({"bi": ch["bi"], "start_ms": ch["start_ms"], "raw": ch["text"], "fixed": fixed})
        if i % 5 == 0 or i == len(chunks):
            print(f"  {i}/{len(chunks)}  {time.time()-t0:.0f}s", flush=True)

# 4) 组装
per = defaultdict(list)
for x in out:
    per[x["bi"]].append(x)
tot = defaultdict(float)
for b in blocks:
    tot[b["spk"]] += (b["end_ms"] - b["start_ms"]) / 1000
renum = {k: i + 1 for i, k in enumerate(sorted(tot, key=lambda k: -tot[k]))}

with open(OUT, "w") as f:
    f.write("# 会议录音 · 纠错稿\n\n")
    f.write(f"- 时长 01:12:55｜纠错批次 {len(chunks)}｜耗时 {time.time()-t0:.0f}s\n")
    f.write(f"- 引擎：FunASR Paraformer-zh → qwen3.5:9b-q4_K_M（逐批纠错，保留说话人与时间戳）\n")
    f.write("- 说明：**未做人工校订**，专名与数字仍可能有误\n\n---\n\n")
    for bi, b in enumerate(blocks):
        f.write(f"**说话人{renum[b['spk']]}** `[{hms(b['start_ms'])}]`\n\n")
        for x in per.get(bi, []):
            f.write(x["fixed"] + "\n\n")

json.dump(out, open(OUT.replace(".md", "_raw.json"), "w"), ensure_ascii=False, indent=1)
print(f"\n完成：{OUT}  用时 {time.time()-t0:.0f}s")
