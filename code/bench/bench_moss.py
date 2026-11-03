"""MOSS-Transcribe-Diarize 0.9B 本机基准（CPU / transformers）。

用法: python bench/bench_moss.py <audio> [max_new_tokens]
"""
import json
import os
import re
import resource
import sys
import time

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_DIR = os.path.abspath(".cache/moss")
AUDIO = sys.argv[1]
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号（[S01]、[S02]、[S03]…）开头，"
    "正文为对应的语音内容，并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
if os.environ.get("MOSS_HOTWORDS"):
    PROMPT += "热词提示：" + os.environ["MOSS_HOTWORDS"]

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
DEV = os.environ.get("MOSS_DEVICE", "cpu")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, dtype=torch.float32, attn_implementation="sdpa"
).to(DEV).eval()
load_s = time.time() - t0

wav, sr = sf.read(AUDIO, dtype="float32")
if wav.ndim > 1:
    wav = wav.mean(1)
assert sr == 16000, sr

messages = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": PROMPT}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, audio=[wav])
inputs = {k: v.to(DEV) for k, v in inputs.items()}

t1 = time.time()
with torch.inference_mode():
    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
gen_s = time.time() - t1

new_ids = out[0][inputs["input_ids"].shape[1]:]
raw = processor.tokenizer.decode(new_ids, skip_special_tokens=True)

# 解析 [start][Sxx]text[end]
seg = re.findall(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", raw, re.S)
segments = [
    {"start": float(a), "end": float(d), "spk": b, "text": c.strip()} for a, b, c, d in seg
]
dur = len(wav) / sr
res = {
    "model": "MOSS-Transcribe-Diarize 0.9B (fp32, CPU)",
    "audio": AUDIO,
    "audio_sec": round(dur, 2),
    "load_sec": round(load_s, 2), "device": DEV,
    "generate_sec": round(gen_s, 2),
    "rtf": round(gen_s / dur, 3),
    "speed_x_realtime": round(dur / gen_s, 2),
    "new_tokens": int(len(new_ids)),
    "tok_per_sec": round(len(new_ids) / gen_s, 2),
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1),
    "speakers": sorted({s["spk"] for s in segments}),
    "raw": raw.strip(),
    "segments": segments,
}
print(json.dumps(res, ensure_ascii=False, indent=2))
tag = os.path.basename(AUDIO).rsplit(".", 1)[0]
with open(f"bench/result_moss_{tag}.json", "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
