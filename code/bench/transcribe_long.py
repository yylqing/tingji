"""长音频转写：FunASR Paraformer + VAD + CT-Punc + CAM++ 说话人分离。

用法: python bench/transcribe_long.py <audio> <out_prefix> [hotwords_csv]
"""
import json
import os
import resource
import subprocess
import sys
import time

os.environ.setdefault("MODELSCOPE_CACHE", os.path.abspath(".cache/modelscope"))

AUDIO = sys.argv[1]
OUT = sys.argv[2]
HOTWORDS = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []

dur = float(
    subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", AUDIO]
    ).decode().strip()
)

from funasr import AutoModel  # noqa: E402

t0 = time.time()
model = AutoModel(
    model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
    spk_model="cam++", device="cpu", disable_update=True,
)
load_s = time.time() - t0

kw = {"batch_size_s": 300, "return_spk_res": True}
if HOTWORDS:
    kw["hotword"] = " ".join(HOTWORDS)

t1 = time.time()
res = model.generate(input=AUDIO, **kw)
infer_s = time.time() - t1

segs = []
for item in res:
    for s in item.get("sentence_info", []) or []:
        segs.append({
            "start_ms": int(s.get("start", 0)),
            "end_ms": int(s.get("end", 0)),
            "spk": int(s.get("spk", -1)),
            "text": (s.get("text") or "").strip(),
        })
segs.sort(key=lambda x: x["start_ms"])

base = {"audio": AUDIO, "audio_sec": round(dur, 1), "model_load_sec": round(load_s, 1),
        "infer_sec": round(infer_s, 1), "rtf": round(infer_s / dur, 4),
        "speed_x_realtime": round(dur / infer_s, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1),
        "hotwords": HOTWORDS, "num_segments": len(segs)}

json.dump({**base, "segments": segs}, open(f"{OUT}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(base, ensure_ascii=False, indent=1))
