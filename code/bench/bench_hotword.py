"""热词效果对比：基线 / 模型级热词 / 文本级热词后处理。

用法: python bench/bench_hotword.py <audio> <hotword_file_or_csv>
"""
import json
import os
import sys
import time

os.environ.setdefault("MODELSCOPE_CACHE", os.path.abspath(".cache/modelscope"))

from funasr import AutoModel  # noqa: E402

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "bench/in/hotword_sample.wav"
HOTWORDS = sys.argv[2].split(",") if len(sys.argv) > 2 else []
LABEL = sys.argv[3] if len(sys.argv) > 3 else "baseline"

model = AutoModel(
    model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
    disable_update=True, device="cpu",
)

uses_postprocess = LABEL == "postprocess"
kw = {}
if LABEL in ("model", "postprocess"):
    if uses_postprocess:
        kw["postprocess_hotwords"] = {h: h for h in HOTWORDS}
        kw["postprocess_hotword_threshold"] = 0.85
        kw["return_postprocess_hotword_matches"] = True
    else:
        kw["hotword"] = " ".join(HOTWORDS)

t = time.time()
res = model.generate(input=AUDIO, **kw)
dt = time.time() - t

item = res[0]
out = {
    "label": LABEL,
    "audio": AUDIO,
    "hotwords": HOTWORDS,
    "infer_sec": round(dt, 2),
    "text": item.get("text", ""),
    "hotword_matches": item.get("postprocess_hotword_matches")
    or item.get("hotword_matches")
    or [],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
tag = os.path.basename(AUDIO).rsplit(".", 1)[0]
with open(f"bench/result_hotword_{tag}_{LABEL}.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
