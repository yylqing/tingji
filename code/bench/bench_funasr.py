"""真机基准：FunASR Paraformer-large + FSMN-VAD + CT-Punc + CAM++ 说话人分离。

测量：模型加载耗时、端到端耗时、RTF、峰值内存、逐句时间戳与说话人。
"""
import json
import os
import resource
import sys
import time

os.environ.setdefault("MODELSCOPE_CACHE", os.path.abspath(".cache/modelscope"))

from funasr import AutoModel  # noqa: E402

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "bench/in/meeting_sample.wav"
MODE = sys.argv[2] if len(sys.argv) > 2 else "full"

t0 = time.time()
kwargs = dict(
    model="paraformer-zh",
    vad_model="fsmn-vad",
    punc_model="ct-punc",
    disable_update=True,
    device="cpu",
)
if MODE == "full":
    kwargs["spk_model"] = "cam++"
model = AutoModel(**kwargs)
load_s = time.time() - t0

t1 = time.time()
res = model.generate(input=AUDIO, batch_size_s=300, return_spk_res=(MODE == "full"))
infer_s = time.time() - t1

peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

import subprocess

dur = float(
    subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", AUDIO]
    )
    .decode()
    .strip()
)

out = {
    "audio": AUDIO,
    "audio_sec": round(dur, 2),
    "mode": MODE,
    "model_load_sec": round(load_s, 2),
    "infer_sec": round(infer_s, 2),
    "rtf": round(infer_s / dur, 4),
    "speed_x_realtime": round(dur / infer_s, 2),
    "peak_rss_mb": round(peak_mb, 1),
    "sentences": [],
}
for s in res:
    for item in s.get("sentence_info", []) or []:
        out["sentences"].append(
            {
                "start_ms": item.get("start"),
                "end_ms": item.get("end"),
                "spk": item.get("spk"),
                "text": item.get("text"),
            }
        )
    if not s.get("sentence_info"):
        out["text"] = s.get("text")

print(json.dumps(out, ensure_ascii=False, indent=2))
with open(f"bench/result_{MODE}.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
