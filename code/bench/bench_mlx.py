"""对照基准：mlx-whisper large-v3-turbo 在 M4 GPU 上的中文/中英混说表现。"""
import json
import os
import resource
import subprocess
import sys
import time

import mlx_whisper

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "bench/in/meeting_sample.wav"
MODEL = os.environ.get("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")

t0 = time.time()
r = mlx_whisper.transcribe(
    AUDIO,
    path_or_hf_repo=MODEL,
    language="zh",
    initial_prompt="以下是普通话会议内容，可能包含英文技术术语，如 ClickHouse、API、OKR。",
    word_timestamps=False,
    verbose=False,
)
infer_s = time.time() - t0
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
dur = float(
    subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", AUDIO]
    )
    .decode()
    .strip()
)

out = {
    "model": MODEL,
    "audio_sec": round(dur, 2),
    "infer_sec": round(infer_s, 2),
    "rtf": round(infer_s / dur, 4),
    "speed_x_realtime": round(dur / infer_s, 2),
    "peak_rss_mb": round(peak_mb, 1),
    "text": r["text"].strip(),
    "segments": [
        {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
        for s in r["segments"]
    ],
}
print(json.dumps(out, ensure_ascii=False, indent=2))
with open("bench/result_mlx.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
