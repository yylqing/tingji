"""基准：sherpa-onnx 离线说话人分离（pyannote segmentation-3.0 + 3D-Speaker CAM++）。"""
import json
import resource
import subprocess
import sys
import time

import soundfile as sf
import sherpa_onnx

AUDIO = sys.argv[1] if len(sys.argv) > 1 else "bench/in/meeting_sample.wav"
BASE = ".cache/sherpa"

cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
    segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=f"{BASE}/sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
        ),
        num_threads=4,
    ),
    embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=f"{BASE}/campplus.onnx", num_threads=4
    ),
    clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
    min_duration_on=0.2,
    min_duration_off=0.3,
)

t0 = time.time()
sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
assert sd.sample_rate == 16000, sd.sample_rate

audio, sr = sf.read(AUDIO, dtype="float32")
assert sr == 16000, sr

t1 = time.time()
res = sd.process(audio).sort_by_start_time()
infer_s = time.time() - t1
load_s = t1 - t0

dur = len(audio) / sr
peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
segments = [{"start": round(r.start, 2), "end": round(r.end, 2), "spk": r.speaker} for r in res]
speakers = sorted({s["spk"] for s in segments})

out = {
    "engine": "sherpa-onnx 2.0.1 / pyannote-seg-3.0 + campplus zh_en",
    "audio_sec": round(dur, 2),
    "model_load_sec": round(load_s, 2),
    "diarize_sec": round(infer_s, 2),
    "rtf": round(infer_s / dur, 4),
    "speed_x_realtime": round(dur / infer_s, 2),
    "peak_rss_mb": round(peak_mb, 1),
    "num_speakers": len(speakers),
    "speakers": speakers,
    "segments": segments,
}
print(json.dumps(out, ensure_ascii=False, indent=2))
with open("bench/result_sherpa.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
