"""目标说话人提取（VoxLock）验证：给定目标说话人样本，从混合音频中提取该人声。"""
import json, resource, sys, time
import numpy as np, soundfile as sf, sherpa_onnx as so

BASE = ".cache/sherpa/sherpa-onnx-voxlock-int8"
combined = sys.argv[1]
voice = sys.argv[2]
out_path = sys.argv[3]

vox = so.OfflineSpeechDenoiserVoxLockModelConfig(
    chunk_model=f"{BASE}/chunk_int8.onnx",
    enroll_model=f"{BASE}/enroll.onnx",
    voice_sample=voice,
)
mc = so.OfflineSpeechDenoiserModelConfig(voxlock=vox, num_threads=4)
cfg = so.OfflineSpeechDenoiserConfig(model=mc)

t0 = time.time(); d = so.OfflineSpeechDenoiser(cfg); load = time.time() - t0
audio, sr = sf.read(combined, dtype="float32")
if audio.ndim > 1: audio = audio.mean(axis=1)
t1 = time.time(); res = d.run(audio, sr); infer = time.time() - t1
out = np.asarray(res.samples, dtype=np.float32)
sf.write(out_path, out, res.sample_rate)

dur = len(audio) / sr
print(json.dumps({
    "input": combined, "voice_sample": voice, "out": out_path,
    "audio_sec": round(dur, 2), "load_sec": round(load, 2), "infer_sec": round(infer, 2),
    "rtf": round(infer / dur, 3), "speed_x": round(dur / infer, 1),
    "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 1),
}, ensure_ascii=False, indent=2))
