"""验证重叠检测特征：段内嵌入自一致性 + 聚类边距。"""
import json, sys, itertools
import numpy as np, soundfile as sf, sherpa_onnx as so

MODEL = ".cache/sherpa/campplus.onnx"
cfg = so.SpeakerEmbeddingExtractorConfig(model=MODEL, num_threads=4)
ex = so.SpeakerEmbeddingExtractor(cfg)

def emb(samples, sr=16000):
    st = ex.create_stream()
    st.accept_waveform(sr, samples)
    st.input_finished()
    if not ex.is_ready(st):
        return None
    return np.array(ex.compute(st), dtype=np.float32)

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def analyze(wav, js, label):
    audio, sr = sf.read(wav, dtype="float32")
    if audio.ndim > 1: audio = audio.mean(1)
    segs = json.load(open(js))["sentences"]
    out = []
    for s in segs:
        a, b = int(s["start_ms"] / 1000 * sr), int(s["end_ms"] / 1000 * sr)
        clip = audio[a:b]
        win, hop = int(0.5 * sr), int(0.25 * sr)
        embs = []
        for i in range(0, max(1, len(clip) - win + 1), hop):
            w = clip[i:i + win]
            if len(w) < int(0.3 * sr): continue
            e = emb(w)
            if e is not None: embs.append(e)
        cons = np.mean([cos(x, y) for x, y in itertools.combinations(embs, 2)]) if len(embs) > 1 else 1.0
        whole = emb(clip) if len(clip) > int(0.3 * sr) else None
        out.append({"spk": s["spk"], "start": s["start_ms"] / 1000, "text": s["text"],
                    "consistency": round(float(cons), 3), "vec": whole, "n_win": len(embs)})
    # 用已知 spk 标签算质心，再算 sim1/sim2/margin
    cents = {}
    for spk in {o["spk"] for o in out}:
        vs = [o["vec"] for o in out if o["spk"] == spk and o["vec"] is not None]
        c = np.mean(vs, 0); cents[spk] = c / np.linalg.norm(c)
    print(f"\n=== {label} ===")
    print(f"{'spk':>4} {'起(s)':>7} {'一致性':>7} {'sim1':>6} {'sim2':>6} {'margin':>7}  文本")
    for o in out:
        if o["vec"] is None: continue
        v = o["vec"] / np.linalg.norm(o["vec"])
        sims = sorted(((cos(v, c), k) for k, c in cents.items()), reverse=True)
        sim1 = sims[0][0]; sim2 = sims[1][0] if len(sims) > 1 else 0.0
        print(f"{o['spk']:>4} {o['start']:>7.1f} {o['consistency']:>7.3f} {sim1:>6.3f} {sim2:>6.3f} {sim1-sim2:>7.3f}  {o['text'][:26]}")
    cons_all = [o["consistency"] for o in out]
    print(f"一致性: 均值={np.mean(cons_all):.3f} 最低={min(cons_all):.3f}")
    return out

analyze("bench/in/meeting_sample.wav", "bench/result_full.json", "无重叠基线")
analyze("bench/in/overlap_sample.wav", "bench/result_overlap.json", "均衡重叠")
