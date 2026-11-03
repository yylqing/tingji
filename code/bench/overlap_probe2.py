"""重叠检测特征 v2：1s 窗口自一致性 + 前后半段一致性。"""
import json, itertools
import numpy as np, soundfile as sf, sherpa_onnx as so
cfg = so.SpeakerEmbeddingExtractorConfig(model=".cache/sherpa/campplus.onnx", num_threads=4)
ex = so.SpeakerEmbeddingExtractor(cfg)
def emb(x, sr=16000):
    st=ex.create_stream(); st.accept_waveform(sr,x); st.input_finished()
    return np.array(ex.compute(st),dtype=np.float32) if ex.is_ready(st) else None
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
def run(wav, js, label):
    a,sr=sf.read(wav,dtype="float32")
    if a.ndim>1: a=a.mean(1)
    segs=json.load(open(js))["sentences"]; rows=[]
    for s in segs:
        clip=a[int(s["start_ms"]/1000*sr):int(s["end_ms"]/1000*sr)]
        win,hop=int(1.0*sr),int(0.5*sr); es=[]
        for i in range(0,max(1,len(clip)-win+1),hop):
            w=clip[i:i+win]
            if len(w)<int(0.5*sr): continue
            e=emb(w)
            if e is not None: es.append(e)
        cons=np.mean([cos(x,y) for x,y in itertools.combinations(es,2)]) if len(es)>1 else float('nan')
        h=len(clip)//2
        e1,e2=(emb(clip[:h]),emb(clip[h:])) if h>int(0.5*sr) else (None,None)
        half=cos(e1,e2) if e1 is not None and e2 is not None else float('nan')
        rows.append((s["start_ms"]/1000,cons,half,len(es),s["text"]))
    print(f"\n=== {label} ===")
    print(f"{'起(s)':>6} {'1s窗一致性':>10} {'前后半段':>9}  文本")
    for r in rows: print(f"{r[0]:>6.1f} {r[1]:>10.3f} {r[2]:>9.3f}  {r[4][:30]}")
    v=[r[1] for r in rows if r[1]==r[1]]; h=[r[2] for r in rows if r[2]==r[2]]
    print(f"一致性 均值={np.mean(v):.3f} 最低={min(v):.3f} | 半段 均值={np.mean(h):.3f} 最低={min(h):.3f}")
run("bench/in/meeting_sample.wav","bench/result_full.json","无重叠基线")
run("bench/in/overlap_sample.wav","bench/result_overlap.json","均衡重叠")
