import json, sys, time
from funasr import AutoModel
m = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", disable_update=True, device="cpu")
t=time.time(); r = m.generate(input=sys.argv[1]); dt=time.time()-t
print(json.dumps({"file": sys.argv[1], "infer_sec": round(dt,2), "text": r[0].get("text","")}, ensure_ascii=False, indent=2))
