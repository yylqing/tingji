"""生成一套**完全虚构**的演示数据，用于截图/演示（不含任何真实姓名或录音）。

用法：
    PYTHONPATH=code .venv-funasr/bin/python bench/demo_seed.py /tmp/tingji-demo

它会创建：
    /tmp/tingji-demo/config.yaml   干净的演示配置（本地环回、无令牌、无 Obsidian）
    /tmp/tingji-demo/data/app.db   3 个任务 + 段落 + 整理稿 + 纪要 + 2 个声纹

然后：
    PYTHONPATH=code .venv-funasr/bin/python -m app.cli -c /tmp/tingji-demo/config.yaml \
        service --port 8091
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def w(cfg_dir: Path) -> None:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "data").mkdir(exist_ok=True)
    (cfg_dir / "work").mkdir(exist_ok=True)
    (cfg_dir / "in").mkdir(exist_ok=True)
    (cfg_dir / "out").mkdir(exist_ok=True)
    cfg = f"""app:
  data_dir: {cfg_dir}/data
  work_dir: {cfg_dir}/work
  input_dir: {cfg_dir}/in
  output_dir: {cfg_dir}/out
  log_level: INFO
  max_concurrency: 1

audio:
  target_sr: 16000
  loudness_normalize: true
  preview_m4a: true

asr:
  provider: auto
  model: paraformer-zh
  device: cpu

diarization:
  enabled: true
  backend: sherpa
  max_speakers: 0

refine:
  enabled: false

summary:
  enabled: false

correction:
  enabled: false

roles:
  enabled: false

obsidian:
  vault_path: ""

server:
  bind: 127.0.0.1
  port: 8091
  token: ""

openai_api:
  enabled: true
  api_key: ""
"""
    (cfg_dir / "config.yaml").write_text(cfg)


LINES = [
    ("张三", "我们先过一下这周的进展。支付链路的改造已经上线灰度，错误率从千分之三降到千分之一点二。"),
    ("李四", "风控那边有个依赖，规则引擎的版本还没对齐，下周才能联调。"),
    ("张三", "这个风险要记一下。如果下周三还联调不了，我们就先回滚灰度，保证主链路稳定。"),
    ("王五", "我补充一下数据：这周新增商户 1,240 家，环比增长 18%，其中餐饮类占了一半。"),
    ("李四", "餐饮类增长主要来自地推团队，但他们的转化率数据还没回流，要等下个版本埋点。"),
    ("张三", "那埋点这块谁跟？"),
    ("王五", "我跟，周五之前给到指标定义，下周一出第一版报表。"),
    ("张三", "好。第二个议题是下季度的预算，服务器成本涨得比较快，主要是推理集群。"),
    ("李四", "推理集群现在 32 张卡，利用率大概 60%，可以先把批处理任务挪到晚上。"),
    ("张三", "这个方案可以试，先跑两周看效果。预算先按增长 20% 报，留出弹性。"),
    ("王五", "我整理一份成本拆分表，按业务线拆，周三发群里。"),
    ("张三", "最后确认一下待办：风控联调、埋点指标、成本拆分表、灰度观察。散会。"),
]


def seed(cfg_dir: Path) -> None:
    sys.path.insert(0, str(ROOT / "code"))
    from app.db import DB

    db = DB(cfg_dir / "data" / "app.db")
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    jobs = [
        ("产品周会.m4a", "done", 100, "完成，用时 186s", 41.5, "会议纪要"),
        ("客户需求沟通.m4a", "done", 100, "完成，用时 92s", 18.2, "会议纪要"),
        ("访谈-咖啡店主理人.m4a", "done", 100, "完成，用时 143s", 26.4, "访谈"),
    ]
    ids = []
    for name, status, prog, msg, dur, _t in jobs:
        jid = db.create_job(name, str(cfg_dir / "in" / name), f"demo-{name}", 1024, dur * 60)
        db.update_job(jid, status=status, progress=prog, message=msg, stage="archived")
        ids.append(jid)

    segs = []
    spk_ids = {"张三": 0, "李四": 1, "王五": 2}
    t = 6000
    for i, (who, text) in enumerate(LINES):
        dur = max(1800, len(text) * 120)
        segs.append({
            "start_ms": t, "end_ms": t + dur,
            "raw_text": text, "clean_text": text, "final_text": text,
            "spk": spk_ids[who], "quality": 1.0, "char_rate": 4.2,
        })
        t += dur + random.randint(200, 900)
    db.save_segments(ids[0], segs)

    for jid in ids[1:]:
        db.save_segments(jid, [{**s, "spk": s["spk"] if s["spk"] < 2 else 0} for s in segs[:6]])

    # 整理稿（每两段合成一段）
    paras = []
    for i in range(0, len(segs), 2):
        chunk = segs[i:i + 2]
        paras.append({
            "idx": i // 2,
            "start_ms": chunk[0]["start_ms"], "end_ms": chunk[-1]["end_ms"],
            "spk": chunk[0]["spk"],
            "text": "".join(c["final_text"] for c in chunk),
        })
    db.save_paragraphs(ids[0], paras)

    summary = {
        "doc_type": "会议纪要",
        "title": "支付链路灰度与下季度预算",
        "topic": "支付灰度进展、风控联调风险、推理成本与预算",
        "summary": "本周支付链路改造进入灰度，错误率明显下降；风控规则引擎版本未对齐，存在联调延期风险。"
                   "下季度预算受推理集群成本上涨影响，计划通过夜间批处理提升利用率。",
        "chapters": [
            {"start_ms": 6000, "title": "支付灰度进展", "summary": "错误率从千分之三降到千分之一点二"},
            {"start_ms": 246000, "title": "风控联调风险", "summary": "规则引擎版本未对齐，可能延期一周"},
            {"start_ms": 420000, "title": "推理成本与预算", "summary": "32 张卡利用率 60%，拟夜间跑批处理"},
        ],
        "decisions": ["灰度继续观察两周，异常则回滚", "预算按增长 20% 申报，留弹性"],
        "todos": [{"task": "风控联调", "owner": "李四"},
                  {"task": "埋点指标定义", "owner": "王五"},
                  {"task": "成本拆分表", "owner": "王五"}],
        "keywords": ["支付灰度", "风控联调", "推理集群", "预算"],
    }
    # 说话人显示名（任务页从 work/<jid>/speakers.json 读）
    import json as _json
    work = cfg_dir / "work" / ids[0]
    work.mkdir(parents=True, exist_ok=True)
    (work / "speakers.json").write_text(
        _json.dumps({"names": {"0": "张三", "1": "李四", "2": "王五"}}, ensure_ascii=False))

    db.save_summary(ids[0], summary)
    db.save_export(ids[0], {"note": "录音笔记/会议纪要/2026/2026-11-03 1000 支付链路灰度与下季度预算"})

    # 声纹：两个虚构人物，各 3 条随机向量（只用于展示信道列）
    import numpy as np
    rng = np.random.default_rng(42)
    for name, n_chan, per in (("张三", 2, 2), ("李四", 1, 3)):
        sid = db.create_speaker(name)
        base = rng.standard_normal(192).astype("float32")
        for c in range(n_chan):
            v0 = base if c == 0 else rng.standard_normal(192).astype("float32")
            for _ in range(per):
                v = v0 + 0.02 * rng.standard_normal(192).astype("float32")
                v /= np.linalg.norm(v)
                db.add_voiceprint(sid, v.tobytes(), 192, source="web", quality=1.0)
    print("演示数据已生成：", cfg_dir)
    print("任务 ID：", ids)
    print("启动：PYTHONPATH=code .venv-funasr/bin/python -m app.cli -c", cfg_dir / "config.yaml",
          "service --port 8091")


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tingji-demo")
    random.seed(7)
    w(target)
    seed(target)
