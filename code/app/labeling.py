"""说话人校对：抽样标注 → 与自动聚类比对 → 给出可量化的准确率。

目的：在此之前所有聚类参数（F0 阈值、簇合并、平滑）都没有真值可校准；
这里用最少的人工投入（每簇听 2–3 段、每段 5 秒）换一个可复现的数字。
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict

MIXED = "混合（多人）"


def label_queue(segs: list[dict], per_cluster: int = 3, min_dur: float = 1.2,
                seed: int = 7) -> list[dict]:
    """按簇分层抽样：每个簇取若干"中等长度、内容较多"的段，便于人耳判断。"""
    by: dict[int, list[dict]] = defaultdict(list)
    for s in segs:
        dur = (s["end_ms"] - s["start_ms"]) / 1000
        text = (s.get("raw_text") or "").strip()
        if dur < min_dur or len(text) < 4:
            continue
        if s.get("speaker_kind") in ("nav", "noise"):
            continue
        purity = s.get("spk_purity")
        if purity is not None and float(purity) < 0.8:
            continue        # 分离器自己都觉得不纯的段，先别让人去猜
        by[int(s["spk"])].append(s)

    rng = random.Random(seed)
    out: list[dict] = []
    for spk in sorted(by):
        pool = sorted(by[spk], key=lambda x: -len(x.get("raw_text") or ""))
        pool = pool[: max(per_cluster * 3, per_cluster)]
        pick = rng.sample(pool, min(per_cluster, len(pool)))
        for s in pick:
            out.append({
                "idx": s["idx"], "spk": int(spk), "start_ms": s["start_ms"],
                "end_ms": min(s["end_ms"], s["start_ms"] + 6000),
                "text": s.get("raw_text") or "",
            })
    rng.shuffle(out)
    return out


def accuracy(segs: list[dict], labels: dict[int, str]) -> dict:
    """把人工标签与自动簇比对：每个簇映射到出现最多的人工标签，再算准确率。"""
    if not labels:
        return {"labeled": 0, "mixed": 0, "accuracy": None, "mapping": {}, "confusion": {}}
    by_idx = {int(s["idx"]): s for s in segs}
    conf: dict[tuple[int, str], int] = defaultdict(int)
    for idx, lab in labels.items():
        s = by_idx.get(int(idx))
        if not s or s.get("spk") is None or int(s["spk"]) < 0:
            continue
        conf[(int(s["spk"]), str(lab))] += 1
    if not conf:
        return {"labeled": 0, "mixed": 0, "accuracy": None, "mapping": {}, "confusion": {}}

    # 每个簇取多数人工标签（冲突时按计数降序贪心独占）
    per_cluster: dict[int, Counter] = defaultdict(Counter)
    for (spk, lab), n in conf.items():
        if lab == MIXED:
            continue
        per_cluster[spk][lab] += n
    # 穷举最优映射：簇数 ≤6、每簇候选 ≤3，组合数很小
    from itertools import product
    keys = sorted(per_cluster)
    cands = {}
    for k in keys:
        c = [lab for lab, _ in sorted(per_cluster[k].items(), key=lambda kv: (-kv[1], kv[0]))]
        cands[k] = c[:3] or [""]
    best, best_key = None, None
    for combo in product(*[cands[k] for k in keys]):
        score = sum(conf.get((k, lab), 0) for k, lab in zip(keys, combo))
        distinct = len(set(combo))
        key = (score, distinct)          # 先最大化命中数，再倾向于不重复
        if best_key is None or key > best_key:
            best, best_key = combo, key
    mapping = dict(zip(keys, best or [""] * len(keys)))

    mixed = sum(n for (spk, lab), n in conf.items() if lab == MIXED)
    judged = {k: v for k, v in conf.items() if k[1] != MIXED}
    total = sum(judged.values())
    correct = sum(n for (spk, lab), n in judged.items() if mapping.get(spk) == lab)
    return {
        "labeled": total,
        "mixed": mixed,
        "accuracy": round(correct / total, 3) if total else None,
        "mapping": {str(k): v for k, v in sorted(mapping.items())},
        "per_cluster": {
            str(spk): {"labeled": sum(c.values()), "top": c.most_common(1)[0][0],
                       "purity": round(c.most_common(1)[0][1] / sum(c.values()), 2)}
            for spk, c in sorted(per_cluster.items())
        },
        "confusion": {f"{spk}|{lab}": n for (spk, lab), n in sorted(conf.items())},
    }
