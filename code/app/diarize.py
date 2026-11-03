"""说话人聚类：在线分配 → 簇合并 → 双轮质心 → 短句回填 → 重叠嫌疑。"""
from __future__ import annotations

import numpy as np


def cluster(embs: np.ndarray, durations: np.ndarray, threshold: float = 0.40,
            seed_dist: float = 0.35, assign_cos: float = 0.55,
            n_clusters: int | None = None) -> np.ndarray:
    """两段式说话人聚类。

    1) 用较严格的 average-linkage 距离切出「核心簇」（只含高置信同人片段）；
    2) 其余片段按"与核心簇质心余弦 ≥ assign_cos"就近归属，否则自成一簇。

    真实录音里短片段/远场片段向量很弱，直接全局切会切成一堆单点；
    先取核心、再吸附，能显著稳住簇数。
    threshold 仅保留兼容（等同 seed_dist 的宽松版）。
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    n = len(embs)
    labels = np.full(n, -1, dtype=int)
    valid = [i for i, e in enumerate(embs) if e is not None and float(np.linalg.norm(e)) > 1e-6]
    if not valid:
        return labels
    if len(valid) == 1:
        labels[valid[0]] = 0
        return labels

    M = np.stack([embs[i] for i in valid])
    Z = linkage(M, method="average", metric="cosine")
    if n_clusters and n_clusters >= 1:
        # 用户指定人数：直接按最大簇数切，并用其结果（不再做两段式"离群点自成簇"）
        raw = fcluster(Z, t=min(n_clusters, len(valid)), criterion="maxclust")
        uniq = {v: i for i, v in enumerate(sorted(set(raw.tolist())))}
        for j, lab in zip(valid, raw):
            labels[j] = uniq[int(lab)]
        return labels
    raw = fcluster(Z, t=seed_dist, criterion="distance")

    # 核心簇质心（只保留 ≥1s 的片段参与质心，避免短片段污染）
    cents, core = [], []
    for k in sorted(set(raw.tolist())):
        idx = [j for j, r in enumerate(raw) if r == k]
        long_idx = [j for j in idx if durations[valid[j]] >= 1.0] or idx
        c = np.mean(M[long_idx], axis=0)
        cents.append(c / (np.linalg.norm(c) + 1e-9))
        core.append((k, idx))
    C = np.stack(cents)

    assign = np.full(len(valid), -1, dtype=int)
    for j in range(len(valid)):
        sims = C @ M[j]
        best = int(np.argmax(sims))
        if float(sims[best]) >= assign_cos:
            assign[j] = best
        else:
            assign[j] = len(cents) + j          # 独立簇
    uniq = {v: i for i, v in enumerate(sorted(set(assign.tolist())))}
    for j, i in enumerate(valid):
        labels[i] = uniq[int(assign[j])]
    return labels


def centroids(embs: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out = {}
    for k in sorted(set(labels.tolist())):
        if k < 0:
            continue
        idx = np.where(labels == k)[0]
        vs = [embs[i] for i in idx if embs[i] is not None]
        if not vs:
            continue
        c = np.mean(vs, axis=0)
        out[k] = c / (np.linalg.norm(c) + 1e-9)
    return out


def merge_by_centroid(labels: np.ndarray, cents: dict[int, np.ndarray], threshold: float = 0.72) -> np.ndarray:
    """质心余弦 ≥ threshold 的簇做连通分量合并。"""
    keys = list(cents)
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if float(cents[a] @ cents[b]) >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    remap = {k: find(k) for k in keys}
    out = labels.copy()
    for i, l in enumerate(labels):
        if l >= 0:
            out[i] = remap.get(int(l), int(l))
    # 重新压紧编号
    uniq = {v: i for i, v in enumerate(sorted(set(out[out >= 0].tolist())))}
    for i in range(len(out)):
        if out[i] >= 0:
            out[i] = uniq[int(out[i])]
    return out


def reassign(embs: np.ndarray, labels: np.ndarray, cents: dict[int, np.ndarray]) -> np.ndarray:
    """把每段重新分配给最近质心（只对已聚类的段）。"""
    out = labels.copy()
    keys = sorted(cents)
    if not keys:
        return out
    mat = np.stack([cents[k] for k in keys])
    for i, e in enumerate(embs):
        if e is None or labels[i] < 0:
            continue
        out[i] = keys[int(np.argmax(mat @ e))]
    return out


def fill_short(segs: list[dict], labels: np.ndarray, min_sec: float = 1.0) -> np.ndarray:
    """短句不参与聚类：前后同簇则继承，否则继承前一句。"""
    out = labels.copy()
    n = len(segs)
    for i in range(n):
        dur = (segs[i]["end_ms"] - segs[i]["start_ms"]) / 1000
        if dur >= min_sec and out[i] >= 0:
            continue
        prev = next((out[j] for j in range(i - 1, -1, -1) if out[j] >= 0), -1)
        nxt = next((out[j] for j in range(i + 1, n) if out[j] >= 0), -1)
        out[i] = prev if (prev >= 0 and prev == nxt) else (prev if prev >= 0 else nxt)
    return out


def drop_small_clusters(embs: np.ndarray, labels: np.ndarray, durations: np.ndarray,
                        min_total_sec: float = 5.0, min_ratio: float = 0.03) -> np.ndarray:
    """把总时长过小的簇（噪声/误聚）并到最近的其它簇。"""
    out = labels.copy()
    keys = [k for k in sorted(set(labels.tolist())) if k >= 0]
    if len(keys) < 2:
        return out
    totals = {k: float(sum(d for d, l in zip(durations, labels) if l == k)) for k in keys}
    speech = sum(totals.values()) or 1.0
    kept = [k for k in keys if totals[k] >= max(min_total_sec, min_ratio * speech)]
    if not kept:
        return out
    cents = {k: None for k in kept}
    for k in kept:
        idx = [i for i, l in enumerate(labels) if l == k and embs[i] is not None]
        if idx:
            c = np.mean([embs[i] for i in idx], axis=0)
            cents[k] = c / (np.linalg.norm(c) + 1e-9)
    kept = [k for k in kept if cents[k] is not None]
    for i, l in enumerate(labels):
        if l in kept or embs[i] is None:
            continue
        sims = [(float(embs[i] @ cents[k]), k) for k in kept]
        out[i] = max(sims)[1]
    return out


def overlap_scores(audio: np.ndarray, segs: list[dict], sr: int, embed_fn) -> np.ndarray:
    """每段的段内 1s 窗嵌入一致性（NaN 表示不可计算）。"""
    scores = np.full(len(segs), np.nan)
    win, hop = int(1.0 * sr), int(0.5 * sr)
    for i, s in enumerate(segs):
        a, b = int(s["start_ms"] / 1000 * sr), int(s["end_ms"] / 1000 * sr)
        clip = audio[a:b]
        if len(clip) < win + hop:
            continue
        es = [embed_fn(clip[j:j + win]) for j in range(0, len(clip) - win + 1, hop)]
        es = [e for e in es if e is not None]
        if len(es) < 2:
            continue
        sims = [float(es[x] @ es[y]) for x in range(len(es)) for y in range(x + 1, len(es))]
        scores[i] = float(np.mean(sims))
    return scores


def adaptive_overlap_flags(scores: np.ndarray, durations: np.ndarray,
                           floor: float = 0.45, k: float = 3.0,
                           min_sec: float = 2.0, pct: float = 3.0) -> np.ndarray:
    """相对阈值：低于「P{pct} 与 中位数−k×MAD 的较大者」且不低于绝对下限。

    真实录音整体一致性远低于合成音频（实测中位数 0.65），固定阈值会误报 30%；
    而 MAD 在长音频上又会被大量弱片段带偏，因此同时用分位数兜底。
    """
    flags = np.zeros(len(scores), dtype=int)
    ok = ~np.isnan(scores)
    if ok.sum() < 8:
        return flags
    v = scores[ok]
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)) + 1e-6)
    thr = max(floor, float(np.percentile(v, pct)), med - k * mad)
    for i, sc in enumerate(scores):
        if not np.isnan(sc) and sc < thr and durations[i] >= min_sec:
            flags[i] = 1
    return flags


def merge_until(embs: np.ndarray, labels: np.ndarray, cap: int) -> tuple[np.ndarray, int]:
    """合并最相似的簇，直到簇数 ≤ cap。返回 (labels, 合并次数)。

    自动聚类在低信噪比音频上容易裂成几十个簇（实测 121 段裂成 61 人），
    用"质心最相似就先并"可以稳住。
    """
    labels = labels.copy()
    n_merged = 0
    while True:
        keys = sorted(set(int(x) for x in labels if x >= 0))
        if len(keys) <= cap or len(keys) <= 1:
            break
        cents = centroids(embs, labels)
        best, best_sim = None, -2.0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = cents[keys[i]], cents[keys[j]]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na == 0 or nb == 0:
                    continue
                sim = float((a / na) @ (b / nb))
                if sim > best_sim:
                    best, best_sim = (keys[j], keys[i]), sim
        if not best:
            break
        src, dst = best
        labels[labels == src] = dst
        n_merged += 1
    return labels, n_merged


def centroids_clean(embs: np.ndarray, labels: np.ndarray, segs: list[dict],
                    min_dur: float = 2.0, min_purity: float = 0.99) -> dict[int, np.ndarray]:
    """只用"较长且说话人纯度高"的段算质心；样本不足时自动退回全部段。

    实测：在跨信道录音上，用全部段算质心会把两个人都"稀释"，
    只用干净段后，目标人物相似度从 0.324 提到 0.357，区分度更好。
    """
    dur = np.array([(s["end_ms"] - s["start_ms"]) / 1000 for s in segs])
    pur = np.array([float(s.get("spk_purity") or 0) for s in segs])
    mask = (dur >= min_dur) & (pur >= min_purity)
    out = {}
    for k in sorted(set(int(x) for x in labels if x >= 0)):
        idx = mask & (labels == k)
        if idx.sum() < 3:
            idx = labels == k
        if idx.sum() == 0:
            continue
        v = embs[idx].mean(axis=0)
        n = np.linalg.norm(v)
        out[k] = v / n if n else v
    return out
