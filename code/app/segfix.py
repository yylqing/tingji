"""说话人标签后处理：平滑 + 跨说话人重切。

对应 VoiceStudio `segmentation.py` 里 `_fill_and_smooth` / `_resplit_core` 的思路，
但我们只有**句级**时间戳，所以用"前后半段嵌入 + 能量谷"做近似重切：

1. `smooth_labels`：A-B-A 单点翻转回填 + 短 run 合并（抑制聚类抖动）；
2. `resplit_by_speaker`：一句里前/后半段指向不同说话人时，在停顿处切成两段。
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("tingji.segfix")

SR = 16000


# ---------------------------------------------------------------- 平滑
def smooth_labels(labels: np.ndarray, durations: np.ndarray | None = None,
                  min_run: int = 1, max_smooth_dur: float = 2.5) -> np.ndarray:
    """A-B-A 平滑：夹在两个相同标签之间的孤立标签改回邻标签。

    只平滑**短段**（默认 <2.5s）——聚类噪声几乎都出在短段上；
    长段是可靠判断，A-B-A-B 的快速对白不应被抹平。
    """
    out = labels.copy()
    n = len(out)

    def short(i: int) -> bool:
        if durations is None:
            return True
        return float(durations[i]) <= max_smooth_dur

    # 单点翻转
    for i in range(1, n - 1):
        if out[i - 1] == out[i + 1] != out[i] and out[i - 1] >= 0 and short(i):
            out[i] = out[i - 1]

    if min_run > 1:
        i = 0
        while i < n:
            j = i
            while j + 1 < n and out[j + 1] == out[i]:
                j += 1
            run_len = j - i + 1
            left = out[i - 1] if i > 0 else -1
            right = out[j + 1] if j + 1 < n else -1
            if (run_len <= min_run and left == right and left >= 0 and out[i] != left
                    and all(short(k) for k in range(i, j + 1))):
                out[i:j + 1] = left
            i = j + 1
    return out


# ---------------------------------------------------------------- 重切
def find_split(clip: np.ndarray, embed_fn, centroids: dict[int, np.ndarray],
               sr: int = SR, win: float = 1.2, hop: float = 0.4,
               own_threshold: float = 0.6, min_side: int = 2) -> int | None:
    """用滑窗嵌入找"换人"的最佳切点（不依赖静音）。

    对每个候选切点 k：左侧窗口稳定归 A、右侧稳定归 B（A≠B），
    且两侧对各自簇的平均相似度都要够高。返回切点样本下标。
    """
    if not centroids or len(centroids) < 2:
        return None
    keys = sorted(centroids)
    mat = np.stack([centroids[k] for k in keys])
    w, h = int(win * sr), int(hop * sr)
    if len(clip) < (min_side * 2) * h + w:
        return None
    centers, labels, sims = [], [], []
    for st in range(0, len(clip) - w + 1, h):
        e = embed_fn(clip[st:st + w])
        if e is None:
            continue
        sc = mat @ e
        j = int(np.argmax(sc))
        centers.append(st + w // 2)
        labels.append(keys[j])
        sims.append(float(sc[j]))
    n = len(labels)
    if n < min_side * 2 + 1:
        return None
    best, best_score = None, -1.0
    for k in range(min_side, n - min_side + 1):
        left, right = labels[:k], labels[k:]
        a = max(set(left), key=left.count)
        b = max(set(right), key=right.count)
        if a == b:
            continue
        purity_l = left.count(a) / len(left)
        purity_r = right.count(b) / len(right)
        if purity_l < 0.7 or purity_r < 0.7:
            continue
        s_left = float(np.mean([sims[i] for i in range(k) if labels[i] == a]))
        s_right = float(np.mean([sims[i] for i in range(k, n) if labels[i] == b]))
        if s_left < own_threshold or s_right < own_threshold:
            continue
        # 纯度优先（真实换人点两侧纯度同时接近 1），相似度仅作平手时的微弱加权
        score = (purity_l + purity_r) + 0.05 * (s_left + s_right)
        if score > best_score:
            best, best_score = (centers[k - 1] + centers[k]) // 2, score
    return best


PUNCT = "，。！？；、,.!?;"


def _split_text_by_ratio(text: str, ratio: float) -> tuple[str, str]:
    """按比例找最近标点切分；没有标点就按字数硬切。"""
    if not text:
        return "", ""
    target = max(1, min(len(text) - 1, int(round(len(text) * ratio))))
    best, best_d = target, len(text)
    for i, ch in enumerate(text):
        if ch in PUNCT:
            d = abs(i + 1 - target)
            if d < best_d:
                best, best_d = i + 1, d
    best = max(1, min(len(text) - 1, best))
    return text[:best].strip(), text[best:].strip()


def resplit_by_speaker(segs: list[dict], labels: np.ndarray, audio: np.ndarray,
                       embed_fn, centroids: dict[int, np.ndarray],
                       min_duration: float = 3.0, max_split: int = 40,
                       sim12_threshold: float = 0.55, own_threshold: float = 0.6,
                       min_chars: int = 10, sr: int = SR) -> tuple[list[dict], np.ndarray]:
    """把"一句里两个人"的段按停顿切成两段。返回 (新段列表, 新标签)。

    判据（三者同时满足才切）：
      1) 前后半段嵌入相似度低（< sim12_threshold）；
      2) 两半各自最近的簇不同；
      3) 两半对各自簇的相似度都够高（> own_threshold）。
    """
    if not centroids or len(centroids) < 2:
        return segs, labels
    keys = sorted(centroids)
    mat = np.stack([centroids[k] for k in keys])

    out_segs: list[dict] = []
    out_labels: list[int] = []
    n_split = 0

    for seg, lab in zip(segs, labels):
        dur = (seg["end_ms"] - seg["start_ms"]) / 1000
        text = seg.get("raw_text") or ""
        if (n_split >= max_split or dur < min_duration or len(text) < min_chars
                or seg.get("speaker_kind") in ("nav", "noise")):
            out_segs.append(seg)
            out_labels.append(int(lab))
            continue

        a = int(seg["start_ms"] / 1000 * sr)
        b = int(seg["end_ms"] / 1000 * sr)
        clip = audio[a:b]
        if len(clip) < min_duration * sr * 0.8:
            out_segs.append(seg)
            out_labels.append(int(lab))
            continue

        mid = find_split(clip, embed_fn, centroids, sr, own_threshold=own_threshold)
        if mid is None:
            out_segs.append(seg)
            out_labels.append(int(lab))
            continue
        h1, h2 = clip[:mid], clip[mid:]
        e1, e2 = embed_fn(h1), embed_fn(h2)
        if e1 is None or e2 is None:
            out_segs.append(seg)
            out_labels.append(int(lab))
            continue
        s1, s2 = mat @ e1, mat @ e2
        k1, k2 = keys[int(np.argmax(s1))], keys[int(np.argmax(s2))]
        if k1 == k2:
            out_segs.append(seg)
            out_labels.append(int(lab))
            continue

        split_ms = seg["start_ms"] + int(mid / sr * 1000)
        ratio = mid / max(len(clip), 1)
        left, right = _split_text_by_ratio(text, ratio)
        if len(left) < 4 or len(right) < 4:
            out_segs.append(seg)
            out_labels.append(int(lab))
            continue

        base = {k: v for k, v in seg.items() if k not in ("raw_text", "clean_text", "final_text")}
        s_left = {**base, "start_ms": seg["start_ms"], "end_ms": split_ms, "raw_text": left}
        s_right = {**base, "start_ms": split_ms, "end_ms": seg["end_ms"], "raw_text": right}
        out_segs.extend([s_left, s_right])
        out_labels.extend([k1, k2])
        n_split += 1

    if n_split:
        log.info("跨说话人重切 %d 段", n_split)
    return out_segs, np.array(out_labels, dtype=int)


def stats(labels: np.ndarray) -> dict:
    vals, cnt = np.unique(labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnt) if v >= 0}
