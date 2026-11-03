"""声学特征：基频（F0）、能量、浊音比例，用于
1) 修正说话人聚类（男女基频差异大，比声纹更稳）；
2) 识别非人声（喇叭/提示音/导航播报）。
"""
from __future__ import annotations

import numpy as np

SR = 16000


def _frames(x: np.ndarray, win: int = 640, hop: int = 320) -> np.ndarray:
    if len(x) < win:
        return np.empty((0, win), dtype=np.float32)
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def f0_track(x: np.ndarray, sr: int = SR, fmin: int = 60, fmax: int = 400) -> np.ndarray:
    """自相关法逐帧估基频，返回浊音帧的 F0 数组（Hz）。"""
    fr = _frames(x)
    if len(fr) == 0:
        return np.array([])
    fr = fr - fr.mean(axis=1, keepdims=True)
    # 加窗
    fr = fr * np.hanning(fr.shape[1])[None, :]
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    out = []
    for f in fr:
        e = float(np.dot(f, f))
        if e < 1e-6:
            continue
        ac = np.correlate(f, f, mode="full")[len(f) - 1:]
        ac = ac / (ac[0] + 1e-9)
        if len(ac) <= lag_max:
            continue
        # 跳过主瓣：从第一个过零点之后找峰，避免总落在最小 lag
        neg = np.where(ac[lag_min:lag_max] < 0)[0]
        start = lag_min + int(neg[0]) if len(neg) else lag_min
        if start >= lag_max:
            continue
        k = int(np.argmax(ac[start:lag_max])) + start
        if ac[k] < 0.25:                       # 周期性不够 → 清音/噪声
            continue
        out.append(sr / k)
    return np.array(out, dtype=np.float32)


def _zcr(x: np.ndarray) -> float:
    """过零率：噪声/清音高，浊音低。"""
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))


def _centroid(x: np.ndarray, sr: int = SR) -> float:
    """频谱质心（Hz）：摩擦音/噪声偏高，元音偏低。"""
    if len(x) < 32:
        return 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    total = spec.sum()
    return float((freqs * spec).sum() / total) if total > 0 else 0.0


def features(clip: np.ndarray, sr: int = SR) -> dict:
    """返回 rms / f0 / voiced / f0_std / zcr / centroid / speech。

    `speech` 是 0–1 的"像人声"打分：浊音比例高、频谱质心落在语音区间、能量不低。
    灵感来自 WhisprRT 的静音三特征（能量 + 过零率 + 频谱质心）。
    """
    if clip.ndim > 1:
        clip = clip.mean(axis=1)
    if len(clip) < int(0.2 * sr):
        return {"rms": 0.0, "f0": 0.0, "voiced": 0.0, "f0_std": 0.0,
                "zcr": 0.0, "centroid": 0.0, "speech": 0.0}
    rms = float(np.sqrt(np.mean(clip ** 2)))
    f0s = f0_track(clip, sr)
    total = max(1, len(_frames(clip)))
    voiced = len(f0s) / total
    zcr = _zcr(clip)
    centroid = _centroid(clip, sr)
    # 人声典型：质心 200–3500Hz、浊音比 >0.15、过零率 <0.25
    if rms < 0.004 or centroid < 80:
        speech = 0.0                      # 静音/直流：直接判 0
    else:
        c_ok = 1.0 if 150 <= centroid <= 4000 else max(0.0, 1 - abs(centroid - 2000) / 6000)
        v_ok = min(1.0, voiced / 0.3)
        z_ok = 1.0 if zcr < 0.25 else max(0.0, 1 - (zcr - 0.25) / 0.35)
        speech = round(max(0.0, min(1.0, 0.5 * v_ok + 0.3 * c_ok + 0.2 * z_ok)), 2)
    return {
        "rms": round(rms, 4),
        "f0": float(np.median(f0s)) if len(f0s) else 0.0,
        "voiced": round(voiced, 2),
        "f0_std": float(np.std(f0s)) if len(f0s) else 0.0,
        "zcr": round(zcr, 3),
        "centroid": round(centroid, 1),
        "speech": speech,
    }


def split_by_f0(segs: list[dict], labels: np.ndarray, boundary: float = 165.0,
                min_side: int = 3) -> np.ndarray:
    """当一个大簇内部基频呈明显双峰时，按 F0 拆成两人。

    男女对话中位基频通常分别在 180–260 / 90–150 Hz，比情感化语音的声纹更稳。
    """
    out = labels.copy()
    for lab in sorted(set(labels.tolist())):
        if lab < 0:
            continue
        idx = [i for i, l in enumerate(labels) if l == lab]
        f0 = np.array([segs[i].get("f0") or 0 for i in idx])
        f0 = f0[f0 > 0]
        if len(f0) < min_side * 2:
            continue
        low = int((f0 < boundary).sum())
        high = int((f0 >= boundary).sum())
        if low >= min_side and high >= min_side and min(low, high) / len(f0) >= 0.2:
            new = labels.max() + 1
            for i in idx:
                v = segs[i].get("f0") or 0
                if v > 0 and v >= boundary:
                    out[i] = new
    # 压紧编号
    uniq = {v: i for i, v in enumerate(sorted(set(out[out >= 0].tolist())))}
    for i in range(len(out)):
        if out[i] >= 0:
            out[i] = uniq[int(out[i])]
    return out


NAV_WORDS = ("右转", "左转", "前方", "公里", "米处", "到达", "导航", "驶入", "驶出",
             "停车", "路口", "限速", "摄像头", "车道", "红绿灯", "请沿", "高架", "匝道",
             "掉头", "靠右", "靠左", "直行", "目的地", "已到达", "油量", "胎压")
# 停车场/商场广播常见话术
PA_WORDS = ("照顾", "老人", "小孩", "贵重物品", "请勿", "注意安全", "停车场", "出入口",
            "缴费", "扫码", "月卡", "临停", "限高", "洗手间", "顾客", "请保管", "温馨提示",
            "闭店", "营业", "电梯", "扶梯", "禁止")


def looks_like_nav(text: str) -> bool:
    t = text or ""
    if not t or len(t) > 45:
        return False
    if any(w in t for w in PA_WORDS):
        return True
    return any(w in t for w in NAV_WORDS)


def looks_like_noise(seg: dict, rms_median: float = 0.0) -> bool:
    """喇叭/提示音/噪声。

    两条判据（WhisprRT 的静音三特征启发）：
      1) 极短文本 + 明显更响 + 无浊音（喇叭、提示音）；
      2) 人声打分很低（<0.25）且文本极短（摩擦音、键盘、路面噪声被误当语音）。
    """
    text = (seg.get("raw_text") or "").strip()
    voiced = seg.get("voiced") or 0
    rms = seg.get("rms") or 0
    speech = seg.get("speech")
    loud = rms_median > 0 and rms > rms_median * 1.8
    if loud and voiced < 0.2 and len(text) <= 3:
        return True
    if speech is not None and float(speech) < 0.25 and len(text) <= 4:
        return True
    return False


def cap_clusters(segs: list[dict], labels: np.ndarray, max_n: int) -> np.ndarray:
    """人数上限约束：保留最大的 max_n 个簇，其余并到基频最接近的簇。"""
    if not max_n or max_n <= 0:
        return labels
    keys = [k for k in sorted(set(labels.tolist())) if k >= 0]
    if len(keys) <= max_n:
        return labels
    counts = {k: int((labels == k).sum()) for k in keys}
    keep = [k for k, _ in sorted(counts.items(), key=lambda x: -x[1])[:max_n]]
    drop = [k for k in keys if k not in keep]

    def centre(k):
        f = [segs[i].get("f0") or 0 for i in range(len(labels)) if labels[i] == k]
        f = [x for x in f if x > 0]
        return float(np.median(f)) if f else 0.0

    cents = {k: centre(k) for k in keep}
    out = labels.copy()
    for k in drop:
        c = centre(k)
        cand = [x for x in keep if cents[x] > 0]
        tgt = min(cand, key=lambda x: abs(cents[x] - c)) if (cand and c > 0) else keep[0]
        out[labels == k] = tgt
    uniq = {v: i for i, v in enumerate(sorted(set(out[out >= 0].tolist())))}
    for i in range(len(out)):
        if out[i] >= 0:
            out[i] = uniq[int(out[i])]
    return out


def speech_ratio(audio: np.ndarray, sr: int = SR, win: float = 0.03, hop: float = 0.01) -> float:
    """整段音频里"像人声"的时间占比（0–1）。

    判据：帧能量高于自适应噪声底 + 过零率不过高。
    用途：整段无人声时直接跳过 ASR（避免白跑几十分钟），
    也用于日志里记录"这段音频有多少是有效语音"。
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    w, h = max(1, int(win * sr)), max(1, int(hop * sr))
    if len(audio) < w * 4:
        return 0.0
    n = 1 + (len(audio) - w) // h
    idx = np.arange(w)[None, :] + h * np.arange(n)[:, None]
    frames = audio[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    floor = max(0.002, 0.08 * float(np.percentile(rms, 90)))
    sign = np.signbit(frames).astype(np.int8)
    zcr = np.mean(np.abs(np.diff(sign, axis=1)), axis=1)
    voiced = (rms > floor) & (zcr < 0.35)
    return float(np.mean(voiced))
