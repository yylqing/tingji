"""帧级说话人分离（sherpa-onnx + pyannote 分割模型）。

相比"整段算一个声纹向量再聚类"，帧级分割能给出**逐段 turn**，
因此可以判断"这句 ASR 里其实有两个人"，并在换人处切开。

实测（12.7 分钟情绪录音）：37s，RTF 0.05，170 个 turn，2 人。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("tingji.diar_sherpa")

from .settings import ROOT as _ROOT

# 模型路径按项目根解析：服务的工作目录是 code/，相对路径会找不到
SEG_MODEL = str(_ROOT / ".cache/sherpa/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx")
EMB_MODEL = str(_ROOT / ".cache/sherpa/campplus.onnx")


def available(root: Path | None = None) -> bool:
    base = root or Path.cwd()
    return (base / SEG_MODEL).exists() and (base / EMB_MODEL).exists()


def _auto_cluster(turns: list[tuple[float, float]], audio: np.ndarray, sr: int,
                  embed_fn, max_k: int = 6, min_turn: float = 1.0,
                  drop_share: float = 0.05) -> list[int]:
    """对 turn 做嵌入聚类，用轮廓系数选 K，并丢掉只占极小比例的小簇。

    实测：sherpa 自带的自动聚类（num_clusters=-1）在这段低信噪比录音上会裂成
    几十个簇（阈值 0.9 仍有 17 人）；而按 turn 聚类 + 轮廓系数选 K 能稳定落到
    "两个说话人 + 一簇噪声"的形态。
    """
    from .engines import SpeakerEngine
    from . import diarize as dia
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return [0] * len(turns)

    spk = SpeakerEngine(EMB_MODEL)
    idx = [i for i, (st, en) in enumerate(turns) if en - st >= min_turn]
    if len(idx) < 6:
        return [0] * len(turns)
    E = []
    for i in idx:
        st, en = turns[i]
        e = spk.embed(audio[int(st * sr):int(en * sr)])
        if e is None:
            return [0] * len(turns)
        E.append(e)
    E = np.stack(E)
    D = np.array([turns[i][1] - turns[i][0] for i in idx])

    best_lab, best_sil = None, -2.0
    for k in range(2, min(max_k, len(idx) // 3) + 1):
        lab = dia.cluster(E, D, 0.4, n_clusters=k)
        sizes = np.bincount(lab[lab >= 0])
        if (sizes >= 3).sum() < 2:
            continue
        try:
            sil = float(silhouette_score(E, lab, metric="cosine"))
        except Exception:
            continue
        if sil > best_sil:
            best_lab, best_sil = lab, sil
    if best_lab is None:
        return [0] * len(turns)

    # 丢掉占比过小的簇，并到最近质心
    total = float(D.sum())
    cents = dia.centroids(E, best_lab)
    keep = {k_: v for k_, v in cents.items()
            if float(D[best_lab == k_].sum()) / max(total, 1e-6) >= drop_share} or cents
    keys = sorted(keep)
    M = np.stack([keep[k_] for k_ in keys])
    for j in range(len(best_lab)):
        best_lab[j] = keys[int(np.argmax(M @ E[j]))]
    labels: list[int | None] = []
    for (st, en) in turns:
        if en - st < 0.3:
            labels.append(None)
            continue
        e = spk.embed(audio[int(st * sr):int(en * sr)])
        labels.append(int(keys[int(np.argmax(M @ e))]) if e is not None else None)
    # 短 turn / 提取失败：归到时间上最近的已标注 turn
    for i, v in enumerate(labels):
        if v is not None:
            continue
        best, best_d = 0, 1e9
        for j, w in enumerate(labels):
            if w is None:
                continue
            d = abs(turns[j][0] - turns[i][0])
            if d < best_d:
                best, best_d = w, d
        labels[i] = best
    return labels


def mask_non_speech(audio: np.ndarray, speech: list[tuple[float, float]], sr: int = 16000,
                    recovery_ratio: float = 0.08, min_gap: float = 1.0,
                    max_gap: float = 20.0) -> np.ndarray:
    """把 VAD/ASR 判定为非语音的部分置 0（Brevia 的 masked diarization）。

    同时做**安静语音兜底**：夹在两段语音之间、时长 1–20 秒、电平不低于整体语音
    8% 的"空隙"保留下来——音乐里的轻声、电话音如果被整段抹掉，说话人就凭空少了。
    """
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not speech:
        return audio
    out = np.zeros_like(audio)
    speech = sorted(speech)
    # 整体语音电平（用于兜底判定）
    levels = []
    for st, en in speech:
        a, b = int(max(0, st) * sr), int(min(len(audio) / sr, en) * sr)
        if b > a:
            levels.append(float(np.sqrt(np.mean(audio[a:b] ** 2) + 1e-12)))
    base = float(np.median(levels)) if levels else 0.0

    prev_end = 0.0
    for st, en in speech:
        a, b = int(max(0, st) * sr), int(min(len(audio) / sr, en) * sr)
        # 兜底：上一段与本段之间的空隙
        gap_a, gap_b = int(prev_end * sr), a
        if base > 0 and gap_b > gap_a:
            dur = (gap_b - gap_a) / sr
            if min_gap <= dur <= max_gap:
                lvl = float(np.sqrt(np.mean(audio[gap_a:gap_b] ** 2) + 1e-12))
                if lvl / base >= recovery_ratio:
                    out[gap_a:gap_b] = audio[gap_a:gap_b]
        if b > a:
            out[a:b] = audio[a:b]
        prev_end = max(prev_end, en)
    return out


def centroids_from_turns(audio: np.ndarray, turns: list[tuple[float, float, int]],
                         embed_fn, sr: int = 16000, window_sec: float = 15.0) -> dict[int, np.ndarray]:
    """按"每个 turn 切 15 秒窗口"提声纹再求均值（Brevia 的做法）。

    比"每个 ASR 段算一个向量"稳：长 turn 不会因为整段平均而被稀释，
    短段（<1.5s）直接跳过，不污染质心。
    """
    buckets: dict[int, list[np.ndarray]] = {}
    for st, en, spk in turns:
        a, b = int(max(0, st) * sr), int(min(len(audio) / sr, en) * sr)
        if b - a < int(1.5 * sr):
            continue
        step = max(int(window_sec * sr), int(1.5 * sr))
        for w0 in range(a, b, step):
            w1 = min(b, w0 + step)
            if w1 - w0 < int(1.5 * sr):
                continue
            e = embed_fn(audio[w0:w1])
            if e is not None:
                buckets.setdefault(int(spk), []).append(np.asarray(e, dtype=np.float32))
    out: dict[int, np.ndarray] = {}
    for spk, vs in buckets.items():
        if not vs:
            continue
        m = np.mean(np.stack(vs), axis=0)
        n = np.linalg.norm(m)
        if n:
            out[spk] = m / n
    return out


def run(audio: np.ndarray, sr: int = 16000, num_speakers: int = 0,
        threshold: float = 0.5, min_on: float = 0.25, min_off: float = 0.3,
        auto: bool = True) -> list[tuple[float, float, int]]:
    """返回 [(start_s, end_s, speaker_id), ...]，按时间排序。

    num_speakers>1 时按指定人数切；否则用 turn 聚类 + 轮廓系数自动定 K。
    """
    import sherpa_onnx

    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL)),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB_MODEL),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers and num_speakers > 1 else 2,
            threshold=threshold),
        min_duration_on=min_on,
        min_duration_off=min_off,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    raw = sd.process(audio).sort_by_start_time()
    turns = [(float(x.start), float(x.end)) for x in raw]
    if not turns:
        return []
    if num_speakers and num_speakers > 1:
        labels = [int(x.speaker) for x in raw]
    elif auto:
        try:
            from .engines import SpeakerEngine
            eng = SpeakerEngine(EMB_MODEL)
            labels = _auto_cluster(turns, audio, sr, eng.embed)
            log.info("自动人数：%d", len(set(labels)))
        except Exception as e:
            log.warning("自动人数估计失败，沿用 sherpa 结果: %s", e)
            labels = [int(x.speaker) for x in raw]
    else:
        labels = [int(x.speaker) for x in raw]
    return [(st, en, labels[i]) for i, (st, en) in enumerate(turns)]


def _split_text(text: str, ratios: list[float]) -> list[str]:
    """按累计比例把文本切成若干段，尽量在标点处断开。"""
    if not text:
        return [""] * len(ratios)
    import re
    marks = "，。！？；、,.!?;"
    pieces, prev = [], 0
    for i, r in enumerate(ratios[:-1]):
        target = max(prev + 1, min(len(text) - 1, int(round(len(text) * r))))
        best, best_d = target, 10 ** 9
        for j in range(prev + 1, min(len(text), target + 8)):
            if text[j - 1] in marks and abs(j - target) < best_d:
                best, best_d = j, abs(j - target)
        pieces.append(text[prev:best].strip())
        prev = best
    pieces.append(text[prev:].strip())
    return pieces


def apply(segs: list[dict], turns: list[tuple[float, float, int]],
          min_piece: float = 0.25, max_pieces: int = 5) -> tuple[list[dict], np.ndarray]:
    """按 turn 把 ASR 段切开或整体归人。

    - 段内只有一个说话人 → 直接归人；
    - 段内跨多人 → 在换人处切开，文本按时间比例、就近标点分配；
    - 切出的碎片若 < min_piece 秒则并入相邻片（避免产生碎渣）。
    """
    out: list[dict] = []
    labels: list[int] = []
    PURITY: dict[int, float] = {}

    for seg in segs:
        a, b = seg["start_ms"] / 1000, seg["end_ms"] / 1000
        ov = []
        for (st, en, sp) in turns:
            o = min(b, en) - max(a, st)
            if o > 0.05:
                ov.append([st, en, sp, o])
        if not ov:
            # 没有 turn 覆盖（静音间隙里的短句）：归给时间上最近的 turn
            if turns:
                near = min(turns, key=lambda t: min(abs(t[0] - b), abs(a - t[1])))
                out.append(seg)
                labels.append(int(near[2]))
            else:
                out.append(seg)
                labels.append(int(seg.get("spk") if seg.get("spk") is not None else 0))
            continue

        # 只保留占比够的 turn
        total = sum(x[3] for x in ov)
        ov = [x for x in ov if x[3] / total >= 0.08]
        speakers = {x[2] for x in ov}

        if len(speakers) == 1:
            out.append(seg)
            labels.append(int(ov[0][2]))
            seg["spk_purity"] = round(ov[0][3] / max(total, 1e-6), 2)
            continue

        # 构造片段边界：turn 起止裁剪到段内
        bounds = []
        for (st, en, sp, _o) in sorted(ov):
            s0, s1 = max(a, st), min(b, en)
            if s1 - s0 < min_piece:
                continue
            if bounds and bounds[-1][2] == sp and abs(bounds[-1][1] - s0) < 0.05:
                bounds[-1][1] = s1
            else:
                bounds.append([s0, s1, sp])
        if len(bounds) < 2:
            # 切不出来，归给占比最大的
            best = max(ov, key=lambda x: x[3])
            out.append(seg)
            labels.append(int(best[2]))
            continue
        if len(bounds) > max_pieces:
            bounds = bounds[:max_pieces - 1] + [[bounds[max_pieces - 1][0], b, bounds[max_pieces - 1][2]]]

        ratios = [(x[1] - a) / max(b - a, 1e-6) for x in bounds]
        texts = _split_text(seg.get("raw_text") or "", ratios)
        base = {k: v for k, v in seg.items() if k not in ("raw_text", "clean_text", "final_text")}
        for (s0, s1, sp), txt in zip(bounds, texts):
            piece = {**base, "start_ms": int(s0 * 1000), "end_ms": int(s1 * 1000),
                     "raw_text": txt, "spk_purity": 1.0}
            out.append(piece)
            labels.append(int(sp))

    log.info("帧级分离：%d 段 → %d 段", len(segs), len(out))
    return out, np.array(labels, dtype=int)


def probe_speaker_count(audio: np.ndarray, sr: int = 16000, seconds: float = 90.0,
                        min_turn: float = 1.0) -> int:
    """在音频开头若干秒上快速跑一次分离，估计有几个人（用于自动选引擎）。

    比"语音占比"更能区分会议与对话：会议≥3 人，一对一对话=2 人，独白=1 人。
    """
    try:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        head = audio[: int(seconds * sr)]
        if len(head) < sr * 20:
            return 0
        turns = run(head, sr, num_speakers=0)
        spks = {sp for st, en, sp in turns if en - st >= min_turn}
        return len(spks)
    except Exception as e:
        log.warning("说话人数探测失败: %s", e)
        return 0
