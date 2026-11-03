"""处理流水线：归一化 → 分窗 → 转写 → 说话人 → 清理 → 热词 → 纠错 → 归档。

支持断点续跑：每个阶段完成写入 work/<id>/state.json，ASR 逐窗缓存到 asr/wN.json。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from . import diarize as dia
from .settings import ROOT
from . import obsidian as ob
from . import textproc as tp
from .engines import AsrEngine, FFmpeg, LlmEngine, SpeakerEngine

log = logging.getLogger("tingji.pipeline")

STAGES = ["ingested", "normalized", "planned", "transcribed", "diarized",
          "identified", "emotion", "cleaned", "hotworded", "corrected", "refined", "roles", "summarized", "archived"]
PROGRESS = {"ingested": 2, "normalized": 8, "planned": 10, "transcribed": 42,
            "diarized": 52, "identified": 56, "emotion": 60, "cleaned": 65, "hotworded": 68,
            "corrected": 80, "refined": 86, "roles": 90, "summarized": 95, "archived": 100}


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def hms(ms: int) -> str:
    s = int(ms // 1000)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------- 分窗
def plan_windows(wav: Path, duration: float, target: float) -> list[dict]:
    if duration <= target * 1.15:
        return [{"idx": 0, "start": 0.0, "end": duration}]
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(wav), "-af",
             "silencedetect=noise=-32dB:d=0.25", "-f", "null", "-"],
            capture_output=True, text=True,
        ).stderr
        sil = [(float(a), float(a) + float(b)) for a, b in
               re.findall(r"silence_start: ([\d.]+).*?silence_end: ([\d.]+)", out, re.S)]
    except Exception:
        sil = []
    cuts, k = [], 1
    while k * target < duration - 10:
        t = k * target
        best, bestd = t, 1e9
        for a, b in sil:
            mid = (a + b) / 2
            if abs(mid - t) < bestd and abs(mid - t) <= 120:
                best, bestd = mid, abs(mid - t)
        cuts.append(best)
        k += 1
    windows, prev = [], 0.0
    for i, c in enumerate(cuts + [duration]):
        windows.append({"idx": i, "start": prev, "end": c})
        prev = c
    return windows


def _load_diar_cache(dpath: Path, n: int, vp_sig: str | None = None) -> dict | None:
    """读取说话人缓存；文件损坏或段数不符时作废（返回 None）。

    段数不符是真实踩过的坑：重跑后 ASR 段数变了，缓存却是旧的，旧代码会
    "静默跳过"整个分离阶段，导致 spk 缺失、后面归档直接 KeyError。
    """
    if not dpath.exists():
        return None
    try:
        c = json.loads(dpath.read_text())
    except Exception:
        log.info("说话人缓存损坏，已作废")
        dpath.unlink(missing_ok=True)
        return None
    if len(c.get("labels", [])) != n:
        log.info("说话人缓存段数不符（%d≠%d），已作废并重算", len(c.get("labels", [])), n)
        dpath.unlink(missing_ok=True)
        return None
    if vp_sig and c.get("vp_sig") and c["vp_sig"] != vp_sig:
        # 声纹库变了（新登记/删除）→ 必须重新认名，否则新声纹永远不生效
        log.info("声纹库已变化，作废认名缓存")
        dpath.unlink(missing_ok=True)
        return None
    return c


def _use_sherpa_diar(cfg) -> bool:
    """是否使用帧级（pyannote 分割）分离。"""
    backend = getattr(cfg.diarization, "backend", "sherpa")
    if backend not in ("sherpa", "auto"):
        return False
    try:
        from . import diar_sherpa
        return diar_sherpa.available(ROOT)
    except Exception:
        return False


# ---------------------------------------------------------------- 并发锁
class JobLock:
    """同一 job 目录不允许两个进程同时写（压测时踩过并发写坏 w0.wav）。"""

    def __init__(self, wdir: Path):
        self.path = wdir / ".lock"
        self.fh = None

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            raise RuntimeError("该任务正在被另一个进程处理（job 目录已加锁）")
        return self

    def __exit__(self, *exc):
        try:
            import fcntl
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        finally:
            self.fh.close()


# ---------------------------------------------------------------- 状态
class State:
    def __init__(self, wdir: Path):
        self.path = wdir / "state.json"
        wdir.mkdir(parents=True, exist_ok=True)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"stages": []}

    def done(self, name: str) -> bool:
        return name in self.data.get("stages", [])

    def mark(self, name: str, **kw) -> None:
        st = self.data.setdefault("stages", [])
        if name not in st:
            st.append(name)
        self.data.update(kw)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1))

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- 主流程
def run_job(db, cfg, job_id: str) -> None:
    job = db.get_job(job_id)
    src = Path(job["source_path"])
    wdir = cfg.work_path / job_id
    wdir.mkdir(parents=True, exist_ok=True)
    outdir = cfg.output_path / Path(job["source_name"]).stem
    outdir.mkdir(parents=True, exist_ok=True)

    # 资源守卫：磁盘/内存不足就别开始，否则跑到一半失败更难查
    try:
        from . import resource
        ok, why = resource.check(cfg, float(job.get("duration_sec") or 0))
        if not ok:
            db.update_job(job_id, status="failed", message=f"资源不足，未开始：{why}")
            db.log(job_id, "error", "resource", why)
            return
    except Exception as e:
        log.warning("资源检查失败（忽略）: %s", e)

    with JobLock(wdir):
        _run_job_locked(db, cfg, job_id, job, src, wdir, outdir)


def _run_job_locked(db, cfg, job_id, job, src, wdir, outdir) -> None:
    state = State(wdir)
    outdir = cfg.output_path / Path(job["source_name"]).stem
    outdir.mkdir(parents=True, exist_ok=True)
    if state.data.get("stages"):
        db.log(job_id, "info", "resume", f"断点续跑，已完成阶段：{state.data['stages']}")

    def stage(name: str, msg: str = "", skip: bool = False) -> None:
        db.update_job(job_id, stage=name, progress=PROGRESS.get(name, 0) or 0, message=msg or name)
        if not skip:
            state.mark(name)
        db.log(job_id, "info", name, msg or name)

    t0 = time.time()
    info = FFmpeg.probe(src)
    duration = info["duration"]

    # 1) 归一化
    wav = wdir / "16k.wav"
    if state.done("normalized") and wav.exists():
        stage("normalized", "已归一化（跳过）", skip=True)
    else:
        stage("normalized", "解码为 16k 单声道")
        FFmpeg.normalize(src, wav, cfg.audio.target_sr, cfg.audio.loudness_normalize)
        if cfg.audio.preview_m4a:
            FFmpeg.to_m4a(wav, wdir / "preview.m4a")

    # 2) 分窗
    stage("planned", "切分处理窗口")
    windows = plan_windows(wav, duration, cfg.asr.window_speech_sec)
    db.log(job_id, "info", "planned", f"{len(windows)} 个窗口", windows)

    # 3) 转写（逐窗缓存）
    hot_rows = db.list_hotwords(only_enabled=True) if cfg.hotwords.enabled else []
    hot_l1 = [r["term"] for r in sorted(hot_rows, key=lambda r: -r["weight"])][: cfg.hotwords.l1_max]
    asrdir = wdir / "asr"

    # 静音门控：整段几乎没有有效语音时直接跳过，避免白跑几十分钟
    _ratio = 1.0
    try:
        from . import acoustics as _ac
        _a, _sr = sf.read(str(wav), dtype="float32")
        if _a.ndim > 1:
            _a = _a.mean(axis=1)
        _ratio = _ac.speech_ratio(_a, int(_sr))
        db.log(job_id, "info", "transcribed", f"有效语音占比 {_ratio:.1%}")
        if duration > 5 and _ratio < float(getattr(cfg.asr, "min_speech_ratio", 0.02)):
            db.update_job(job_id, status="done", progress=100,
                          message=f"未检测到有效语音（{_ratio:.1%}），已跳过识别")
            db.log(job_id, "info", "transcribed", "静音门控：跳过识别")
            return
    except Exception as _e:
        log.warning("静音门控失败（忽略）: %s", _e)

    # 自动选引擎：按识别前就能拿到的特征分流（会议 vs 叙事）
    provider = getattr(cfg.asr, "provider", "funasr")
    if provider == "auto":
        try:
            from .asr_select import choose
            from .nano import FunasrNanoEngine
            nano_ok = FunasrNanoEngine(".").available()
            n_spk = 0
            try:
                from . import diar_sherpa as _ds
                _audio, _sr0 = sf.read(str(wav), dtype="float32")
                n_spk = _ds.probe_speaker_count(_audio, int(_sr0))
                db.log(job_id, "info", "transcribed", f"说话人数探测：{n_spk} 人")
            except Exception as _pe:
                log.warning("人数探测失败（忽略）: %s", _pe)
            provider, why = choose(duration, _ratio, nano_ok, n_spk)
            db.log(job_id, "info", "transcribed", f"自动选引擎 → {provider}（{why}）")
        except Exception as e:
            provider = "funasr"
            db.log(job_id, "warning", "transcribed", f"自动选引擎失败，用 funasr：{e}")

    asrdir.mkdir(exist_ok=True)
    segs: list[dict] = []
    asr = None
    for w in windows:
        cache = asrdir / f"w{w['idx']}.json"
        if cache.exists():
            got = json.loads(cache.read_text())
            db.log(job_id, "info", "transcribed", f"窗口 {w['idx']} 命中缓存 +{len(got)} 句")
        else:
            if asr is None:
                asr = AsrEngine(cfg.asr, hot_l1=hot_l1, provider=provider)
            ww = wdir / f"w{w['idx']}.wav"
            if not ww.exists():
                FFmpeg.cut(wav, w["start"], w["end"] - w["start"], ww)
            got = asr.transcribe(ww, offset_ms=int(w["start"] * 1000))
            cache.write_text(json.dumps(got, ensure_ascii=False))
        segs.extend(got)
        prog = 10 + int(40 * (w["idx"] + 1) / len(windows))
        db.update_job(job_id, progress=prog, message=f"转写 {w['idx'] + 1}/{len(windows)}")
    segs.sort(key=lambda s: s["start_ms"])
    # ASR 幻觉过滤：噪音/音乐上模型会编出固定套话或整段重复
    try:
        from . import hallucination as hal
        kept, dropped = [], []
        for s in segs:
            bad, why = hal.is_hallucination(s.get("raw_text") or "")
            if bad:
                dropped.append({"start_ms": s["start_ms"], "text": (s.get("raw_text") or "")[:60], "why": why})
            else:
                kept.append(s)
        if dropped:
            db.log(job_id, "info", "cleaned",
                   f"滤除疑似 ASR 幻觉 {len(dropped)} 段", dropped[:20])
            segs = kept
    except Exception as e:
        log.warning("幻觉过滤失败（忽略）: %s", e)
    for i, s in enumerate(segs):
        s["idx"] = i
        s["quality"], s["char_rate"] = tp.quality_flag(s["raw_text"], (s["end_ms"] - s["start_ms"]) / 1000)
    db.save_segments(job_id, segs)
    stage("transcribed", f"{len(segs)} 句")

    # 3.5) 情绪与声学事件
    has_emotion = any(s.get("emotion") for s in segs)
    if has_emotion:
        stage("emotion", "ASR 已内置情绪标签（SenseVoice）", skip=True)
        db.save_segments(job_id, segs)
    elif getattr(cfg, "emotion", None) and cfg.emotion.enabled and len(segs):
        try:
            from . import emotion as emo_mod
            stage("emotion", "识别情绪与声学事件")
            eng = emo_mod.EmotionEngine(cfg.emotion.device)
            utts = eng.detect(str(wav))
            emo_mod.attach(segs, utts)
            db.save_segments(job_id, segs)
            mood = emo_mod.summarize_mood(segs)
            db.log(job_id, "info", "emotion", f"{len(utts)} 段情绪；基调：{mood or '中性'}")
        except Exception as e:
            log.warning("情绪识别失败（忽略）: %s", e)
            db.log(job_id, "warning", "emotion", f"情绪识别失败：{e}")

    # 4) 说话人
    labels = np.full(len(segs), -1, dtype=int)
    clusters: dict[int, float] = {}
    names: dict[int, str] = {}
    cents: dict[int, np.ndarray] = {}
    dpath = wdir / "diarize.json"
    for _s0 in segs:
        _s0.setdefault("spk", -1)      # 任何路径都不能让 spk 缺失
    cache = _load_diar_cache(dpath, len(segs), db.voiceprint_signature())
    if cache is not None and cfg.diarization.enabled and len(segs) > 1:
        for s_, l in zip(segs, cache["labels"]):
            s_["spk"] = int(l)
        for s_, o in zip(segs, cache.get("overlap", [])):
            s_["overlap"] = int(o)
        clusters = {int(k): v for k, v in cache.get("clusters", {}).items()}
        names = {int(k): v for k, v in cache.get("names", {}).items()}
        for s_ in segs:
            if s_.get("spk") in names:
                s_["speaker_name"] = names[s_.get("spk")]
        stage("diarized", "命中说话人缓存", skip=True)
        stage("identified", "命中认名缓存", skip=True)
        db.save_segments(job_id, segs)
    elif cfg.diarization.enabled and len(segs) > 1 and _use_sherpa_diar(cfg):
        stage("diarized", "帧级说话人分离（pyannote 分割）")
        try:
            from . import diar_sherpa
            audio_all, _sr = sf.read(str(wav), dtype="float32")
            if audio_all.ndim > 1:
                audio_all = audio_all.mean(axis=1)
            speech_iv = [(x["start_ms"] / 1000, x["end_ms"] / 1000) for x in segs]
            if getattr(cfg.diarization, "mask_non_speech", True):
                masked = diar_sherpa.mask_non_speech(audio_all, speech_iv, int(_sr))
                db.log(job_id, "info", "diarized",
                       f"非语音掩蔽：{len(speech_iv)} 段语音，其余置零")
            else:
                masked = audio_all
            turns = diar_sherpa.run(masked, int(_sr),
                                     num_speakers=cfg.diarization.max_speakers)
            n_before = len(segs)
            segs, labels = diar_sherpa.apply(segs, turns)
            # 切开后重算时长/质量/声学特征
            from . import textproc as _tp
            for x in segs:
                dur = (x["end_ms"] - x["start_ms"]) / 1000
                x["quality"], x["char_rate"] = _tp.quality_flag(x.get("raw_text") or "", dur)
                a = int(x["start_ms"] / 1000 * _sr)
                b = int(x["end_ms"] / 1000 * _sr)
                clip = audio_all[a:b]
                try:
                    from . import acoustics as ac
                    x.update(ac.features(clip, _sr))
                except Exception:
                    pass
            # 为切开后的段重算嵌入 → 质心（认名依赖它）
            spk_eng = SpeakerEngine(cfg.diarization.campplus_model)
            embs = []
            for x in segs:
                a = int(x["start_ms"] / 1000 * _sr)
                b = int(x["end_ms"] / 1000 * _sr)
                clip = audio_all[a:b]
                embs.append(spk_eng.embed(clip) if len(clip) > int(0.3 * _sr) else None)
            E = np.stack([e if e is not None else np.zeros(192, dtype=np.float32) for e in embs])

            # 自动聚类容易裂成一堆小簇：按质心相似度合并到上限
            cap = int(cfg.diarization.max_speakers or getattr(cfg.diarization, "max_auto_speakers", 8))
            n_before_merge = len(set(int(v) for v in labels if v >= 0))
            labels, n_merged = dia.merge_until(E, labels, cap)
            if n_merged:
                db.log(job_id, "info", "diarized",
                       f"簇数 {n_before_merge} → {cap}（合并 {n_merged} 次）")
            for s_, l in zip(segs, labels):
                s_["spk"] = int(l)
            try:
                cents = diar_sherpa.centroids_from_turns(
                    audio_all, turns, spk_eng.embed, int(_sr),
                    float(getattr(cfg.diarization, "embed_window_sec", 15.0)))
                if not cents:
                    cents = dia.centroids_clean(E, labels, segs)
                db.log(job_id, "info", "diarized",
                       f"声纹质心：{len(cents)} 人（按 turn 15s 窗口）")
            except Exception as _e:
                cents = dia.centroids_clean(E, labels, segs)
            clusters = {}
            for s_ in segs:
                clusters[s_["spk"]] = clusters.get(s_["spk"], 0) + (s_["end_ms"] - s_["start_ms"]) / 1000
            db.log(job_id, "info", "diarized",
                   f"帧级分离：{len(turns)} 个 turn，{n_before} 段 → {len(segs)} 段，{len(clusters)} 人")
        except Exception as e:
            log.warning("帧级分离失败: %s", e)
            db.log(job_id, "warning", "diarized", f"帧级分离失败，说话人未知：{e}")
            clusters = {}
    elif cfg.diarization.enabled and len(segs) > 1:
        stage("diarized", "提取声纹并聚类")
        spk = SpeakerEngine(cfg.diarization.campplus_model)
        embs, durs = [], []
        with sf.SoundFile(str(wav)) as f:
            sr = f.samplerate
            for s in segs:
                a, b = int(s["start_ms"] / 1000 * sr), int(s["end_ms"] / 1000 * sr)
                f.seek(a)
                clip = f.read(max(0, b - a), dtype="float32", always_2d=False)
                if clip.ndim > 1:
                    clip = clip.mean(axis=1)
                embs.append(spk.embed(clip) if len(clip) > int(0.3 * sr) else None)
                durs.append((s["end_ms"] - s["start_ms"]) / 1000)
                try:
                    from . import acoustics as ac
                    s.update(ac.features(clip, sr))
                except Exception:
                    pass
        D = np.array(durs)
        E = np.stack([e if e is not None else np.zeros(192, dtype=np.float32) for e in embs])
        forced = bool(cfg.diarization.max_speakers)
        labels = dia.cluster(E, D, cfg.diarization.cluster_threshold,
                             n_clusters=(cfg.diarization.max_speakers or None))
        if not forced:
            # 用户显式指定人数时，不再合簇/丢小簇，否则会被"优化"成 1 人
            cents = dia.centroids(E, labels)
            labels = dia.merge_by_centroid(labels, cents, cfg.diarization.merge_threshold)
            cents = dia.centroids(E, labels)
            labels = dia.reassign(E, labels, cents)
        try:
            from . import acoustics as ac
            before = len(set(labels[labels >= 0].tolist()))
            labels = ac.split_by_f0(segs, labels)
            labels = ac.cap_clusters(segs, labels, cfg.diarization.max_speakers)
            after = len(set(labels[labels >= 0].tolist()))
            if after != before:
                db.log(job_id, "info", "diarized", f"按基频拆分 {before} → {after} 人")
        except Exception as e:
            db.log(job_id, "warning", "diarized", f"基频拆分失败：{e}")
        labels = dia.fill_short(segs, labels, cfg.diarization.min_cluster_segment_sec)
        if not forced:
            labels = dia.drop_small_clusters(E, labels, D)

        # 标签平滑：抑制 A-B-A 抖动；短 run 合并
        try:
            from . import segfix
            before = len(set(labels[labels >= 0].tolist()))
            labels = segfix.smooth_labels(labels, D,
                                                       min_run=int(getattr(cfg.diarization, "smooth_min_run", 1)))
            db.log(job_id, "info", "diarized", f"标签平滑：{before} 人，分布 {segfix.stats(labels)}")
        except Exception as e:
            db.log(job_id, "warning", "diarized", f"标签平滑失败：{e}")
        cents = dia.centroids_clean(E, labels, segs)

        # 跨说话人重切：一句里前后半段指向不同人 → 按停顿切开
        if getattr(cfg.diarization, "resplit", True) and len(cents) >= 2:
            try:
                from . import segfix
                audio_all, _sr = sf.read(str(wav), dtype="float32")
                if audio_all.ndim > 1:
                    audio_all = audio_all.mean(axis=1)
                new_segs, new_labels = segfix.resplit_by_speaker(
                    segs, labels, audio_all, spk.embed, cents,
                    max_split=int(getattr(cfg.diarization, "resplit_max", 40)))
                if len(new_segs) != len(segs):
                    segs = new_segs
                    labels = new_labels
                    # 重切后重算时长/质量，并重建嵌入数组（供后续统计）
                    durs = [(x["end_ms"] - x["start_ms"]) / 1000 for x in segs]
                    D = np.array(durs)
                    for x in segs:
                        x["quality"], x["char_rate"] = tp.quality_flag(
                            x.get("raw_text") or "", (x["end_ms"] - x["start_ms"]) / 1000)
                    db.log(job_id, "info", "diarized", f"重切后 {len(segs)} 段")
            except Exception as e:
                db.log(job_id, "warning", "diarized", f"重切失败：{e}")

        for s, l in zip(segs, labels):
            s["spk"] = int(l)
        for k in sorted(set(labels.tolist())):
            if k >= 0:
                clusters[k] = round(float(sum(d for d, l in zip(durs, labels) if l == k)), 1)
        if cfg.diarization.overlap_detect and len(segs) <= 3000:
            audio, _ = sf.read(str(wav), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            sc = dia.overlap_scores(audio, segs, 16000, spk.embed)
            ov = dia.adaptive_overlap_flags(sc, D, cfg.diarization.overlap_floor)
            for s, o in zip(segs, ov):
                s["overlap"] = int(o)
        try:
            from . import acoustics as ac
            n_nav = n_noise = 0
            rms_vals = sorted([s_.get("rms") or 0 for s_ in segs if (s_.get("rms") or 0) > 0])
            rms_med = rms_vals[len(rms_vals) // 2] if rms_vals else 0.0
            for s_ in segs:
                if ac.looks_like_nav(s_.get("raw_text") or ""):
                    s_["speaker_kind"] = "nav"; n_nav += 1
                elif ac.looks_like_noise(s_, rms_med):
                    s_["speaker_kind"] = "noise"; n_noise += 1
                else:
                    s_["speaker_kind"] = "human"
            if n_nav or n_noise:
                db.log(job_id, "info", "diarized", f"非人声：导航/播报 {n_nav} 段，噪声 {n_noise} 段")
        except Exception:
            pass
        stage("diarized", f"{len(clusters)} 位说话人")

    # 5) 声纹认名（帧级/聚类两条路径共用）
    if cfg.diarization.enabled and cents:
        stage("identified", "匹配声纹库")
        names, matches = match_voiceprints(
            db, cents,
            threshold=float(getattr(cfg.diarization, "match_threshold", 0.35)),
            margin=float(getattr(cfg.diarization, "match_margin", 0.10)))
        for s in segs:
            if s.get("spk") in names:
                s["speaker_name"] = names[s["spk"]]
        dpath.write_text(json.dumps({
            "labels": [int(x) for x in labels.tolist()],
            "overlap": [int(s_.get("overlap", 0)) for s_ in segs],
            "clusters": {str(k): v for k, v in clusters.items()},
            "names": {str(k): v for k, v in names.items()},
            "vp_sig": db.voiceprint_signature(),
        }, ensure_ascii=False))
        (wdir / "speakers.json").write_text(json.dumps(
            {"clusters": {str(k): v for k, v in clusters.items()},
             "names": {str(k): v for k, v in names.items()},
             "matches": matches}, ensure_ascii=False, indent=1))
        # 未认出的说话人按首次出现顺序重新编号（簇 id 可能是 1/2/3，直接 +1 会从"说话人2"开始）
        known = {k for k in names}
        order: list[int] = []
        for s_ in segs:
            k = s_.get("spk")
            if isinstance(k, int) and k >= 0 and k not in known and k not in order:
                order.append(k)
        for i, k in enumerate(order, 1):
            names[k] = f"说话人{i}"
        for s_ in segs:
            if s_.get("spk") in names:
                s_["speaker_name"] = names[s_["spk"]]
        db.log(job_id, "info", "identified", f"认名 {len(names)} 人", matches)
    db.save_segments(job_id, segs)

    # 6) 规则清理
    stage("cleaned", "去除口语填充词")
    from . import itn as itn_mod
    for s in segs:
        s["clean_text"], _ = tp.clean_text(s["raw_text"], cfg.text.filler_removal, cfg.text.repeat_removal)
        if s.get("boundary") in ("pause", "cut"):
            s["clean_text"], _trimmed = tp.trim_uncertain_tail(s["clean_text"], s["boundary"])
        if cfg.text.itn:
            s["clean_text"] = itn_mod.normalize(s["clean_text"])
        s["final_text"] = s["clean_text"]

    # 7) 热词 L2
    stage("hotworded", "应用热词")
    matcher = None
    if cfg.hotwords.enabled and hot_rows:
        matcher = tp.HotwordMatcher(hot_rows, cfg.hotwords.l2_threshold)
        hits = 0
        for s in segs:
            new, ms = matcher.apply(s["final_text"] or "")
            if ms:
                s["final_text"] = new
                hits += len(ms)
        db.log(job_id, "info", "hotworded", f"热词修正 {hits} 处")

    # 7.5) 正则纠错规则（单位/邮箱/固定错法）
    if getattr(cfg.text, "rules_enabled", True):
        try:
            from .rules import RuleCorrector
            rc = RuleCorrector(getattr(cfg.text, "rules_file", "config/rules.txt"))
            if rc.rules:
                total = 0
                for s in segs:
                    new, n = rc.apply(s["final_text"] or "")
                    if n:
                        s["final_text"] = new
                        total += n
                if total:
                    db.log(job_id, "info", "hotworded", f"规则替换 {total} 处", rc.errors[:5])
        except Exception as e:
            log.warning("规则纠错失败（忽略）: %s", e)
    db.save_segments(job_id, segs)

    # 8) 可选：LLM 纠错
    degraded = 0
    if cfg.correction.enabled and len(segs):
        try:
            stage("corrected", f"LLM 纠错（{cfg.correction.mode}）")
            n = correct_all(db, cfg, job_id, segs, hot_rows, matcher)
            db.log(job_id, "info", "corrected", f"处理 {n} 批")
        except Exception as e:
            log.warning("纠错失败，降级: %s", e)
            db.log(job_id, "warning", "corrected", f"纠错失败，降级：{e}")
            degraded = 1
    db.save_segments(job_id, segs)

    # 9) 段落化润色（口语→书面，保留段落级时间）
    paragraphs: list[dict] = []
    if getattr(cfg, "refine", None) and cfg.refine.enabled and len(segs):
        rpath = wdir / "paragraphs.json"
        if rpath.exists() and state.done("refined"):
            paragraphs = json.loads(rpath.read_text())
            stage("refined", "命中段落缓存", skip=True)
        else:
            try:
                from . import refine as rf
                stage("refined", "段落化润色")
                paragraphs = rf.refine(db, cfg, job_id, segs, matcher, hot_rows,
                                       max_blocks=cfg.refine.max_blocks)
                rpath.write_text(json.dumps(paragraphs, ensure_ascii=False, indent=1))
            except Exception as e:
                log.warning("段落润色失败: %s", e)
                db.log(job_id, "warning", "refined", f"段落润色失败：{e}")
    if paragraphs:
        db.save_paragraphs(job_id, paragraphs)
        stage("refined", f"{len(paragraphs)} 段")

    # 9.5) 说话人角色消歧（仅在有依据时命名）
    if paragraphs and getattr(cfg, "roles", None) and cfg.roles.enabled:
        try:
            from . import roles as rl
            stage("roles", "推断说话人身份")
            cand = rl.detect(db, cfg, job_id, paragraphs, names)
            if cand:
                names = rl.display_names(cand, names)
                for s_ in segs:
                    if s_.get("spk") in names:
                        s_["speaker_name"] = names[s_.get("spk")]
                rj = wdir / "roles.json"
                rj.write_text(json.dumps({str(k): v for k, v in cand.items()}, ensure_ascii=False, indent=1))
                db.save_segments(job_id, segs)
        except Exception as e:
            log.warning("角色消歧失败: %s", e)
            db.log(job_id, "warning", "roles", f"角色消歧失败：{e}")

    # 10) 智能纪要
    summary = None
    if paragraphs and getattr(cfg, "summary", None) and cfg.summary.enabled:
        try:
            from . import summarize as sm
            stage("summarized", "生成摘要与章节")
            summary = sm.summarize(db, cfg, job_id, paragraphs, names)
            if summary:
                db.save_summary(job_id, summary)
        except Exception as e:
            log.warning("纪要生成失败: %s", e)
            db.log(job_id, "warning", "summarized", f"纪要失败：{e}")

    # 11) 归档
    stage("archived", "写出逐字稿")
    write_outputs(cfg, job, segs, duration, outdir, wdir, names, paragraphs, summary)

    if cfg.obsidian.vault_path:
        try:
            exp = ob.write_vault(cfg, job, segs, duration, names, wdir, src, paragraphs, summary)
            db.save_export(job_id, exp)
            db.log(job_id, "info", "obsidian", f"已写入 {exp.get('note')}")
            if not cfg.obsidian.keep_source and src.exists():
                src.unlink()
                db.log(job_id, "info", "cleanup", "已删除源文件")
        except Exception as e:
            log.warning("Obsidian 归档失败: %s", e)
            db.log(job_id, "warning", "obsidian", f"归档失败：{e}")
            degraded = 1

    db.update_job(job_id, status="done", progress=100, finished_at=now_str(),
                  degraded=degraded, message=f"完成，用时 {time.time() - t0:.0f}s")
    try:
        from . import notify
        notify.send(cfg, dict(job), "done", f"用时 {time.time() - t0:.0f}s",
                    note=str((db.get_export(job_id) or {}).get("note") or ""))
    except Exception:
        pass
    db.log(job_id, "info", "archived", f"输出 {outdir}", {"elapsed": round(time.time() - t0, 1)})


def now_str() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ---------------------------------------------------------------- 声纹认名
def cluster_channels(vectors: list[np.ndarray], threshold: float = 0.75,
                     max_channels: int = 4) -> tuple[list[np.ndarray], list[int]]:
    """把同一人的样本聚成若干信道簇，返回 (每簇质心, 每条样本的簇号)。"""
    clusters: list[list] = []
    ids: list[int] = []
    for v in vectors:
        v = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(v)
        if n == 0:
            ids.append(-1)
            continue
        v = v / n
        if clusters:
            sims = [float(v @ c) for c, _ in clusters]
            i = int(np.argmax(sims))
            if sims[i] >= threshold or len(clusters) >= max_channels:
                c, cnt = clusters[i]
                merged = (c * cnt + v) / (cnt + 1)
                clusters[i] = [merged / (np.linalg.norm(merged) + 1e-9), cnt + 1]
                ids.append(i)
                continue
        clusters.append([v.copy(), 1])
        ids.append(len(clusters) - 1)
    return [c for c, _ in clusters], ids


def group_channels(vectors: list[np.ndarray], threshold: float = 0.75,
                   max_channels: int = 4) -> list[np.ndarray]:
    """把同一人的样本按"信道"聚成若干簇，返回每簇质心。

    为什么需要：录音设备不同（网页麦克风 / 车机 / 手机）时，同一人的声纹差异
    可能比不同人还大（实测同设备 0.90，跨设备 0.30）。把所有样本平均成一个
    质心会把两种信道都"拉糊"；按信道分开、匹配时取最高分，跨设备更稳。
    """
    return cluster_channels(vectors, threshold, max_channels)[0]


def match_voiceprints(db, cents: dict[int, np.ndarray], threshold: float = 0.72,
                      margin: float = 0.05, channel_threshold: float = 0.75,
                      max_channels: int = 4) -> tuple[dict[int, str], dict]:
    vps = db.all_voiceprints()
    if not vps or not cents:
        return {}, {}
    # 按人分组 → 每组内做信道聚类 → 得到"每人若干个信道质心"
    per_person: dict[str, list[np.ndarray]] = {}
    for v in vps:
        arr = np.frombuffer(v["vector"], dtype=np.float32)
        n = np.linalg.norm(arr)
        if n == 0:
            continue
        per_person.setdefault(v["name"], []).append(arr / n)
    M, owners = [], []
    for nm, vecs in per_person.items():
        for c in group_channels(vecs, channel_threshold, max_channels):
            M.append(c)
            owners.append(nm)
    if not M:
        return {}, {}
    M = np.stack(M)
    names, detail = {}, {}
    for k, c in cents.items():
        c = np.asarray(c, dtype=np.float32)
        cn = np.linalg.norm(c)
        if cn == 0:
            continue
        sims = M @ (c / cn)          # 必须归一化，否则模长会被当成相似度
        # 同一人可能有多条样本：按人取最大相似度，避免"自己比自己"吃掉领先幅度
        per: dict[str, float] = {}
        for nm, sc in zip(owners, sims.tolist()):
            if sc > per.get(nm, -1.0):
                per[nm] = sc
        ranked = sorted(per.items(), key=lambda kv: -kv[1])
        top1_name, top1 = ranked[0]
        top2 = ranked[1][1] if len(ranked) > 1 else 0.0
        detail[str(k)] = {"name": top1_name, "score": round(top1, 3),
                          "margin": round(top1 - top2, 3),
                          "samples": len(per_person.get(top1_name, [])),
                          "channels": sum(1 for x in owners if x == top1_name),
                          "confidence": ("high" if top1 >= 0.5 else "low")}
        if top1 >= threshold and (top1 - top2) >= margin:
            names[k] = top1_name
    return names, detail


# ---------------------------------------------------------------- 纠错
META_START = ("（注", "(注", "由于您提供", "抱歉", "作为AI", "我无法", "很抱歉")
META_HARD = ("请补充完整", "无法进行校对", "暂无可输出", "我将立即为您", "无法提取有效")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
MEANING = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

SYS = "你是中文会议转写稿的校对员。输入是语音识别原始输出，标点混乱、有口语填充词、可能有同音字与英文术语错误。"

RULES_BLOCK = """要求：
1. 去掉口语填充词与无意义重复，但不删减实质信息；
2. 重新断句并使用正确标点；
3. 修正明显的同音错字与英文术语（sars→SaaS、口顶→Copilot、Infinity Band→InfiniBand），不确定的不要改；
4. 输出必须是**一整段连续文字**：禁止编号列表、小标题、分点、总结；
5. 不增加原文没有的事实、数字或人名；**不要输出任何时间戳**；
6. 只输出整理后的正文。"""

RULES_SEG = """要求：
1. 逐句整理，**输入几句就输出几句**，不得合并、拆分、增删；
2. 每句去掉口语填充词、重写标点、修正明显的同音错字与英文术语；
3. 不增加原文没有的事实、数字、人名；不要输出时间戳；
4. 只输出 JSON 数组，元素形如 {"i": 原序号, "text": "整理后文本"}，不要任何解释或代码块标记。"""


def _need_llm(text: str, segs: list[dict], hot_terms: set[str]) -> bool:
    if re.search(r"[A-Za-z]{2,}|\d", text):
        return True
    if any((s.get("char_rate") or 9) < 3 for s in segs):
        return True
    return any(t in text for t in hot_terms)


def _sanitize(fixed: str, raw: str) -> str:
    t = (fixed or "").strip()
    if not t or t.startswith(META_START) or any(h in t for h in META_HARD):
        return raw
    t, _ = tp.strip_repeats(t)          # 模型偶尔会引入"掉掉"这类重复
    t = re.sub(r"[（(][^（）()]*(原文|校对|补充|无法|转写)[^（）()]*[）)]\s*$", "", t).strip()
    return t if len(MEANING.findall(t)) >= 2 else raw


def _parse_json_array(s: str) -> list | None:
    s = re.sub(r"^```(?:json)?|```$", "", (s or "").strip(), flags=re.M).strip()
    a, b = s.find("["), s.rfind("]")
    if a < 0 or b < 0:
        return None
    try:
        v = json.loads(s[a:b + 1])
        return v if isinstance(v, list) else None
    except Exception:
        return None


def correct_all(db, cfg, job_id: str, segs: list[dict], hot_rows, matcher) -> int:
    llm = LlmEngine(cfg.llm, cfg.correction.model)
    hot_terms = {r["term"] for r in hot_rows}
    calls = 0
    if cfg.correction.mode == "segment":
        batch, size = [], 0
        for s in segs:
            batch.append(s)
            size += len(s["raw_text"])
            if len(batch) >= 12 or size >= 400:
                calls += _correct_segments(llm, cfg, batch, hot_terms, matcher)
                batch, size = [], 0
        if batch:
            calls += _correct_segments(llm, cfg, batch, hot_terms, matcher)
    else:
        for b in tp.blocks_of(segs):
            buf = []
            for s in b["items"]:
                buf.append(s)
                if sum(len(x["raw_text"]) for x in buf) >= cfg.correction.batch_chars:
                    calls += _correct_block(llm, cfg, buf, hot_terms, matcher)
                    buf = []
            if buf:
                calls += _correct_block(llm, cfg, buf, hot_terms, matcher)
    return calls


def _correct_segments(llm, cfg, batch, hot_terms, matcher) -> int:
    raw_all = "\n".join(f"{i}. {s['raw_text']}" for i, s in enumerate(batch))
    if len(MEANING.findall(raw_all)) < 4:
        return 0
    if cfg.correction.selective and not _need_llm(raw_all, batch, hot_terms):
        return 0
    out = llm.chat(f"{RULES_SEG}\n\n原文：\n{raw_all}", system=SYS, max_tokens=1500)
    items = _parse_json_array(out)
    if not items:
        return 0
    idx_map = {}
    for x in items:
        if isinstance(x, dict):
            try:
                idx_map[int(x.get("i", -1))] = str(x.get("text", ""))
            except Exception:
                pass
    if set(idx_map) != set(range(len(batch))):
        return 0
    for i, s in enumerate(batch):
        raw = s["raw_text"]
        fixed = _sanitize(idx_map[i], raw)
        if TIME_RE.search(fixed):
            continue
        ratio = len(fixed) / max(len(raw), 1)
        if not (1 - cfg.correction.max_length_ratio <= ratio <= 1.15):
            continue
        if matcher:
            fixed, _ = matcher.apply(fixed)
        s["final_text"] = fixed
    return 1


def _correct_block(llm, cfg, batch, hot_terms, matcher) -> int:
    raw = "".join(s["raw_text"] for s in batch)
    if len(MEANING.findall(raw)) < 4:
        return 0
    if cfg.correction.selective and not _need_llm(raw, batch, hot_terms):
        return 0
    fixed = llm.chat(f"{RULES_BLOCK}\n\n原文：\n{raw}", system=SYS, max_tokens=1200)
    fixed = _sanitize(fixed, raw)
    if TIME_RE.search(fixed):
        fixed = raw
    ratio = len(fixed) / max(len(raw), 1)
    if not (1 - cfg.correction.max_length_ratio <= ratio <= 1 + cfg.correction.max_length_ratio):
        fixed = raw
    if matcher:
        fixed, _ = matcher.apply(fixed)
    total = sum(max(len(s["raw_text"]), 1) for s in batch)
    pos = 0
    for i, s in enumerate(batch):
        share = max(len(s["raw_text"]), 1) / total
        n = len(fixed) if i == len(batch) - 1 else max(1, int(round(len(fixed) * share)))
        s["final_text"] = fixed[pos:pos + n].strip() or s["clean_text"]
        pos += n
    return 1


# ---------------------------------------------------------------- 输出
def write_outputs(cfg, job, segs, duration, outdir: Path, wdir: Path,
                  names: dict[int, str] | None = None,
                  paragraphs: list[dict] | None = None,
                  summary: dict | None = None) -> None:
    names = names or {}
    blocks = tp.blocks_of(segs)
    per: dict[int, float] = {}
    for b in blocks:
        per[b["spk"]] = per.get(b["spk"], 0) + (b["end_ms"] - b["start_ms"]) / 1000
    renum = {k: i + 1 for i, k in enumerate(sorted(per, key=lambda k: -per[k]))}
    display = {k: (names.get(k) or f"说话人{renum[k]}") for k in per}

    rows = [{
        "speaker": display.get(s.get("spk"), "未知"),
        "spk": s.get("spk"),
        "start_ms": s["start_ms"], "end_ms": s["end_ms"],
        "text": s["final_text"] or s["clean_text"] or s["raw_text"],
        "raw_text": s["raw_text"], "clean_text": s.get("clean_text"),
        "quality": s["quality"], "overlap": s.get("overlap", 0),
    } for s in segs]
    (outdir / "transcript.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    if paragraphs:
        (outdir / "paragraphs.json").write_text(json.dumps(paragraphs, ensure_ascii=False, indent=1), encoding="utf-8")

    engine = getattr(cfg.asr, "provider", "funasr")
    engine_name = "SenseVoiceSmall" if engine == "sensevoice" else "FunASR Paraformer-zh + CT-Punc"
    md = [f"# {Path(job['source_name']).stem} · 逐字稿\n",
          f"- 时长 {hms(int(duration * 1000))}｜句子 {len(segs)}｜说话人 {len(per)}\n",
          f"- 引擎 {engine_name} + CAM++\n"]
    for k in sorted(per, key=lambda x: -per[x]):
        md.append(f"- {display[k]}：{per[k] / 60:.1f} 分钟")
    md.append("\n---\n")
    last = None
    for s in segs:
        if not (s.get("final_text") or s.get("clean_text") or "").strip():
            continue
        k = s.get("spk")
        if k != last:
            md.append(f"\n**{display.get(k, '未知')}**\n")
            last = k
        flag = " ⚠️重叠" if s.get("overlap") else (" ⚠️低质量" if s["quality"] == "low_quality" else "")
        emo = ""
        try:
            from .emotion import label_of
            emo = label_of(s)
        except Exception:
            pass
        md.append(f"`[{hms(s['start_ms'])}–{hms(s['end_ms'])}]`{flag} {emo}{s['final_text'] or s['clean_text']}\n")
    (outdir / "transcript.md").write_text("\n".join(md), encoding="utf-8")

    # 整理稿（段落级，可读优先）
    if paragraphs:
        pm = [f"# {Path(job['source_name']).stem} · 整理稿\n",
              f"- 时长 {hms(int(duration * 1000))}｜段落 {len(paragraphs)}\n\n---\n"]
        last = None
        for p in paragraphs:
            if p.get("spk") != last:
                pm.append(f"\n**{display.get(p.get('spk'), '未知')}**\n")
                last = p.get("spk")
            emo = ""
            try:
                from .emotion import EMO_ZH
                e_ = p.get("emotion")
                if e_ and e_ not in ("NEUTRAL", "EMO_UNKNOWN", "HAPPY"):
                    emo = f"（{EMO_ZH.get(e_, '')}）"
                elif p.get("event") == "Cry":
                    emo = "（哭）"
            except Exception:
                pass
            pm.append(f"`[{hms(p['start_ms'])}–{hms(p['end_ms'])}]` {emo}{p['text']}\n")
        (outdir / "refined.md").write_text("\n".join(pm), encoding="utf-8")

    if (wdir / "preview.m4a").exists():
        (outdir / "preview.m4a").write_bytes((wdir / "preview.m4a").read_bytes())
