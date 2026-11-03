"""OpenAI 兼容 API：/v1/models、/v1/audio/transcriptions（同步 + 202 异步扩展）。

设计见 docs/06-开发方案.md §5.2。
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import textproc as tp
from .engines import AsrEngine, FFmpeg
from .obsidian import srt_time

# 浏览器 MediaRecorder 默认产出 webm/opus，iOS 常见 m4a，安卓常见 3gp/amr——
# 这些都必须收，否则前端录音直接 400。
log = logging.getLogger("tingji.api")

ALLOWED = {
    ".wav", ".mp3", ".aac", ".m4a", ".mp4", ".mov",
    ".webm", ".weba", ".opus", ".ogg", ".oga", ".flac",
    ".wma", ".amr", ".3gp", ".caf", ".aiff", ".aif", ".m4b", ".mkv",
}


def _auth(cfg, request: Request) -> None:
    key = getattr(cfg.openai_api, "api_key", "")
    if not key:
        return
    got = request.headers.get("authorization", "").replace("Bearer ", "").strip()
    if got != key:
        raise HTTPException(401, "invalid api key")


def _vtt(segs, names) -> str:
    out = ["WEBVTT", ""]
    for s in segs:
        out.append(f"{srt_time(s['start_ms']).replace(',', '.')} --> {srt_time(s['end_ms']).replace(',', '.')}")
        who = names.get(s.get("spk", -1))
        out.append(f"[{who}] {s['text']}" if who else s["text"])
        out.append("")
    return "\n".join(out)


def _srt(segs, names) -> str:
    out = []
    for i, s in enumerate(segs, 1):
        out.append(f"{i}\n{srt_time(s['start_ms'])} --> {srt_time(s['end_ms'])}")
        who = names.get(s.get("spk", -1))
        out.append(f"[{who}] {s['text']}" if who else s["text"])
        out.append("")
    return "\n".join(out)


def parse_prompt(prompt: str, limit: int = 50) -> list[str]:
    """把 OpenAI 的 prompt（热词提示）拆成词条。

    Memos/Whisper 客户端一般写成"人名：张三、李四。术语：Kubernetes、OAuth。"，
    这里按标点/空白拆开，去掉标签词和过长的句子。
    """
    import re as _re
    if not prompt:
        return []
    text = _re.sub(r"(人名|术语|专有名词|关键词|热词|提示词)[:：]", " ", prompt)
    parts = _re.split(r"[、,，。;；\s/|]+", text)
    out, seen = [], set()
    for w in parts:
        w = w.strip(" ：:()（）\"'“”")
        if not w or len(w) > 20 or w in seen:
            continue
        if w in ("", "无"):
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out


def quick_transcribe(cfg, db, path: Path, include_speakers: bool = False,
                     extra_hotwords: list[str] | None = None) -> dict:
    """同步转写：归一化 + ASR（含标点/时间戳/热词），可选说话人。"""
    hot_rows = db.list_hotwords(only_enabled=True) if cfg.hotwords.enabled else []
    if extra_hotwords:
        # 请求级热词（来自 prompt）：权重高于库里的，排前面
        hot_rows = [{"term": t, "weight": 999, "enabled": 1} for t in extra_hotwords] + list(hot_rows)
    hot_l1 = [r["term"] for r in sorted(hot_rows, key=lambda r: -r["weight"])][: cfg.hotwords.l1_max]
    tmpdir = Path(tempfile.mkdtemp())
    wav = tmpdir / "16k.wav"
    FFmpeg.normalize(path, wav)
    asr = AsrEngine(cfg.asr, hot_l1=hot_l1)
    segs = asr.transcribe(wav)
    for s in segs:
        s["text"] = tp.clean_text(s["raw_text"])[0]
    matcher = tp.HotwordMatcher(hot_rows, cfg.hotwords.l2_threshold) if hot_rows else None
    if matcher:
        for s in segs:
            s["text"], _ = matcher.apply(s["text"])
    names: dict[int, str] = {}
    if include_speakers and len(segs) > 1 and cfg.diarization.enabled:
        from . import diarize as dia
        from .engines import SpeakerEngine
        import soundfile as sf
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
        E = np.stack([e if e is not None else np.zeros(192, dtype=np.float32) for e in embs])
        D = np.array(durs)
        labels = dia.cluster(E, D, cfg.diarization.cluster_threshold,
                             n_clusters=(cfg.diarization.max_speakers or None))
        cents = dia.centroids(E, labels)
        labels = dia.merge_by_centroid(labels, cents, cfg.diarization.merge_threshold)
        labels = dia.reassign(E, labels, dia.centroids(E, labels))
        labels = dia.fill_short(segs, labels, cfg.diarization.min_cluster_segment_sec)
        for s, l in zip(segs, labels):
            s["spk"] = int(l)
        # 用已登记的声纹给说话人命名；匹配不上就退回"说话人N"
        try:
            from .pipeline import match_voiceprints
            names, _detail = match_voiceprints(
                db, cents,
                float(getattr(cfg.diarization, "match_threshold", 0.35)),
                float(getattr(cfg.diarization, "match_margin", 0.08)),
                float(getattr(cfg.diarization, "channel_threshold", 0.75)),
                int(getattr(cfg.diarization, "max_channels", 4)))
        except Exception as e:
            log.warning("声纹匹配失败（忽略）: %s", e)
    return {"segments": segs, "names": names}


def speaker_text(segs: list[dict], names: dict[int, str]) -> str:
    """把分段转成"说话人：内容"，相邻同一人合并，避免一行一句。"""
    lines: list[tuple[str, str]] = []
    for s in segs:
        txt = (s.get("text") or "").strip()
        if not txt:
            continue
        spk = s.get("spk", -1)
        who = names.get(spk) or (f"说话人{spk + 1}" if isinstance(spk, int) and spk >= 0 else "未知")
        if lines and lines[-1][0] == who:
            lines[-1] = (who, lines[-1][1] + txt)
        else:
            lines.append((who, txt))
    return "\n".join(f"{who}：{txt}" for who, txt in lines)


def build_router(cfg, db) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    def models(request: Request):
        _auth(cfg, request)
        return {"object": "list", "data": [
            {"id": "tingji-asr", "object": "model", "owned_by": "tingji"},
            {"id": "tingji-asr-hq", "object": "model", "owned_by": "tingji"},
            {"id": "whisper-1", "object": "model", "owned_by": "tingji"},
        ]}

    @router.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request,
                             file: UploadFile = File(...),
                             model: str = Form("tingji-asr"),
                             language: str = Form("zh"),
                             prompt: str = Form(""),
                             response_format: str = Form("json"),
                             temperature: float = Form(0.0)):
        _auth(cfg, request)
        suffix = Path(file.filename or "a.wav").suffix.lower()
        # 白名单外的扩展名也接受：先落盘再用 ffprobe 判断能不能解码，
        # 失败才报 400（客户端给的名字千奇百怪，别因为后缀拒掉能解的音频）
        if not suffix or len(suffix) > 6:
            suffix = ".bin"
        raw = await file.read()
        max_bytes = int(getattr(cfg.openai_api, "max_upload_gb", 4) * 1024 ** 3)
        if len(raw) > max_bytes:
            return JSONResponse({"error": {"message": "file too large"}}, status_code=413)
        tmpdir = Path(tempfile.mkdtemp())
        src = tmpdir / f"in{suffix}"
        src.write_bytes(raw)
        try:
            info = FFmpeg.probe(src)
        except Exception:
            return JSONResponse({"error": {"message": f"unsupported format {suffix}（ffmpeg 无法解码）"}},
                                status_code=400)
        duration = info["duration"]
        if duration <= 0:
            return JSONResponse({"error": {"message": "empty audio"}}, status_code=400)

        sync_max = int(getattr(cfg.openai_api, "sync_max_audio_sec", 1800))
        if duration > sync_max:
            # 异步：入队走完整流水线
            from .pipeline import sha256_file
            from .worker import ingest
            dest = cfg.input_path / Path(file.filename or f"api{suffix}").name
            dest.write_bytes(raw)
            jid = ingest(db, cfg, dest)
            if not jid:                       # 已有同内容任务：复用
                ex = db.find_by_sha(sha256_file(dest))
                jid = ex["id"] if ex else None
            if not jid:
                return JSONResponse({"error": {"message": "cannot enqueue"}}, status_code=500)
            jrow = db.get_job(jid) or {}
            body = {"id": jid, "object": "transcription.job", "status": jrow.get("status", "queued"),
                    "estimated_sec": int(duration / 12 * 1.0)}
            return JSONResponse(body, status_code=202,
                                headers={"Location": f"/v1/audio/transcriptions/{jid}"})

        t0 = time.time()
        include_spk = model.endswith("-hq") or model.endswith("_hq") or model.endswith("-speakers")
        out = quick_transcribe(cfg, db, src, include_speakers=include_spk,
                               extra_hotwords=parse_prompt(prompt))
        segs, names = out["segments"], out["names"]
        if include_spk:
            spk_ids = {s.get("spk") for s in segs
                       if isinstance(s.get("spk"), int) and s.get("spk") >= 0}
            names = {k: names.get(k) or f"说话人{k + 1}" for k in sorted(spk_ids)}
        text = speaker_text(segs, names) if include_spk else "".join(s["text"] for s in segs)
        elapsed = round(time.time() - t0, 1)

        if response_format == "text":
            return PlainTextResponse(text)
        if response_format == "srt":
            return PlainTextResponse(_srt(segs, names), media_type="application/x-subrip")
        if response_format == "vtt":
            return PlainTextResponse(_vtt(segs, names), media_type="text/vtt")
        if response_format == "verbose_json":
            return {
                "task": "transcribe", "language": language or "zh",
                "duration": round(duration, 2), "text": text,
                "segments": [{
                    "id": i, "start": round(s["start_ms"] / 1000, 2), "end": round(s["end_ms"] / 1000, 2),
                    "text": s["text"],
                    **({"speaker": names.get(s.get("spk", -1)) or f"说话人{(s.get('spk') or 0) + 1}"}
                       if include_spk and s.get("spk") is not None and s.get("spk") >= 0 else {}),
                } for i, s in enumerate(segs)],
                "x_tingji": {"model": model, "elapsed_sec": elapsed,
                             "hotwords": len(db.list_hotwords(True)),
                             **({"speakers": [names[k] for k in sorted(names)]} if include_spk else {})},
            }
        return {"text": text, "x_tingji": {"model": model, "elapsed_sec": elapsed}}

    @router.get("/v1/audio/transcriptions/{jid}")
    def job_status(jid: str, request: Request):
        _auth(cfg, request)
        job = db.get_job(jid)
        if not job:
            raise HTTPException(404, "not found")
        if job["status"] != "done":
            return {"id": jid, "status": job["status"], "progress": job["progress"],
                    "stage": job["stage"]}
        rows = db.get_segments(jid)
        return {"id": jid, "status": "done", "text": "".join(
            (r["final_text"] or r["clean_text"] or r["raw_text"]) for r in rows),
            "segments": [{"start": round(r["start_ms"] / 1000, 2), "end": round(r["end_ms"] / 1000, 2),
                          "text": r["final_text"] or r["clean_text"] or r["raw_text"]} for r in rows]}

    @router.delete("/v1/audio/transcriptions/{jid}")
    def cancel(jid: str, request: Request):
        _auth(cfg, request)
        db.update_job(jid, status="canceled", message="API 取消")
        return {"id": jid, "status": "canceled"}

    return router
