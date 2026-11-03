"""引擎封装：ffmpeg / FunASR / CAM++ / Ollama。"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("tingji.engines")


# ---------------------------------------------------------------- ffmpeg
class FFmpeg:
    @staticmethod
    def probe(path: Path) -> dict:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration,size,bit_rate:stream=sample_rate,channels,codec_name",
             "-of", "json", str(path)]
        )
        d = json.loads(out)
        fmt = d.get("format", {})
        st = (d.get("streams") or [{}])[0]
        return {
            "duration": float(fmt.get("duration") or 0),
            "size": int(fmt.get("size") or 0),
            "sample_rate": int(st.get("sample_rate") or 0),
            "channels": int(st.get("channels") or 0),
            "codec": st.get("codec_name"),
        }

    @staticmethod
    def normalize(src: Path, dst: Path, sr: int = 16000, loudnorm: bool = True) -> None:
        af = ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"] if loudnorm else []
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), *af,
             "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
            check=True,
        )

    @staticmethod
    def cut(src: Path, start_sec: float, dur_sec: float, dst: Path) -> None:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start_sec:.3f}",
             "-t", f"{dur_sec:.3f}", "-i", str(src), "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", str(dst)],
            check=True,
        )

    @staticmethod
    def to_m4a(src: Path, dst: Path, bitrate: str = "64k") -> None:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
             "-c:a", "aac", "-b:a", bitrate, str(dst)],
            check=True,
        )


# ---------------------------------------------------------------- ASR
class AsrEngine:
    """语音识别。provider=funasr（Paraformer，会议默认）或 sensevoice（叙事/情绪场景）。"""

    _model = None
    _provider = None

    def __init__(self, cfg, hot_l1: list[str] | None = None, provider: str | None = None):
        self.cfg = cfg
        self.hotword = " ".join(hot_l1 or [])
        self.provider = provider or getattr(cfg, "provider", "funasr")

    def _load(self):
        if AsrEngine._model is not None and AsrEngine._provider == self.provider:
            return AsrEngine._model
        from funasr import AutoModel

        def build(name: str):
            return AutoModel(model=name, vad_model="fsmn-vad", punc_model="ct-punc",
                             device=self.cfg.device, disable_update=True, log_level="ERROR")

        def build_sv():
            return AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                             device=self.cfg.device, disable_update=True, log_level="ERROR")

        if self.provider == "sensevoice":
            log.info("加载 SenseVoice …")
            AsrEngine._model = AutoModel(
                model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                device=self.cfg.device, disable_update=True, log_level="ERROR",
            )
        else:
            log.info("加载 FunASR Paraformer …")
            name = (self.cfg.model or "").strip() or "paraformer-zh"
            try:
                AsrEngine._model = build(name)
            except Exception as e:
                # 常见两种：模型名写空/写错（FunASR 会退化成找 "Conformer"），
                # 或缓存状态被上一次失败污染（报 not registered）。
                # 这里重置缓存并用完整 ModelScope id 再试一次，失败才抛。
                log.warning("模型 %s 加载失败（%s），用完整 id 重试一次", name, e)
                AsrEngine._model = None
                full = {
                    "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    "sensevoice": "iic/SenseVoiceSmall",
                }.get(name, "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
                AsrEngine._model = build(full)
        AsrEngine._provider = self.provider
        log.info("ASR 就绪（%s）", self.provider)
        return AsrEngine._model

    def transcribe(self, wav: Path, offset_ms: int = 0) -> list[dict]:
        if self.provider == "sensevoice":
            return self._transcribe_sensevoice(wav, offset_ms)
        if self.provider == "funasr_nano":
            from .nano import FunasrNanoEngine
            import soundfile as sf_
            eng = FunasrNanoEngine(".", threads=max(4, (self.cfg.device == "cpu") and 6 or 6))
            if not eng.available():
                raise RuntimeError("Fun-ASR-Nano 模型未安装（.cache/sherpa/sherpa-onnx-funasr-nano-int8-2025-12-30）")
            au, sr_ = sf_.read(str(wav), dtype="float32")
            return eng.transcribe(au, int(sr_), offset_ms)
        model = self._load()
        kw: dict[str, Any] = {"sentence_timestamp": True, "return_raw_text": True,
                              "batch_size_s": self.cfg.batch_size_s}
        if self.hotword:
            kw["hotword"] = self.hotword
        res = model.generate(input=str(wav), **kw)[0]
        from . import words as wm
        segs = []
        for s in res.get("sentence_info") or []:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            start = int(s["start"]) + offset_ms
            end = int(s["end"]) + offset_ms
            # Paraformer：raw_text 的空格分隔 token 与 timestamp 一一对应 → 精确词级
            toks = (s.get("raw_text") or "").split()
            tss = s.get("timestamp") or []
            if toks and len(toks) == len(tss):
                ws = wm.from_tokens(toks, tss, offset_ms)
            else:
                ws = wm.interpolate(text, start, end)
            segs.append({
                "start_ms": start,
                "end_ms": end,
                "raw_text": text,
                "words": ws,
            })
        return segs


    def _transcribe_sensevoice(self, wav: Path, offset_ms: int = 0) -> list[dict]:
        """SenseVoice：一次拿到文本 + 情绪 + 事件；标点由模型自带。"""
        import re as _re
        model = self._load()
        res = model.generate(input=str(wav), language="zh", use_itn=True, sentence_timestamp=True)
        tag = _re.compile(r"<\|([A-Za-z_]+)\|>")
        emotions = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}
        events = {"Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough"}
        out = []
        for it in res:
            full = it.get("text") or ""
            parts = [p_ for p_ in _re.split(r"(?=<\|(?:zh|en|ja|ko|yue)\|>)", full) if p_.strip()]
            for i, s in enumerate(it.get("sentence_info") or []):
                seg_text = parts[i] if i < len(parts) else ""
                tags = tag.findall(seg_text)
                emo = next((t for t in tags if t.upper() in emotions or t == "EMO_UNKNOWN"), "EMO_UNKNOWN")
                ev = next((t for t in tags if t in events), "Speech")
                text = tag.sub("", s.get("text") or "").strip()
                if not text:
                    continue
                _st = int(s.get("start", 0)) + offset_ms
                _en = int(s.get("end", 0)) + offset_ms
                from . import words as _wm
                out.append({
                    "start_ms": _st,
                    "end_ms": _en,
                    "raw_text": text,
                    "words": _wm.interpolate(text, _st, _en),
                    "emotion": emo.upper() if emo != "EMO_UNKNOWN" else "EMO_UNKNOWN",
                    "event": ev,
                })
        return out


# ---------------------------------------------------------------- 说话人
class SpeakerEngine:
    """CAM++ ONNX（sherpa-onnx）抽取 192 维声纹向量。"""

    def __init__(self, model_path: str):
        import sherpa_onnx as so
        self.sr = 16000
        cfg = so.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=4)
        self.ex = so.SpeakerEmbeddingExtractor(cfg)

    def embed(self, samples: np.ndarray) -> np.ndarray | None:
        st = self.ex.create_stream()
        st.accept_waveform(self.sr, samples)
        st.input_finished()
        if not self.ex.is_ready(st):
            return None
        v = np.array(self.ex.compute(st), dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else None


# ---------------------------------------------------------------- LLM
class LlmEngine:
    def __init__(self, cfg, model: str):
        self.cfg = cfg
        self.model = model

    def chat(self, prompt: str, system: str = "", json_mode: bool = False,
             max_tokens: int = 1200, temperature: float = 0.15) -> str:
        import httpx
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "keep_alive": self.cfg.keep_alive,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"
        r = httpx.post(self.cfg.base_url.rstrip("/") + "/api/generate", json=payload,
                       timeout=self.cfg.timeout_sec)
        r.raise_for_status()
        return (r.json().get("response") or "").strip()

    def alive(self) -> bool:
        import httpx
        try:
            httpx.get(self.cfg.base_url.rstrip("/") + "/api/version", timeout=3).raise_for_status()
            return True
        except Exception:
            return False
