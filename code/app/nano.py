"""Fun-ASR-Nano 引擎（sherpa-onnx 原生支持，无需 torch/funasr）。

借鉴 Brevia 的做法：
- **目标 8 秒一块**，在 6–10 秒区间内找能量最低点交付（不硬切句子）；
- 段首回补 300ms，切点落在换气处，减少首字被切掉；
- **每块不超过模型容量**：FunASR-Nano 的 KV 容量约 512，实测 >28s 开始截断，
  所以硬上限 22s，超长连续独白会被切开而不是整段丢弃。

实测（M4 CPU，int8）：RTF 0.13–0.20，自带标点，不需要额外接标点模型。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("tingji.nano")

MODEL_DIR = str(__import__("pathlib").Path(__file__).resolve().parent.parent.parent
                 / ".cache/sherpa/sherpa-onnx-funasr-nano-int8-2025-12-30")
MAX_SPEECH_SEC = 22.0
TARGET_SEC = 8.0
MIN_CUT_SEC = 6.0
MAX_CUT_SEC = 10.0
HEAD_PAD_SEC = 0.3


def split_chunks(audio: np.ndarray, sr: int, target: float = TARGET_SEC,
                 lo: float = MIN_CUT_SEC, hi: float = MAX_CUT_SEC,
                 max_sec: float = MAX_SPEECH_SEC, pad: float = HEAD_PAD_SEC,
                 overlap: float = HEAD_PAD_SEC,
                 silence_floor: float = 0.004) -> list[dict]:
    """按能量谷切块，返回 [{start, end, pad, boundary}, ...]。

    - `start/end`：**交付水位**（本块负责的新内容区间）；
    - `pad`：解码时向前多带的样本数（段首回补），不会越过上一块的 `end`；
    - `boundary`：`endpoint`（切在静音上，句末标点可信）/ `pause`（切在低能量但非静音）
      / `cut`（到 22 秒硬上限，句末标点不可信）。
    """
    n = len(audio)
    if n == 0:
        return []
    out: list[dict] = []
    i = 0
    hop = max(1, sr // 10)
    pad_n = int(pad * sr)
    ov_n = int(overlap * sr)
    while i < n:
        hard_end = min(n, i + int(max_sec * sr))
        target_end = min(n, i + int(target * sr))
        if target_end >= n:
            out.append({"start": i, "end": n, "pad": min(pad_n, i), "boundary": "endpoint"})
            break
        s0 = i + int(lo * sr)
        s1 = min(n, i + int(hi * sr))
        cut, boundary = target_end, "pause"
        if s1 - s0 > hop:
            best, best_v = target_end, float("inf")
            for k in range(s0, s1 - hop, hop):
                v = float(np.mean(audio[k:k + hop] ** 2))
                if v < best_v:
                    best_v, best = v, k
            cut = min(best, hard_end)
            if best_v <= silence_floor ** 2:
                boundary = "endpoint"
        if cut >= hard_end:
            boundary = "cut"
        if cut <= i + int(1.0 * sr):
            cut = min(target_end, hard_end)
        out.append({"start": i, "end": cut, "pad": min(pad_n, i), "boundary": boundary})
        i = cut
    # 交付水位：pad 不得越过上一块的 end（避免重复解码已交付内容）
    for idx in range(1, len(out)):
        out[idx]["pad"] = min(out[idx]["pad"], ov_n)
    return out


def dedupe_seam(prev_text: str, text: str, max_check: int = 12) -> str:
    """切点重叠导致的重复：current 开头若与 prev 结尾重叠，去掉重叠部分。"""
    if not prev_text or not text:
        return text
    limit = min(max_check, len(prev_text), len(text))
    for k in range(limit, 1, -1):
        if prev_text.endswith(text[:k]):
            return text[k:]
    return text


class FunasrNanoEngine:
    """离线识别器；模型缺失时抛错，由上层回退。"""

    _rec = None

    def __init__(self, root: Path | str = ".", threads: int = 6):
        self.root = Path(root)
        self.dir = self.root / MODEL_DIR
        self.threads = threads
        self.language = "中文"

    def available(self) -> bool:
        return (self.dir / "llm.int8.onnx").exists() and (self.dir / "encoder_adaptor.int8.onnx").exists()

    def _load(self):
        if FunasrNanoEngine._rec is None:
            import sherpa_onnx
            d = str(self.dir)
            log.info("加载 Fun-ASR-Nano（sherpa-onnx）…")
            FunasrNanoEngine._rec = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
                encoder_adaptor=f"{d}/encoder_adaptor.int8.onnx",
                llm=f"{d}/llm.int8.onnx",
                embedding=f"{d}/embedding.int8.onnx",
                tokenizer=f"{d}/Qwen3-0.6B",
                language=self.language,
                num_threads=self.threads,
                provider="cpu",
            )
        return FunasrNanoEngine._rec

    def transcribe(self, audio: np.ndarray, sr: int = 16000,
                   offset_ms: int = 0) -> list[dict]:
        rec = self._load()
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        out: list[dict] = []
        prev_text = ""
        for ch in split_chunks(audio, sr):
            a, b, pad = ch["start"], ch["end"], ch["pad"]
            st = rec.create_stream()
            st.accept_waveform(sr, audio[max(0, a - pad):b])   # 段首回补
            rec.decode_stream(st)
            text = (st.result.text or "").strip()
            text = dedupe_seam(prev_text, text)
            if not text:
                continue
            prev_text = text
            out.append({
                "start_ms": int(a / sr * 1000) + offset_ms,
                "end_ms": int(b / sr * 1000) + offset_ms,
                "raw_text": text,
                "boundary": ch["boundary"],
            })
        # 按"是否在句末交付"决定合并：Nano 自带标点，句末就切开，
        # 否则说明这句话被 8 秒目标切断了，接到下一块里（Brevia 的 endpoint 语义）。
        merged: list[dict] = []
        for s in out:
            text = s["raw_text"]
            if merged:
                prev = merged[-1]
                prev_text = prev["raw_text"]
                ends_ok = prev_text[-1:] in "。！？!?…；;" or len(prev_text) >= 60
                if not ends_ok:
                    prev["end_ms"] = s["end_ms"]
                    prev["raw_text"] = prev_text + text
                    continue
            merged.append(dict(s))
        return merged
