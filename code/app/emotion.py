"""情绪与声学事件识别（SenseVoiceSmall）。

输出每个语音段的 emotion（HAPPY/SAD/ANGRY/NEUTRAL/FEARFUL/DISGUSTED/SURPRISED/UNKNOWN）
与 event（Speech/BGM/Applause/Laughter/Cry/Sneeze/Breath/Cough）。

用途：
1. 在逐字稿/整理稿上标注情绪（如"（哽咽）"）；
2. 让段落合并能跨过"哭泣造成的长停顿"，保持语义连贯；
3. 给纪要生成提供情绪基调。
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("tingji.emotion")

TAG = re.compile(r"<\|([A-Za-z_]+)\|>")
EMOTIONS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}
EVENTS = {"Speech", "BGM", "Applause", "Laughter", "Cry", "Sneeze", "Breath", "Cough"}

# 中文展示
EMO_ZH = {"HAPPY": "开心", "SAD": "难过", "ANGRY": "生气", "NEUTRAL": "平静",
          "FEARFUL": "害怕", "DISGUSTED": "厌恶", "SURPRISED": "惊讶", "EMO_UNKNOWN": "未识别"}
EVENT_ZH = {"Cry": "哭", "Laughter": "笑", "Sneeze": "喷嚏", "Breath": "呼吸声",
            "Cough": "咳嗽", "Applause": "掌声", "BGM": "背景声", "Speech": "说话"}

# 这类情绪下，长停顿更可能是哽咽而不是话轮结束
SAD_EMOTIONS = {"SAD", "FEARFUL", "EMO_UNKNOWN"}


class EmotionEngine:
    _model = None

    def __init__(self, device: str = "cpu"):
        self.device = device

    def _load(self):
        if EmotionEngine._model is None:
            from funasr import AutoModel
            log.info("加载 SenseVoice（情绪/事件）...")
            EmotionEngine._model = AutoModel(
                model="iic/SenseVoiceSmall", vad_model="fsmn-vad",
                device=self.device, disable_update=True, log_level="ERROR",
            )
            log.info("SenseVoice 就绪")
        return EmotionEngine._model

    def detect(self, wav_path: str) -> list[dict]:
        """SenseVoice 的句级 text 已剥离标签，情绪/事件要从顶层 text 按顺序对齐。"""
        model = self._load()
        res = model.generate(input=wav_path, language="zh", use_itn=True, sentence_timestamp=True)
        out = []
        for it in res:
            full = it.get("text") or ""
            # 每个 utterance 以 <|zh|>/<|en|> 开头
            parts = [p_ for p_ in re.split(r"(?=<\|(?:zh|en|ja|ko|yue)\|>)", full) if p_.strip()]
            sents = it.get("sentence_info") or []
            for i, s in enumerate(sents):
                text = s.get("text") or ""
                segment_text = parts[i] if i < len(parts) else ""
                tags = TAG.findall(segment_text)
                emo = next((t for t in tags if t in EMOTIONS or t == "EMO_UNKNOWN"), "EMO_UNKNOWN")
                ev = next((t for t in tags if t in EVENTS), None)
                out.append({
                    "start_ms": int(s.get("start", 0)),
                    "end_ms": int(s.get("end", 0)),
                    "text": TAG.sub("", text).strip(),
                    "emotion": emo,
                    "event": ev or "Speech",
                })
        return out

def attach(segs: list[dict], utts: list[dict]) -> None:
    _attach(segs, utts)


def _attach(segs: list[dict], utts: list[dict]) -> None:
        """把 SenseVoice 的情绪映射到我们自己的句子上（按时间重叠最大者）。"""
        for s in segs:
            a, b = s["start_ms"], s["end_ms"]
            best, best_ov = None, 0
            for u in utts:
                ov = min(b, u["end_ms"]) - max(a, u["start_ms"])
                if ov > best_ov:
                    best, best_ov = u, ov
            if best:
                s["emotion"] = best["emotion"]
                s["event"] = best["event"]
            else:
                s["emotion"] = s.get("emotion") or "EMO_UNKNOWN"
                s["event"] = s.get("event") or "Speech"


EmotionEngine.attach = staticmethod(_attach)


def label_of(seg: dict) -> str:
    """给一句生成展示用的情绪标签，如 '（哭）' '（难过）'，平静/说话返回空。"""
    ev = seg.get("event")
    emo = seg.get("emotion")
    if ev and ev in ("Cry", "Laughter"):
        return f"（{EVENT_ZH[ev]}）"
    if emo and emo not in ("NEUTRAL", "EMO_UNKNOWN", "HAPPY"):
        return f"（{EMO_ZH.get(emo, '')}）"
    return ""


def summarize_mood(segs: list[dict]) -> str:
    from collections import Counter
    c = Counter(s.get("emotion") for s in segs if s.get("emotion"))
    c.pop("NEUTRAL", None)
    c.pop("EMO_UNKNOWN", None)
    if not c:
        return ""
    top = ", ".join(f"{EMO_ZH.get(k, k)}×{v}" for k, v in c.most_common(3))
    cries = sum(1 for s in segs if s.get("event") == "Cry")
    return f"{top}" + (f"，哭泣片段 {cries} 段" if cries else "")
