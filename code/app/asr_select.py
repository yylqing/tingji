"""自动选识别引擎（`asr.provider: auto`）。

依据**识别前的音频特征**判断场景，而不是等识别完再猜（那时已经晚了）：

| 特征 | 判断 | 引擎 |
|---|---|---|
| 时长 ≥ 20 分钟 | 会议/讲座 | funasr（Paraformer，快、词级时间戳） |
| 有效语音占比 ≥ 55% 且 ≥5 分钟 | 连续讲话多，会议/演讲 | funasr |
| 其余（碎片化、安静、对话/情绪） | 私人录音、访谈 | funasr_nano（句子连贯性好） |
| Nano 模型未安装 | 回退 | sensevoice（比 Paraformer 稳、比 Nano 快） |

实测依据（10.6 分钟情绪对话）：Nano 段落中位数 1.55s、字数比另两个多 48%、
句子明显更连贯；而 60 秒业务通话里 Paraformer 的术语（ICT）更准——所以按场景分流。
"""
from __future__ import annotations

MEETING = "funasr"
NARRATIVE = "funasr_nano"
FALLBACK = "sensevoice"


def choose(duration_sec: float, speech_ratio: float, nano_available: bool = True,
           n_speakers: int = 0) -> tuple[str, str]:
    """返回 (引擎, 理由)。

    优先看**探测到的说话人数**（比语音占比更能区分场景）：
    ≥3 人 → 会议；1 人 → 独白/讲座；2 人 → 一对一对话（最可能是私人录音）。
    """
    if duration_sec >= 1200:
        return MEETING, f"时长 {duration_sec / 60:.0f} 分钟，按长会处理"
    if n_speakers >= 3:
        return MEETING, f"探测到 {n_speakers} 位说话人，按会议处理"
    if n_speakers == 1:
        return MEETING, "单人长讲，按讲座/独白处理"
    if nano_available:
        why = "一对一对话" if n_speakers == 2 else f"语音占比 {speech_ratio:.0%}、未探到多人"
        return NARRATIVE, f"{why}，按叙事处理"
    return FALLBACK, "Nano 模型缺失，回退 SenseVoice"
