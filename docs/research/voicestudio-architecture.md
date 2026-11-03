# 架构分析 · VoiceStudio 中文版（bayshier/voicestudio-zh）

> 分析对象：https://github.com/bayshier/voicestudio-zh ｜ 分析时间 2026-10-31
> 方法：拉取 `README`、`backend/services/asr_backend.py`（3,387 行）、
> `backend/services/segmentation.py`（764 行）、`docs/engines/funasr.md`、`docs/features/diarization.md`
> 一手源码阅读（raw.githubusercontent），非二手描述。

---

## 0. 一句话概括

**它把"ASR 引擎"抽象成了可插拔的 11 个后端，并把"分段 + 说话人归属"独立成一层后处理**；
我们（听记）把力气花在后者之后（热词/ITN/LLM 顺滑/情绪/纪要）。
**两者的强项正好互补**——它强在引擎工程与失败降级，我们强在内容加工。

| | VoiceStudio | 听记（本项目） |
|---|---|---|
| 定位 | ElevenLabs 替代品（TTS 为主，ASR 是配角） | 会议转写 + Obsidian 归档（ASR 为主） |
| 规模 | 2,057 文件，React + FastAPI，AGPL-3.0 | ~3.4k 行 Python，AGPL 无关（自研） |
| ASR 引擎 | **11 个可插拔**，设备感知路由 | 2 个（Paraformer / SenseVoice） |
| 说话人 | pyannote（gated）或 FunASR cam++ 内置 | 自研 CAM++ 全局聚类 + F0 校正 |
| 时间戳 | **词级**（wav2vec2 强制对齐） | 句级（词级仅 SenseVoice token） |
| 内容加工 | 无（靠引擎自带） | 热词三层 / ITN / LLM 段落顺滑 / 情绪 / 纪要 |

---

## 1. 分层架构

```
React 前端（DubTab / Dictation / BatchQueue / Diagnostics）
        │ REST + SSE/WSS
FastAPI 后端（100+ 端点 · SQLite · 引擎路由）
        │
┌───────┴────────┬──────────────┬─────────────┬──────────────┐
│ ASR 引擎层      │ 分段/归属层   │ 分离/克隆    │ 引擎路由      │
│ 11 个 Backend   │ segmentation │ Demucs/TTS   │ GPU 预检      │
└────────────────┴──────────────┴─────────────┴──────────────┘
        CUDA / MPS / ROCm / CPU（自动检测 + 路由）
```

关键点：**ASR 引擎与"分段/说话人归属"是两层**。引擎只负责出词/句与时间戳，
归属逻辑集中在 `services/segmentation.py`，对所有引擎复用。这是它最值得借鉴的结构。

---

## 2. 引擎抽象：11 个后端一套契约

`ASRBackend`（ABC）规定：

| 成员 | 作用 |
|---|---|
| `id` / `display_name` / `gpu_compat` | 身份与硬件能力声明（`("cuda","cpu")` 等） |
| `is_available() -> (bool, reason)` | 可安装性探测，返回人话原因（"funasr not installed. Install with: uv pip install funasr"） |
| `transcribe(path, *, word_timestamps=True) -> {chunks, segments, language}` | **统一输出形状** |
| `ensure_loaded()` / `unload()` | 显式生命周期（含 `gc.collect()` + `torch.cuda.empty_cache()`） |
| `requires_full_audio_for_speaker_consistency` | 声明"必须整文件一次调用才能保持说话人一致" |
| `warmup()`（部分） | 预热 |

**统一输出**是核心：`chunks`（文本+时间）、`segments`（句级）、`language`。
上层 `segmentation` 不需要知道底层是 Whisper 还是 SenseVoice。

**11 个后端**：WhisperX（默认）、Faster-Whisper、Faster-Whisper（子进程隔离）、
MLX Whisper、PyTorch Whisper、Parakeet TDT（NeMo）、Moonshine、FunASR、sherpa-onnx、OpenAI 兼容（远程）。

---

## 3. 设备感知路由（这是它最"工程"的一处）

`_auto_detect()` 的注释里直接写了实测数据：

> WhisperX / faster-whisper 是 **CTranslate2**，**没有 Metal 后端**——在 Mac 上无论有没有 GPU 都跑 CPU。
> M2 实测：30s 配音片段，whisper-large-v3，**WhisperX(CPU) 90.4s vs MLX(GPU) 20.5s**，
> 3 倍慢于实时，16 分钟视频要转 48 分钟，用户以为卡死（issue #1127）。

于是路由规则变成：

1. **Apple Silicon** → `mlx-whisper`（同一模型上 GPU，再叠 WhisperX 的 wav2vec2 强制对齐，词级时间不变）
2. **ROCm** → `pytorch-whisper`（CTranslate2 无 HIP 后端，GPU 会闲置）
3. 其他 → `whisperx`（CUDA 上正确）
4. 兜底 → `faster-whisper`（无强制对齐）→ `pytorch-whisper`
5. 显式 `OMNIVOICE_ASR_BACKEND` 永远优先

> **对我们的直接意义**：这印证了我们实测的"mlx-whisper 在 M4 上并不比 CPU FunASR 快"——
> 但那是因为我们比的是 **1.4B 的多模态 AR 模型**；对 Whisper 这类编码器-解码器，MLX 的收益是 4 倍。
> 也就是说"MLX 更快"不是普适结论，**取决于模型结构与解码方式**。

---

## 4. 默认链路：WhisperX + pyannote

```
音频 → (可选 Demucs 分离) → faster-whisper 转写（段级）
     → wav2vec2 强制对齐（词级，±10–30ms）
     → pyannote 说话人分离（turns）
     → segmentation：词→句、短段合并、跨说话人重切、标签平滑
     → 字幕/配音/时间轴
```

`docs/features/diarization.md` 明确：pyannote 模型 **gated**（需要 HF token **并且**接受 license），
下载约 600MB；失败时**降级为静音间隔启发式**，并给出原因码：

- `diarization_skipped:no_token` — 没有 token
- `diarization_skipped:401` — 有 token 但没接受条款
- `diarization_skipped:network` — 下载中断

降级后的启发式**自己承认不准**："音高相近或快速抢话的人会被合并"，但保证任务能跑完。

---

## 5. FunASR 路径（和我们最相关的一条）

`FunASRBackend` 的几个细节值得逐条看：

```python
kwargs = {"model": "iic/SenseVoiceSmall", "vad_model": "fsmn-vad",
          "spk_model": "cam++", "spk_mode": "vad_segment",
          "output_timestamp": True, "language": "auto", "use_itn": True}
```

| 细节 | 原因（源码注释） |
|---|---|
| `spk_mode="vad_segment"` | FunASR 1.3.1 默认 `punc_segment` 需要单独的标点模型，SenseVoice 不带标点模型会崩 |
| `output_timestamp=True` | `vad_segment` 模式靠 SenseVoice 的时间戳构建 `sentence_info` |
| **整文件一次 `generate()`** | CAM++ 的簇 ID 是**每次调用内**分配的；分片调用会导致"Speaker 1"在不同片指不同人 |
| 富 token 剥离 | SenseVoice 输出 `<\|zh\|><\|SAD\|><\|Cry\|>` 之类标记，`_clean_funasr_text` 统一剥掉 |
| 句级而非词级 | 文档明确它**不适合口型同步**，词级时间要用 WhisperX/MLX |

> 这正是我们踩过的坑：我们也发现 CAM++ 的簇是"调用内"的，所以**自己做了全局聚类**（把
> embedding 全存下来统一聚）。他们选择"整文件一次调用"来规避，我们选择"自己管全局"——
> 两条路都对，但**我们的做法才能做跨片、可重算、可合并**。

---

## 6. 分段与说话人归属层（`segmentation.py`，最值得抄的部分）

这 764 行解决的就是我们项目里"断句碎、说话人漂移"的问题。核心函数：

| 函数 | 做什么 | 对应我们的痛点 |
|---|---|---|
| `_build_segments_from_words` | 词级 → 句子，按标点和长度切 | 我们没有词级，只能句级 |
| `_merge_short` / `_stitch_adjacent_shorts` | **多轮迭代**把短段折进相邻同人段；超短段（<0.5s 或 <4 字）允许跨人、跨更大间隔合并 | 我们的"30% 句子 <1 秒"问题 |
| `_resplit_core` | 一段里跨多个说话人时，**按词级边界重切**，切出的片段时间用词时间，整体覆盖不变 | 我们没有：一句两人说就归给一个人 |
| `_fill_and_smooth` | 无覆盖的词向前/向后填充；**单个词翻转（A-B-A）平滑回 A** | 我们的短句回填较粗糙 |
| `assign_speakers_heuristic` | pyannote 不可用时的静音间隔兜底 | 我们的"无 token 就用不了"问题不存在（自研） |

`_fill_and_smooth` 的注释写得很直白：

> "Forward/back-fill gaps … and smooth single-word flips, so one mis-attributed word inside a
> speaker's run (diarization noise) doesn't trigger a spurious split."

**这正是我们"说话人 27 人 / 115:6"那类问题的标准解法**：先平滑标签，再合并短段。

---

## 7. 鲁棒性工程（它比我们成熟的地方）

| 机制 | 实现 | 为什么值得学 |
|---|---|---|
| **子进程隔离** | `faster-whisper-isolated` 后端，ASR 崩溃不拖垮整个应用 | 我们 16GB 机器最怕 OOM 把整个 worker 带走 |
| **超时 + 池重置** | `ASRTimeoutError`、`reset_pool_after_wedge()` | 我们目前只有 flock，没有超时熔断 |
| **OOM 回退** | `tests/test_asr_oom_fallback.py` | 我们只是抛错 |
| **compute_type 回退** | float16 失败自动 int8 重试，`ASR_COMPUTE_TYPE` 逃生舱 | 我们没有 |
| **VRAM 降级** | `_degrade_for_vram()` 按显存选模型规模 | 我们靠手动 `--speakers` |
| **pickle 白名单** | VAD 模型的 pickle 全局限制 | 安全 |
| **惰性导入加固** | `_harden_speechbrain_lazy_imports()` | 依赖缺失时给可读错误 |
| **失败原因码** | `diarization_skipped:*` | 用户能自己判断"是 token 问题还是网络问题" |

---

## 8. OpenAI 兼容层

两个方向都有：

1. **对外**：`POST /v1/audio/transcriptions`（json/text/verbose_json/srt/vtt），`voice`/`model` 参数
   映射到当前启用的引擎，无 API key 校验（本地）。
2. **对内**：`OpenAICompatASRBackend` 是**纯网络客户端**——把音频发给自托管的
   **Qwen3-ASR / FunASR / SenseVoice 服务器**，理由是"transformers 还没稳定支持 Qwen3-ASR
   （issue #877），但 OpenAI 兼容接口现在就能用"。

> 这个"用兼容接口绕过上游依赖"的思路很实用：**我们的 `/v1` 已经做了对外，但没做"对内反向接远程引擎"**。

---

## 9. 与我们项目的对比与差距

| 维度 | VoiceStudio | 听记 | 差距判断 |
|---|---|---|---|
| 引擎数量 | 11 | 2 | 他们赢在广度，我们不需要那么多 |
| 词级时间戳 | ✅ wav2vec2 对齐 | ⚠️ 仅句级 | **我们缺** |
| 跨说话人重切 | ✅ 词级重切 | ❌ | **我们缺** |
| 标签平滑 | ✅ A-B-A 平滑 | ⚠️ 粗粒度 | **我们缺** |
| 短段合并 | ✅ 多轮迭代 | ⚠️ 段落级 | 可借鉴 |
| 失败降级 | ✅ 原因码 + 兜底 | ⚠️ 部分 | 可借鉴 |
| 子进程隔离 | ✅ | ❌ | 可借鉴 |
| 热词 | ❌（只有引擎自带） | ✅ 三层 | **我们赢** |
| ITN | ❌ | ✅ 自研规则 | **我们赢** |
| LLM 顺滑 | ❌ | ✅ 段落级 + 校验 | **我们赢** |
| 情绪/事件 | 仅 SenseVoice 标记 | ✅ 已解析并用于合并 | **我们赢** |
| 纪要/章节 | 无（TTS 产品） | ✅ | **我们赢** |
| Obsidian 归档 | ❌ | ✅ | **我们赢** |
| 许可 | AGPL-3.0（传染） | 自研 | **不要抄代码，只抄思路** |

---

## 10. 结论：值得抄的 5 条

1. **把"分段 + 说话人归属"抽成独立层**（我们现在散在 pipeline 里）。
2. **词级重切 + 标签平滑**：这是解决"同一句两人说"和"短段乱切"的正解，优先级最高。
3. **失败原因码 + 优雅降级**：ASR/分离失败要能说清"为什么"，而不是一个 500。
4. **ASR 子进程隔离**：16GB 机器上，一次 OOM 不该毁掉整个任务队列。
5. **允许"远程兼容引擎"作为后端**：将来接 Qwen3-ASR / 更强模型时不用改架构。

## 11. 不建议照搬的 3 条

1. **pyannote 作为默认分离**：gated + 600MB + license 点击，和"开箱即用离线"冲突（我们已避开）。
2. **CTranslate2/Whisper 全家桶**：中文场景我们实测 FunASR 更快更准（见 [技术选型 §2.4](../01-技术选型报告.md)）。
3. **AGPL-3.0 代码**：只能借鉴设计，不能复制实现。
