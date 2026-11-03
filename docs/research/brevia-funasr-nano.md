# 调研 · Brevia 的转写生成实现（Fun-ASR-Nano）

> 对象：https://github.com/zerolovesea/Brevia ｜2026-10-31
> 依据：`backend/asr.py`（1529 行）、`backend/diarization_engine.py`、`backend/models.json` 一手源码
> 背景：你用 Brevia 客户端跑同一段录音，觉得准确率和人声分离明显更好，用的是 Fun-ASR-Nano。

---

## 0. 结论先行

1. **它用的模型是 Fun-ASR-Nano**，通过 **sherpa-onnx 的 `OfflineRecognizer.from_funasr_nano`** 调用
   （encoder_adaptor + llm + embedding + Qwen3-0.6B tokenizer，int8）。**我们本地的 sherpa-onnx 2.0.1 也支持同一条 API**
   ——已实测跑通，不需要 torch/funasr。
2. **它"听起来更准"的一半功劳不在模型，在切分策略**：8 秒目标块 + 能量谷交付 + 300ms 段首回补 +
   切点重叠/水位 + 按模型设最大语音段（Nano 22 秒）。
3. **它的人声分离比我们细**：先用 VAD 把非语音掩成静音，再跑离线分离，然后**每个 turn 按 15 秒窗口提声纹**、
   在线聚类保持说话人 ID。
4. 我们的 A/B 结果：**Nano 不是全面更好**（有胜有负，见 §5），它的优势是自带标点、句子更完整；
   劣势是 CPU 上比 SenseVoice 慢约 10 倍。

---

## 1. 它怎么调模型

```python
self.recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
    encoder_adaptor=f"{path}/encoder_adaptor.int8.onnx",
    llm=f"{path}/llm.int8.onnx",
    embedding=f"{path}/embedding.int8.onnx",
    tokenizer=f"{path}/Qwen3-0.6B",
    language=self._funasr_nano_language(language),   # "中文"（自然语言，不是 zh）
    num_threads=..., provider="cpu",
)
```

两个关键细节：

- **语言提示用自然语言**（`中文`/`英文`/`粤语`），因为 Nano 的 prompt 是
  `"语音转写成{language}"`，塞 ISO 代码会让自动检测把中文误判成日语；
- **音色/语种由 encoder_adaptor + embedding 决定，LLM 只负责解码**——这也是为什么它比
  Paraformer 更能"补全"句子（有 LLM 的语言先验）。

模型清单里给了完整下载信息（我们直接用了）：

| 项 | 值 |
|---|---|
| 归档 | `sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2`（802 MB） |
| 解码后 | ~1.0 GB（llm 573M + encoder 227M + embedding 148M + tokenizer） |
| sha256 | `eb43d7…9476b`（我们校验通过） |
| 国内源 | `modelscope.cn/models/fuyuantech/…`（实测 47 秒下完，17 MB/s） |
| 内存下限 | 清单标注 2.3 GB |

---

## 2. 切分：`SentenceVAD`（最值得抄的部分）

它的注释写得很直白，直接体现了踩过的坑：

> **FunASR Nano 的 KV 容量约 512，实测音频超过 ~28s 后输出开始截断、约 29.5s 起整段解码为空**
> ——连续独白一旦按 30s 硬切就会整段漏识别。

所以它做了四件事：

| 机制 | 参数 | 作用 |
|---|---|---|
| 短停顿检测 | `min_silence_duration=0.1` | 先找到"换气"的位置 |
| 聚合到目标长度 | `target_seconds=8.0` | 太碎的句子识别质量差，凑到 8 秒 |
| **能量谷交付** | 8–22s 窗口内找最低能量点 | 不在一句话中间硬切 |
| 段首回补 | `speech_pad_ms=300` | 抵消端点检测抖动，避免首字被切 |
| 切点回看 | `cut_overlap_ms = max(pad, CUT_OVERLAP_MS)` | 让被切的那个字在下一块里重新听到 |
| 已交付水位 | `delivered_until` | 回补/回看**不得越过**已交付样本，重复部分由接缝去重消掉 |
| 按模型上限 | `funasr-nano: 22.0`，qwen3 50、whisper 30、parakeet 20 | 超容量就切，别整段丢 |

**边界来源标记**（很妙的一个设计）：每个交付段带 `boundary`：

- `endpoint`：说话人真的停顿了 → 句末标点**可信**；
- `pause`：到目标长度切在能量低谷 → 末尾标点是模型对截断音频的**补全，不可信**；
- `cut`：到硬上限被切开 → 同样不可信。

上层据此决定要不要修补句末标点——这比"所有标点一视同仁"精细得多。

**安静语音兜底**：VAD 判成非语音的长段（音乐里的轻声、电话音），如果电平比 >`quiet_speech_level_ratio(0.08)`、
时长在 1–20 秒之间，仍然产出一段，避免"字幕里凭空少一句且没有任何提示"。

---

## 3. 说话人：VAD 掩蔽 + 15 秒窗口声纹 + 在线聚类

`diarization_engine.py` 的流程：

```
读窗口音频 → 把 VAD 判为非语音的部分全部置 0（掩蔽）
          → OfflineDiarizer（pyannote 分割）得到 turn
          → 每个 turn 按 15 秒切窗提声纹（EMBEDDING_WINDOW_MS = 15_000）
          → SpeakerTracker 在线聚类分配 spk-N
```

`SpeakerTracker` 是典型的在线聚类：

- 余弦相似度 ≥ `online_similarity_threshold` → 并入最近簇，**中心做滑动平均后重新归一化**；
- 否则新建簇；达到 `max_speakers` 后强制并入最近簇；
- 短于 `minimum_embedding_seconds` 的片段返回 `None`，**沿用上一个说话人**（不新开簇）。

对比我们：我们是"整段算一个声纹 + 全局层次聚类"，短段噪声会污染簇；
它在**掩蔽后的音频**上、按 15 秒窗口提声纹，稳定得多。

---

## 4. 其它细节

- **模型管理**：`models.json` 每项带 `runtime`/`revision`/`url`/`china_url`/`archive_sha256`/
  `size_bytes`/`memory_floor_bytes`/`platforms`，下载重试 5 次、超时 30s、可用空间校验 ×2。
- **会后精修**：`RefinedASR` 用整窗音频跑高精度模型；实时链路用同一套切分但线程预算更低。
- **多引擎**：qwen3 / whisper / nemo-transducer（Parakeet）/ fire-red-asr / funasr-nano，按语言和
  质量/速度排名自动选。

---

## 5. 我们的 A/B 实测（同一段音频）

### 5.0 完整 10.6 分钟录音三引擎对比（2026-10-31）

用 `bench/compare_engines.py` 对同一段情绪对话各跑一遍（只跑识别+分离，关掉 LLM 环节）：

| 指标 | Paraformer | SenseVoice | **Fun-ASR-Nano** |
|---|---|---|---|
| 总字数 | 1270 | 1240 | **1841** |
| 段数 | 199 | 166 | 177 |
| 段长中位数 | 1.16s | 1.41s | **1.55s** |
| 逗号 / 句号 | 132 / 18 | 86 / 109 | 140 / 89 |
| 标点异常 | **5** | 14 | 12 |
| 耗时（10.6 分钟音频） | 155s | **104s** | 182s |
| 嗯/呃/啊 | 5/0/15 | 6/0/15 | 13/0/22 |

关键片段对照（00:00–01:00）：

| 引擎 | 输出 |
|---|---|
| Paraformer | `这个路你感觉你是嗯你不是要给我了吗你在？干嘛我有？点难过怀孕不好，玩为啥` ← 碎成词块 |
| SenseVoice | `你感觉抵视嗯，你不是要给我了吗？你在干嘛？我有点难过，怀孕不好玩…听不到胎心` ← 关键短语对 |
| **Nano** | `你感觉你是你在干嘛？我有点难过，怀孕不好玩。为啥不好玩？…听不到，担心那明天去听一下` ← **最像自然对话**，但"担心"是编的 |

05:00–06:00 更能看出差距：

- Paraformer：`吧是我都难，过呀觉得好可这咋又？可怜了放歌，曲他咋就好可怜`（几乎不可读）
- SenseVoice：`对不起。你都在这哭吃啥？是。首下来你就有父爱。`（短句拼接）
- Nano：`不会觉得他对不起我。你总在这哭，吃啥生下来你就有父爱吗？`（**语义连贯**）

**结论**：Nano 在这次对比里**句子连贯性明显最好**（字数多 48% 是因为它把轻声、碎片也识别成了完整句子，而不是丢弃），
代价是口水词更多、偶发"编造连接词"。Paraformer 在这段安静/情绪化音频上碎得最厉害——**这正好解释了你在 Brevia 上感觉更准**。

> 复现：`PYTHONPATH=code .venv-funasr/bin/python bench/compare_engines.py "input/xxx.m4a"`
> 注意：脚本会在每次跑之前清空该音频的工作目录——否则 `work/<id>/asr/w0.json` 缓存会让三个引擎输出完全一样（我们踩过）。

### 5.1 单片段对照

环境：M4 CPU、int8、6 线程、8 秒目标块 + 能量谷交付。

| 片段 | Fun-ASR-Nano | 现有引擎 |
|---|---|---|
| 业务通话 60s | `跟mute了哦，我刚mute掉了…IDC那边转过来的…ICP落地接触，然后IC做到一半` | Paraformer：`跟 mute 掉，我我刚掉掉…mui DC那边…ICT落地接触，然后ICT做到一半` |
| 情绪录音 60s | `你感觉你是你在干嘛？有点难过…听我的，担心那明天去听一下，担心去去社康…去食堂听一下` | SenseVoice：`…你在干嘛？我有点难过…听不到胎心，那明天去听一下…去去社康听一下` |

**怎么读这张表**：

- 业务通话：Nano 的句子更完整（`我刚mute掉了` vs `我我刚掉掉`），但把 **ICT 听成 ICP/IC**（Paraformer 对）；
- 情绪录音：SenseVoice 的 **「听不到胎心」** 是关键语义，Nano 给成「听我的，担心」「去食堂听一下」（错），
  这一句 SenseVoice 明显更好；
- **速度**：Nano int8 CPU RTF **0.13–0.20**；SenseVoice ~0.015（快约 10 倍）；Paraformer ~0.047；
- **标点**：Nano 自带标点（不用 CT-Punc），另两个需要。

所以：**你感觉"Brevia 更准"，很可能主要来自切分与说话人跟踪，而不是模型本身**——
它跑的是 macOS 的 MLX 版（`funasr-nano-mlx`，1.6 GB，Metal），我们这次测的是 CPU int8。

---

## 6. 我们采纳了什么

| 采纳项 | 状态 |
|---|---|
| **Fun-ASR-Nano 第三引擎**（sherpa-onnx，自带标点） | ✅ 已实现 `app/nano.py` + `--asr funasr_nano`，端到端跑通 |
| **8 秒目标 + 能量谷切分 + 22 秒硬上限** | ✅ 已实现（`app/nano.py:split_chunks`） |
| 句末标点决定合并（endpoint 语义） | ✅ 已实现（句末标点就切段，否则并入下一块） |
| 段首 300ms 回补 / 切点重叠 / 已交付水位 | ⏳ 下一步（当前用"首尾相接 + 句末切分"近似） |
| 边界来源标记（标点可信度） | ⏳ 下一步 |
| VAD 掩蔽 + 15s 窗口声纹 | ⏳ 下一步（我们目前整段一个声纹） |
| 安静语音兜底 | ⏳ 下一步 |

## 7. 产品化：已实现自动分流（v0.27.0）

`asr.provider: auto`（默认）在识别前探测**说话人数**（开头 90 秒跑一次分离）与**时长**：

| 条件 | 引擎 |
|---|---|
| ≥20 分钟 / ≥3 人 / 1 人 | `funasr`（会议、讲座） |
| **2 人一对一** | **`funasr_nano`**（对话，句子更连贯） |
| 其余 / 探不到 | `funasr_nano` |
| Nano 缺失 | `sensevoice` |

实测：情绪对话（2 人 12.7 分钟）→ `funasr_nano`；业务片段（1 人 60 秒）→ `funasr`。

> 不用"语音占比"做判据：那段情绪录音的语音占比高达 **98%**，反而更像会议——人数才可靠。

> 别指望"换模型"解决所有问题：这次真正的差距在**切分、掩蔽、说话人跟踪**这三件工程活上。
