# 实测基准（可复现）

本目录是 [技术选型报告](../docs/01-技术选型报告.md) §2.4 的全部依据：脚本 + 原始结果 + 测试音频生成方式。

> 测试环境：Apple M4（10 核 CPU / 10 核 GPU）、16GB 统一内存、macOS 26.5.2、ffmpeg 8.1.2。
> 测试时间：2026-10-30。

---

## 1. 测试音频如何生成

无真实会议录音（隐私），用 macOS 内置 TTS 合成一段 39.57s 的中文"会议"，4 段发言、3 个可区分说话人，并故意包含英文术语 `ClickHouse`。

```bash
say -v Tingting -r 190 -o s1.aiff "各位同事，我们今天的周会主要讨论三件事……"
say -v Meijia   -r 180 -o s2.aiff "我先说一下交付的问题……"
say -v Tingting -r 205 -o s3.aiff "我补充一下，数据平台这边我们已经用 ClickHouse 做了压测……"
# 第三个说话人用音高变换制造
ffmpeg -i s3.aiff -af "asetrate=44100*1.12,aresample=44100,atempo=1/1.12" s3_shift.aiff
say -v Meijia -r 185 -o s4.aiff "那这样，交付这块你出一个风险清单……"
```

⚠️ **局限**：TTS 合成、无噪声、无重叠说话、段间无停顿 → 只能测**吞吐与可行性**，不能测**准确率**。

---

## 2. 脚本

| 脚本 | 用途 | 依赖 |
|---|---|---|
| `bench_funasr.py` | FunASR 全链路（ASR+VAD+标点+CAM++ 分离） | `.venv-funasr` |
| `bench_mlx.py` | mlx-whisper large-v3-turbo 对照 | `.venv-mlx` |
| `bench_sherpa.py` | sherpa-onnx 离线分离 | `.venv-funasr`（含 sherpa-onnx） |
| `bench_voxlock.py` | 目标说话人提取（VoxLock） | `.venv-funasr` + voxlock 模型 |
| `overlap_probe2.py` | 重叠检测特征（1s 窗一致性） | `.venv-funasr` |
| `phoneme_coverage.py` | 登记文本音素覆盖率 | `.venv-funasr`（含 pypinyin） |
| `asr_only.py` | 纯 ASR（对比提取前后文本） | `.venv-funasr` |
| `bench_moss.py` | MOSS-Transcribe-Diarize 0.9B 基准 | `.venv-funasr` + 1.82GB 权重 |
| `bench_hotword.py` | 热词三层对比（基线/模型级/文本级） | `.venv-funasr`（含 rapidfuzz+pypinyin） |

```bash
export MODELSCOPE_CACHE="$PWD/.cache/modelscope"
.venv-funasr/bin/python bench/bench_funasr.py bench/in/meeting_sample.wav full

export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
.venv-mlx/bin/python bench/bench_mlx.py bench/in/meeting_sample.wav

.venv-funasr/bin/python bench/bench_sherpa.py bench/in/meeting_sample.wav
```

---

## 3. 结果

### 3.1 FunASR 全链路（主选方案）

```json
{"audio_sec": 39.57, "model_load_sec": 14.12, "infer_sec": 1.86,
 "rtf": 0.047, "speed_x_realtime": 21.25, "peak_rss_mb": 4352.6,
 "num_sentences": 19, "speakers_detected": [0, 1, 2]}
```

| 指标 | 值 |
|---|---|
| 音频时长 | 39.57s |
| 模型加载（热缓存） | 14.12s（4 个模型：ASR 953MB + 标点 1.1GB + VAD + CAM++） |
| 推理耗时 | **1.86s** |
| RTF | **0.047** |
| 实时倍速 | **21.25×** |
| 峰值 RSS | **4.35GB** |
| 识别句子 | 19 句 |
| 说话人 | 3 人（与设计一致） |

**输出片段**（注意 `ClickHouse` 被误识别）：

```
[  0.05-  0.89] S0: 各位同事，
[ 11.67- 13.69] S1: 我先说一下交付的问题。
[ 22.67- 24.47] S2: 我补充一下数据平台这边我们已经认批，
[ 24.47- 25.43] S2: 他做了标准查询，
[ 26.45- 28.55] S2: 但是我担心经营期间的收息会带来一致性问题。
```

### 3.2 mlx-whisper large-v3-turbo（对照，M4 GPU）

```json
{"audio_sec": 39.57, "infer_sec": 12.16, "rtf": 0.307,
 "speed_x_realtime": 3.25, "peak_rss_mb": 1757.2}
```

**输出片段**：

```
数据平台这边我们已经用PASPRO来压着，查形性能提升了大概三倍，
但是我当心践移期间的收鞋会带来一句新问题
```

→ 比 FunASR **慢 6.5 倍**，中文错误更多，还出现替换字符 `�`，分段粒度粗（30s 一段）。

### 3.3 sherpa-onnx 离线分离（pyannote-seg3.0 + CAM++ zh_en）

```json
{"audio_sec": 39.57, "model_load_sec": 0.17, "diarize_sec": 1.26,
 "rtf": 0.032, "speed_x_realtime": 31.35, "peak_rss_mb": 223.4, "num_speakers": 2}
```

| 聚类阈值 | 检出人数 | 说明 |
|---|---|---|
| 0.5 / 0.6 / 0.7 / 0.8 | 2 | 欠分割：把 0–22.7s 判成同一人 |
| 0.9 | 1 | 过度合并 |

→ 该样本上欠分割（可能因段间无停顿）。**仅说明"能跑、很快、很省内存"，不能据此判定分离精度**。

### 3.4 Ollama qwen3.5:9b-q4_K_M 纪要生成

| 指标 | 值 |
|---|---|
| 冷启动加载 | 5.36s |
| prompt tokens | 528–530 |
| 输出 tokens | 288 |
| 生成耗时 | 18.8s |
| 速度 | **15.5 tok/s** |
| 输出 | 合法 JSON（topic/summary/decisions/todos/risks/keywords） |

**关键发现**：默认（`think` 未关）时，700 token 全被"思考"吃掉，`response` 返回**空字符串**。必须 `"think": false`。

### 3.5 重叠说话（23.06s，两人同时说）

构造：轨道 A（张三 0–11.6s + 李四 12–23s）与轨道 B（李四 4–15s）时间轴叠加。

```json
{"audio_sec": 23.06, "infer_sec": 1.34, "rtf": 0.058, "peak_rss_mb": 4071.4}
```

| 真值 | 输出 |
|---|---|
| 4.1–11.9s 张+李同时说 | `S1: 第一是那这一季度的交付结在你出一个风险清单，` ← 串词 |
| 8.5–11.9s 同上 | `S1: 周五之前给我数据切的就先做灰度。` ← 串词 |
| 12.0–14.9s 同上 | `S1: 我先说一下前我下的问题跟人事对齐，` ← 串词 |
| 14.9s 之后（重叠结束） | 恢复正常，S2 正确 |

**一方主导（干扰 −10dB）时 ASR 输出完全正确** —— 说明只有音量接近的重叠才会毁掉转写。

### 3.6 目标说话人提取 VoxLock

| 输入 | 结果 |
|---|---|
| 官方中文样例 32.47s（双人混合） | 5.56s（5.8× 实时），峰值 388MB；文本比原混合干净，但仍有另一方内容 |
| 本机重叠样本 23.06s，提取张三 | 3.61s（6.4× 实时）；仍出现李四的"拟出一个风险平台"，并新增错误"救习计划" |
| 本机重叠样本，提取李四 | 仍出现张三的"各位同事，我们今天的周会…" |

### 3.7 重叠检测特征（1 秒窗段内一致性）

| 数据 | 一致性范围 | 均值 |
|---|---|---|
| 无重叠基线（19 段） | 0.657 – 0.872 | 0.764 |
| 重叠音频中的混合段（3 段） | 0.554 / 0.567 / 0.626 | 0.582 |
| 重叠音频中的非混合段（4 段） | 0.773 – 0.875 | 0.826 |

阈值 0.62 在这 22 段上完全分开（0 漏报 0 误报），但间隔仅 0.031，**必须在真实数据上重标定**。

**已排除的信号**：字/秒——重叠时 ASR 是"丢词"而非"混词"（混合段 5.07 字/秒 vs 基线 5.57），无区分度。

### 3.8 登记文本音素覆盖

```
句数=3  汉字数=129  预估朗读时长≈30.7s
声母 21/21   缺: 无
韵母 35/36   缺: ueng
声调 5/5      缺: 无
综合覆盖率 = 99.1%
```

### 3.9 MOSS-Transcribe-Diarize 0.9B（transformers / CPU fp32）

| 音频 | 时长 | 生成耗时 | RTF | 实时倍速 | 峰值 RSS |
|---|---|---|---|---|---|
| 冒烟片段 | 8.0s | 4.86s | 0.608 | 1.65× | 4.70GB |
| 3 人无重叠 | 39.57s | 18.57 / 20.55s | 0.47–0.52 | **1.93–2.13×** | 5.66–5.9GB |
| 重叠样本 | 23.06s | 12.14 / 13.51s | 0.53–0.59 | 1.71–1.90× | 5.66GB |
| 重叠窗口 | 13.0s | 8.86s | 0.682 | 1.47× | — |
| 3 人（MPS） | 39.57s | 25.21s | 0.637 | 1.57×（**比 CPU 慢**） | 5.66GB |

**质量（热词：`ClickHouse, 联调, 双写, 灰度, 招聘`）**：

```
[S01] 各位同事…第二是数据平台的迁移方案…第三是下个月的招聘计划。
[S02] 我先说一下交付的问题…主要是接口联调卡住了…
[S01] 我补充一下，数据平台这边我们已经用 ClickHouse 做了压测，查询性能提升了大概三倍。
      但是我担心迁移期间的双写会带来一致性问题。
[S02] 好，那这样，交付这块拟出一个风险清单，周五之前给我。数据迁移就先做灰度…
```

→ `迁移/ClickHouse/联调/双写` 全部正确（FunASR 全错）。

**重叠样本**：两人内容能分开（S01 段无对方内容），但时间戳乱序（`[11.81]` 接 `[3.98]`）。

**两个反直觉发现**：
1. MPS 比 CPU 慢（小模型逐 token 自回归，kernel launch 开销占主导）。
2. 内存 5.7GB，远超 0.9B 的直觉——音频特征 + KV cache 是大头（transcribe.cpp 估算每音频分钟 +85MB）。

### 3.10 热词三层效果（9.33s，含 6 个专名）

音频真值：`沈知微 / 柯基 / 豆豆 / 杭州西溪湿地 / 河坊街 / 半亩方塘`

| 层 | 输出 | 专名命中 | 耗时 |
|---|---|---|---|
| 基线 | 沈志威…**和房街**… | 4/6 | 0.43s |
| L1 `hotword` | **沈知微**…和坊街… | 5/6 | 0.48s |
| L2 `postprocess_hotwords` | **沈知微**…**河坊街**… | **6/6** | 0.49s |

**matcher 语义验证**（不依赖模型）：

```
输入: 沈志威带着豆豆去了和房街。
输出: 沈知微带着豆豆去了河坊街。
修正: 沈志威 -> 沈知微 (1.00) / 和房街 -> 河坊街 (1.00)
```

→ 拼音（**不区分声调**）相同即命中 1.0，用户只登记标准写法就能把同音错字拉回来。

MOSS 对照（同音频）：无热词 `沈志威…他家的`；加 prompt 热词后 `沈知微…她家的`（6/6），耗时 5.79s（1.61× 实时）。

---

## 4. 结论

1. **可行性**：这台 M4 跑完整链路余量充足（21× 实时、4.35GB）。
2. **ASR 选型**：中文场景 FunASR 明显优于 Whisper 系（速度 6.5×、质量更好）。
3. **必须加纠错层**：ASR 对英文术语与同音字不可靠，LLM 总结层不会自动纠正。
4. **LLM 有坑**：思考模式会静默吞掉输出。
5. **分离需真实验证**：合成样本不足以评判 DER，必须在真实会议录音上做基准。

---

## 5. 产物清单

| 文件 | 说明 |
|---|---|
| `audio/` | 分段 TTS 音频（aiff/wav） |
| `in/meeting_sample.wav` | 拼接后的测试音频（39.57s / 16k mono） |
| `result_full.json` | FunASR 完整结果（19 句，含时间戳与说话人） |
| `result_mlx.json` | Whisper 对照结果 |
| `result_sherpa.json` | 分离结果 |
| `result_overlap.json` | 重叠音频的 ASR 结果（串词证据） |
| `in/overlap_sample.wav` | 重叠测试音频（23.06s） |
| `in/overlap_dominant.wav` | 一方主导（−10dB 干扰）对照 |
| `audio/vox_*.wav` | VoxLock 提取输出 |
| `result_moss_*.json` | MOSS 各样本的完整输出（含原始 `[start][Sxx]text[end]`） |
| `audio/overlap_win.wav` | 重叠窗口（3–16s），用于窗口重转写测试 |
| `in/hotword_sample.wav` | 热词测试音频（含 6 个专名） |
| `result_hotword_*.json` | 热词三层对比结果 |

> 这些目录（`.venv-*`、`.cache/`、`bench/audio/`）可安全删除以回收约 8GB；保留 `bench/*.py`、`bench/result_*.json`、`bench/in/` 即可复现。
