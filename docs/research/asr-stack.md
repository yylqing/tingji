> 调研时间 2026-10-30，环境：Apple M4 / 16GB / macOS 26.5.2，HuggingFace 直连不可达

# 本地离线中文语音转写技术栈调研

目标组合：**离线中文转写 + 说话人分离 + 声纹识别（已知说话人注册后识别姓名）+ 本地 LLM 纪要总结**。

---

## 0. 方法与可信度说明

- 版本号、许可、平台支持均取一手接口，不取二手博客：
  - `api.github.com/repos/<owner>/<repo>`（star / license / pushed_at / archived）
  - `api.github.com/repos/<owner>/<repo>/releases/latest`（tag + published_at + release notes）
  - `pypi.org/pypi/<pkg>/json`（version / license / requires_python / upload_time / **wheel 文件名列表**）
  - `www.modelscope.cn/api/v1/models/<model_id>`（`License` 字段，用于核对 FunASR 权重许可）
  - 官方文档站 `k2-fsa.github.io/sherpa/onnx/...`、仓库内 README / 源码头文件
- **HuggingFace 在本机不可达**（`curl` 20s 超时返回 000，`web_fetch` 报 fetch failed）。因此所有依赖 HF 模型卡的信息（pyannote 模型 gated 状态与权重许可、Qwen3-ASR / GLM-ASR 权重许可、Turbo 模型卡）改用 **pyannote 官方 README + WhisperX README 交叉验证**，并在文中显式标注。
- 标注约定：`【实测抓取】`= 接口/文件原文；`【估算】`= 由已知数据推算，未在本机验证；`【经验判断】`= 无直接测量支撑的工程判断。

---

## 1. FunASR（阿里达摩院）

| 项 | 值 |
|---|---|
| 最新版本 | **v1.4.17**，tag 发布 2026-10-28T09:40Z，PyPI 上传 2026-10-28T11:19Z【实测抓取】 |
| 许可 | 工具包 **MIT**；模型权重 **Apache License 2.0**（ModelScope 逐条核对）【实测抓取】 |
| 平台 | 纯 Python wheel（`funasr-1.4.17-py3-none-any.whl`，`requires_python >=3.7`），依赖 torch/torchaudio；官方只给 CPU / CUDA 两条路径，**无 MPS** |
| Star / 活跃度 | 20,887 star，最后 push 2026-10-28【实测抓取】 |

**模型矩阵**（参数与语言取自官方 README Model Zoo 表格【实测抓取】）：

| 模型 | 任务 | 语言 | 参数 |
|---|---|---|---|
| **Fun-ASR-Nano**（Fun-ASR-Nano-2512） | ASR（2025-12 新增，LLM-ASR 主力） | zh/en/ja + 7 大方言区、26 种口音 | 800M |
| Fun-ASR-MLT-Nano | ASR | 31 语言 | 800M |
| **SenseVoiceSmall** | ASR + 情感 + 音频事件 | zh/en/ja/ko/yue | 234M |
| MOSS-Transcribe-Diarize | 第三方（OpenMOSS）：ASR + 时间戳 + 匿名说话人 | — | 见官方卡 |
| **Paraformer-zh** | ASR + 时间戳 | zh/en | 220M |
| Paraformer-zh-streaming | 流式 ASR | zh/en | 220M |
| Qwen3-ASR | ASR，52 语言 | 多语 | 1.7B |
| GLM-ASR-Nano | ASR，17 语言 | 多语 | 1.5B |
| Whisper-large-v3 / -turbo | ASR + 翻译 | 多语 | 1550M / 809M |
| **ct-punc** | 标点 | zh/en | 290M |
| **fsmn-vad** | VAD | zh/en | 0.4M |
| **cam++** | 说话人 embedding（流水线组件） | — | 7.2M |
| emotion2vec+large | 情感 | — | 300M |

**关键结论**

1. **不存在 "SenseVoice-Large"**。官方 model zoo 里 SenseVoice 只有 Small（234M）一个检查点；2025-12 之后的准确率继任者是 Fun-ASR-Nano 800M，而非某个 Large 版 SenseVoice。【实测抓取】
2. **时间戳**：Paraformer 原生输出词级时间戳；SenseVoice 默认 `output_timestamp=False`，开启后走本实现的 CTC 对齐路径。CLI 一条命令即可拿到「分离 + 时间戳 + JSON」：`funasr audio.wav --spk --timestamps -f json`。【实测抓取：[README](https://github.com/modelscope/FunASR)、[python_api.md](https://github.com/modelscope/FunASR/blob/main/docs/python_api.md)】
3. **说话人**：`AutoModel(model="iic/SenseVoiceSmall", vad_model="fsmn-vad", spk_model="cam++")` 返回**带 speaker id 与时间戳的 VAD 段**，CAM++ 抽 embedding 后由 AutoModel 聚类。README 明确：这是**录音内匿名编号，不是注册式声纹识别**，官方建议注册场景另接 speaker verification。【实测抓取】
4. **itn（逆文本正则化）**：SenseVoice 走 `use_itn` / `text_norm` 开关（明确文本归一化会覆盖 itn 设置）；Fun-ASR-Nano 默认 `itn=True`。官方文档给了可复现对比：同一段中文样本，开 ITN 得到 242 个 paired unit，关 ITN 得到 217 个——说明 ITN 影响的是"数字/单位/符号的显示形态"，不提升字准。【实测抓取：[python_api.md](https://github.com/modelscope/FunASR/blob/main/docs/python_api.md)】
   - 若需要独立、更强可控的中文 ITN，用 **WeTextProcessing**（wenet-e2e，**Apache-2.0**，832 star，最后 push 2026-10-30，仓库描述即 "Text Normalization & Inverse Text Normalization"）。【实测抓取】
5. **中英混说（code-switching）**：Paraformer-zh 训练集本身含 zh/en，句中夹英文专有名词/术语可用；SenseVoice 语言提示是**整句级** token（zh/en/yue/ja/ko），混说比例高时会退化。【经验判断，依据 README 语言字段 + 语言 token 机制】
6. **macOS arm64**：`pip install funasr` 可直接装（纯 py wheel），torch 装 arm64 轮子即可在 CPU 跑通；官方安装文档要求"先装 PyTorch + torchaudio 再装 funasr"，无 macOS 专属说明、无 MPS 支持，`device` 只演示 `cpu`/`cuda`。【实测抓取】README 另提供 **llama.cpp GGUF 单文件二进制**（`llama-funasr-sensevoice`，标注 Linux / macOS 可用，内置 FSMN-VAD，宣称中文 CER 约为 whisper.cpp 的 1/3）。【实测抓取】
7. **Apple Silicon 原生加速**：第三方 **funasr-mlx 0.2.3**（2026-10-24，**Apache-2.0**，要求 Python ≥3.11），用 MLX 原生重写，覆盖 14 个检查点 / 11 种架构（SenseVoiceSmall、Paraformer 离线与流式、Whisper tiny/large-v3/turbo、ct-punc、fsmn-vad、CAM++、emotion2vec+large、Fun-ASR-Nano、Qwen3-ASR-0.6B、GLM-ASR-Nano、MOSS-Transcribe-Diarize），推理期不依赖 PyTorch；带 W4A16 量化（M4 上实测峰值内存降 41–59%、解码快至 2.7x）。它的实测 RTF：Whisper large-v3 **0.058**、turbo **0.061**；流式 Paraformer 因 MLX 无 fp64，chunk 状态机放大 fp32 噪声导致 CIF 位置漂移（CER 0.156），**流式只可用于中间结果、终稿须用离线模型**。【实测抓取：[PyPI funasr-mlx](https://pypi.org/project/funasr-mlx/)】

来源：[FunASR repo](https://github.com/modelscope/FunASR) · [v1.4.17 release](https://github.com/modelscope/FunASR/releases/tag/v1.4.17) · [PyPI funasr](https://pypi.org/project/funasr/) · [python_api.md](https://github.com/modelscope/FunASR/blob/main/docs/python_api.md) · [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing) · [funasr-mlx](https://pypi.org/project/funasr-mlx/)

---

## 2. sherpa-onnx（k2-fsa / Next-gen Kaldi）

| 项 | 值 |
|---|---|
| 最新版本 | **v2.0.1**，2026-10-26T15:24Z；PyPI 2.0.1 上传 2026-10-27T01:55Z【实测抓取】 |
| 许可 | **Apache-2.0**【实测抓取】 |
| Star | 15,626，最后 push 2026-10-29【实测抓取】 |
| macOS arm64 分发 | ✅ **官方预编译 wheel**：`macosx_11_0_arm64` 覆盖 cp38/cp39/cp310/cp311/cp312/cp313/cp314/cp315；另有 `universal2`、`x86_64`；release 里还有 `osx-arm64` 静态库/动态库/JNI 二进制包（该 release 共 288 个资产）【实测抓取】 |

**离线说话人分离（diarization）**：官方文档给出的组合是
**pyannote segmentation-3.0**（`sherpa-onnx-pyannote-segmentation-3-0.tar.bz2`）或官方 **sherpa-onnx-reverb-diarization-v1/v2** + **说话人 embedding 模型**。文档逐项列出四种搭配的用法：3D-Speaker + `model.onnx`、3D-Speaker + `model.int8.onnx`、NeMo + `model.onnx`、NeMo + `model.int8.onnx`。→ 即 **pyannote 分割 + 3D-Speaker / NeMo 声纹**，int8 可再省一半内存。【实测抓取：[分离文档](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html)】

**声纹模型库**：`speaker-recongition-models` release 共 **36 个 ONNX 资产**，其中 WeSpeaker 9 个、3D-Speaker/CAM++ 9 个、NeMo 3 个，含
`3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx`、`3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx`（中英双语）、`wespeaker_en_voxceleb_CAM++.onnx`。【实测抓取】→ 注册式声纹（识别姓名）用这批模型 + 余弦相似度即可，无需 PyTorch。

**离线 ASR 覆盖**（文档侧栏模型族）【实测抓取】：Whisper、Moonshine、Omnilingual ASR、**SenseVoice**、Cohere Transcribe、**FunASR Nano**、**Qwen3-ASR**、**Paraformer**、NeMo、FireRedAsr、Dolphin、Google MedASR。SenseVoice 页明确支持平台：Linux(x64/aarch64/arm/riscv64)、**macOS(x64/arm64)**、Windows、Android、iOS。

**时间戳与元信息**：C++ 结果结构 `OfflineRecognitionResult` 含 `tokens`、`timestamps`（`timestamps.size() == tokens.size()`，单位秒）、`lang`、`emotion`、`event`，以及 Whisper 的 `segment_timestamps` / 段文本——**词级时间戳是原生字段，不是后处理**。【实测抓取：[offline-stream.h](https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/offline-stream.h)】

**ITN**：Python API 支持传入 **fst archives 做逆文本正则化**（`offline_recognizer.py` 参数说明）。【实测抓取】

**2.0.1 新增**：VoxLock 目标说话人语音隔离（进入 speech denoisers），并一次性补齐 Python / C / C++ / C# / Go / Rust / Java / Kotlin / Swift / Dart / JS(node + wasm) / Pascal 全语言 API。【实测抓取：[release notes](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v2.0.1)】

来源：[repo](https://github.com/k2-fsa/sherpa-onnx) · [v2.0.1](https://github.com/k2-fsa/sherpa-onnx/releases/tag/v2.0.1) · [分离文档](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/index.html) · [SenseVoice 文档](https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html) · [PyPI](https://pypi.org/project/sherpa-onnx/)

---

## 3. WhisperX / faster-whisper / mlx-whisper / large-v3-turbo

### 3.1 WhisperX
| 项 | 值 |
|---|---|
| 最新稳定版 | **3.8.6**，2026-05-25T12:35Z（另有 3.8.7rc1 预发布，2026-06-26）【实测抓取】 |
| 许可 | **BSD-2-Clause** |
| Star / 维护 | 24,946 star；**最后 push 2026-10-22**，未归档——版本号停在 5 月但仓库仍在修 bug（3.8.5 2026-04-01、3.8.4 2026-03-25、3.7.9 回移补丁） |
| Python | `>=3.10,<3.14` |
| 架构 | faster-whisper 批处理推理 + wav2vec2 强制对齐（词级时间戳）+ pyannote 分离 |

- 分离改用 **`pyannote/speaker-diarization-community-1`**，README 明确要 `--hf_token` 并先接受该模型的 user agreement。【实测抓取】
- **许可交叉验证**：WhisperX README 致谢段原文写明 community-1 "licensed under **CC-BY-4.0** by pyannoteAI"——与 pyannote 官方 README 的 gated 描述一致，可确认**权重是 CC-BY-4.0（允许商用，需署名）**，代码 MIT。【实测抓取】
- 宣称 70x 实时（large-v2，GPU 语境）；`interleaved_context=True` 可保持长上下文以改善标点/术语一致性。【实测抓取】

### 3.2 faster-whisper（CTranslate2）
| 项 | 值 |
|---|---|
| 版本 | **1.2.1**，2025-10-31T11:35Z（PyPI 无更新近一年）【实测抓取】 |
| 许可 | **MIT**；Star 26,322，最后 push 2026-10-25（仓库仍活跃，只是未发版） |
| Python | `>=3.9`，纯 `py3-none-any` wheel（后端为 CTranslate2 二进制） |
| macOS | CPU / arm64 可用（CTranslate2 有 macOS arm64 轮子）；**无 MPS、无 ANE**，GPU 仅 CUDA |

关键结论：M4 上只能吃 CPU（int8 量化），可用但不是最快路径；中文准确率继承 Whisper 上限（见 3.4）。

### 3.3 mlx-whisper / MLX 生态
| 项 | 值 |
|---|---|
| mlx-whisper | **0.4.3**，2025-08-29（PyPI），**MIT**，`requires_python >=3.8` |
| mlx-examples | 9,044 star，**MIT**，最后 push 2026-04-06【实测抓取】 |
| 现状 | Apple GPU 原生（统一内存、无显存拷贝）；生态里另有 funasr-mlx 把 FunASR 全家桶搬到 MLX（见 §1.7），是 M 系列上**唯一有公开实测加速数字**的路线 |

### 3.4 whisper large-v3-turbo
- **许可**：Whisper 代码与权重 **MIT**（openai/whisper 仓库）；turbo 为 809M 参数、约 6GB 显存、约 8x 速度，官方措辞是"minimal degradation in accuracy"，且**未针对翻译训练**，即使指定 `--task translate` 也会返回原语言。【实测抓取：[openai/whisper README](https://github.com/openai/whisper)】
- **中文表现（关键）**：FunASR 官方 184 个长中文音频（11,539 秒）对照基准，**turbo CER 21.71%**、large-v3 **20.02%**、SenseVoice-Small **7.81%**、Fun-ASR-Nano **8.06%**、Paraformer-Large **10.18%**。即 **Whisper 系中文错字率约为 FunASR 系的 2–2.7 倍**，turbo 还略差于 large-v3。【实测抓取：[FunASR Discussion #2947](https://github.com/modelscope/FunASR/discussions/2947)，2026-07-15 修订版，H100 环境】

来源：[WhisperX](https://github.com/m-bain/whisperX) · [releases](https://github.com/m-bain/whisperX/releases) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [mlx-whisper](https://pypi.org/project/mlx-whisper/) · [openai/whisper](https://github.com/openai/whisper) · [中文基准](https://github.com/modelscope/FunASR/discussions/2947)

---

## 4. pyannote.audio

| 项 | 值 |
|---|---|
| 最新版本 | **4.0.7**，2026-06-30T14:29Z（4.0.6 2026-06-29、4.0.5 2026-06-22、4.0.4 2026-02-07、4.0.3 2025-12-07、4.0.2 2025-11-19）【实测抓取】 |
| 许可 | **代码 MIT**；**community-1 权重 CC-BY-4.0**（可商用，需署名 pyannoteAI）【实测抓取：pyannote README + WhisperX README 双向确认】 |
| Python | `>=3.10`；依赖 ffmpeg（torchcodec 解码需要） |
| Star | 10,730，最后 push 2026-10-25 |

**是否 gated**：是。官方 README 的 4 步上手第 2–4 步即「接受 `pyannote/speaker-diarization-community-1` 的 user conditions → 在 hf.co/settings/tokens 建 token → 传 `token=`」。**离线部署必须先在线接受一次协议并把权重缓存到本地**，之后可完全断网。【实测抓取】

**商用限制**：代码 MIT 无限制；权重 CC-BY-4.0 —— 允许商用与再分发，义务是**署名 + 保留许可声明 + 标注修改**。不需要付费、不需要 API key（付费的是云端 precision-2）。【实测抓取】注意 4.x 引入**可选遥测**（`Pipeline.from_pretrained` 上报模型来源、音频时长、num_speakers 等，明确不含可识别个人信息），可关闭。

**准确率（DER %，越低越好；legacy 3.1 / community-1 / precision-2）**【实测抓取】：

| 数据集 | 3.1 | community-1 |
|---|---|---|
| AISHELL-4（中文会议） | 12.2 | **11.7** |
| AliMeeting（中文，ch1） | 24.5 | **20.3** |
| AMI-IHM | 18.8 | 17.0 |
| CALLHOME | 28.5 | 26.7 |
| DIHARD 3 | 21.4 | 20.2 |
| VoxConverse | 11.2 | 11.2 |
| REPERE | 7.9 | 8.9（唯一变差） |

**速度**：官方只给 **H100 80GB** 自托管数字——community-1 处理 AMI 约 **31 秒/小时音频**、DIHARD3 约 **37 秒/小时**；precision-2 为 14 秒/小时。**MPS/CPU 无官方数字**（README 只演示 `.to(torch.device("cuda"))`，支持设备未列 MPS）。→ CPU 上按 H100 的 1/30–1/60 粗估，约 **0.5–1x 实时**（1 小时会议需 1–2 小时）【估算，未实测】。

来源：[repo](https://github.com/pyannote/pyannote-audio) · [community-1 README 段](https://github.com/pyannote/pyannote-audio/blob/develop/README.md) · [PyPI](https://pypi.org/project/pyannote.audio/) · [Changelog](https://www.pyannote.ai/changelog/community-1-now-available)

---

## 5. 3D-Speaker / WeSpeaker / NeMo 的开源状态与许可

| 项目 | Star | 许可 | 最后 push | 备注 |
|---|---|---|---|---|
| **3D-Speaker**（阿里） | 3,203 | **Apache-2.0** | 2025-12-08 | 更新变慢，但 CAM++/ERes2Net 权重托管在 ModelScope，可长期用 |
| **WeSpeaker**（wenet-e2e） | 1,446 | **Apache-2.0** | 2026-09-22 | 训练/推理工具链完整，含 ResNet/CAM++ 等 |
| **NeMo**（NVIDIA-NeMo/Speech） | 18,656 | **Apache-2.0** | 活跃 | TitaNet / Sortformer 等，容器与依赖偏重 |

三家**全部 Apache-2.0，可商用**；sherpa-onnx 已把 3D-Speaker / WeSpeaker / NeMo 的 embedding 转成 ONNX，可完全脱离 PyTorch 部署（见 §2）。【实测抓取：各 repo API + sherpa-onnx release 资产清单】

来源：[3D-Speaker](https://github.com/modelscope/3D-Speaker) · [WeSpeaker](https://github.com/wenet-e2e/wespeaker) · [NeMo Speech](https://github.com/NVIDIA-NeMo/Speech)

---

## 6. 声纹识别（注册后识别姓名）方案对比

指标为 **EER %（越低越好）**，三列依次 VoxCeleb1-O / CN-Celeb / 3D-Speaker 数据集，取自 3D-Speaker 官方 README 表格【实测抓取】：

| 模型 | 参数 | 体积(fp32) | Vox1-O | CN-Celeb | 3D-Speaker |
|---|---|---|---|---|---|
| **CAM++** | **7.2M** | ~28MB | 0.65 | 6.78 | 7.75 |
| ResNet34 | 6.34M | ~25MB | 1.05 | 6.92 | 7.29 |
| ERes2Net-base | 6.61M | ~26MB | 0.84 | 6.69 | 7.21 |
| **ERes2NetV2** | 17.8M | ~71MB | 0.61 | **6.14** | 6.52 |
| ERes2Net-large | 22.46M | ~90MB | **0.52** | 6.17 | **6.34** |
| ECAPA-TDNN | 20.8M | ~83MB | 0.86 | 8.01 | 8.87 |

**结论**
- **中文场景首选 CAM++**：7.2M、28MB、CN-Celeb EER 6.78%，与 3 倍大的 ECAPA 相比明显更好；sherpa-onnx 有现成 ONNX（含 `zh-cn` 与 `zh_en_advanced` 双语版），CPU 单条 embedding 亚 100ms 量级。
- **要更准选 ERes2NetV2**（CN-Celeb 6.14，17.8M/71MB），代价是 2.5 倍算力。
- ECAPA-TDNN 中文最弱，不建议作为中文首选（它的优势在英文 VoxCeleb 生态）。
- 注册式识别落地：入库时对每个说话人取 3–5 条语音算 embedding 取均值 + 归一化，识别时余弦相似度阈值（建议 0.6–0.7，需按数据校准）+ 与 diarization 的匿名簇做匈牙利匹配。【经验判断】

来源：[3D-Speaker README 模型表](https://github.com/modelscope/3D-Speaker) · [sherpa-onnx 声纹模型](https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models)

---

## 7. 中文标点恢复、断句与口语填充词去除

**标点**
- **FunASR CT-Punc**：290M，**Apache-2.0**（ModelScope `iic/punc_ct-transformer_cn-en-common-vocab471067-large`，下载量 3,650 万），中英混排标点，流水线里 `punc_model="ct-punc"` 直接可用；sherpa-onnx 亦有对应 ONNX 版本可脱 PyTorch 跑。【实测抓取】
- **paddleNLP 标点模型**：仓库 12,981 star、**Apache-2.0**、最后 push 2026-05-23；中文标点/断句效果稳，但依赖 PaddlePaddle，在 macOS arm64 上安装体验不如 ONNX 路线。【实测抓取】
- 断句：优先用 VAD 段 + CT-Punc 的句末标点做切分，比固定时长切更自然；字幕场景 FunASR CLI 已内置 `--timestamps` 的字幕切分逻辑（按标点与时长/字数约束）。【实测抓取 + 经验判断】

**口语填充词（嗯/啊/额/那个/就是说）**
- 英文有成熟小工具：**erm**（112 star，MIT，本地 CLI，专删 um/uh/er/erm/ah/hmm 等）、**deep_disfluency**（61 star，MIT，检测模型训练/测试）、**awesome-disfluency-detection**（85 star，论文与数据集索引）。【实测抓取】
- **中文没有找到成熟专用工具**（GitHub 搜索无对口项目；HF 上有 FluentWhisper 之类的 demo space，但本机 HF 不可达，**未确认**其能力与许可）。【实测抓取 + 未确认】
- 工程建议：中文填充词去除交给本地 LLM 后处理（显式给出保留/删除规则 + 保留原 timestamp 对齐），或训一个小的 token 分类器（用 LLM 造标注）；纯规则删"嗯/呃/那个"会误伤"那个方案"这类指代，需要看上下文。【经验判断】

来源：[funasr/ct-punc](https://github.com/modelscope/FunASR) · [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP) · [erm](https://github.com/dougcalobrisi/erm) · [deep_disfluency](https://github.com/clp-research/deep_disfluency) · [awesome-disfluency-detection](https://github.com/pariajm/awesome-disfluency-detection)

---

## 8. 本地 LLM（会议纪要 / 说话人归并 / 纠错）

Ollama 官方库现役中文向模型（下载量与更新时间取自库页面）【实测抓取】：

| 模型 | 规模档位 | 量化后体积 | 备注 |
|---|---|---|---|
| **qwen3.5** | 0.8b / 2b / 4b / 9b / 27b / 35b / 122b | 0.8b 1.2–1.3GB；2b 2.7–3.1GB；4b 3.3–4.0GB；**9b（=latest）6.6–7.6GB** | 256K 上下文，文本+图像输入，23.8M 下载，**2 周前更新** |
| qwen3-vl | 多档 | — | 6.9M 下载，1 周前更新 |
| glm-4.6 | 中大型 | — | 2.2M 下载 |
| deepseek-r1（含蒸馏） | 多档 | — | 94.5M 下载，**1 年未更新** |
| qwen3（旧） | 0.6b 523MB / 1.7b 1.4GB / 4b 2.5GB / 8b 5.2GB | — | 42M 下载，**1 年未更新**，已被 3.5 取代 |

**16GB 能跑多大**：qwen3.5 **4b（3.3–4.0GB）**最稳，**9b（6.6–7.6GB）**可用但需与 ASR 错峰；27b 即使 q4 也接近 16GB 上限，不可与 ASR 同时驻留。【估算，依据体积 + macOS 统一内存占用经验】

**分工建议**：纠错（同音字、专名）用 4b 足够；说话人归并（把匿名 SPEAKER_01 映射到真实姓名）本质是短文本分类，2b–4b 即可；整篇纪要总结用 9b 质量明显更好。【经验判断】

来源：[qwen3.5](https://ollama.com/library/qwen3.5/tags) · [qwen3](https://ollama.com/library/qwen3/tags) · [glm-4.6](https://ollama.com/library/glm-4.6)

---

## 9. 端到端"本地会议转写"参考项目

| 项目 | Star | 许可 | 技术栈 | 分离 | 中英混说 | 全离线 | 结论 |
|---|---|---|---|---|---|---|---|
| **Meetily** | **32,317** | MIT | Tauri + Rust 后端 + Next.js；Whisper / Parakeet；Ollama / Claude / Groq | ❌ **社区版无**（README：分离排在 PRO，mid-June 计划） | 取决模型 | ✅ | 会议录制+转写+纪要体验最完整，但**说话人分离是要付费的 PRO 功能**，开源版拿不到 |
| **Buzz** | 22,286 | MIT | 多 whisper 后端（whisper.cpp 等），Vulkan 加速 | 仅"说话人识别"级 | 取决模型 | ✅ | 桌面易用，工程成熟；中文准确率受 Whisper 上限约束 |
| **Vibe** | 7,963 | MIT | Rust（whisper-rs 系）；Whisper / Nemotron 3.5 / Parakeet TDT v3 | ✅ | 取决模型 | ✅ | 功能最贴近需求（分离 + Ollama 总结 + 稳定时间戳 + CLI），Rust 单机性能好 |
| **Scriberr** | 3,125 | MIT | 自托管 Web；WhisperX 环境 + Parakeet / Canary / Whisper | ✅（pyannote 系） | 取决模型 | ✅ | 自托管/多用户场景首选；部署重（Docker + 独立 Python 环境） |
| **Whisper-WebUI** | 2,900 | Apache-2.0 | Gradio + faster-whisper / Whisper | ✅ | 取决模型 | ✅ | 上手最快，最后 push 2025-12-29，维护放缓 |
| **WhisperX** | 24,946 | BSD-2 | faster-whisper + wav2vec2 对齐 + pyannote | ✅ | 取决模型 | ✅ | 作为**库**最合适（词级时间戳 + 分离），不是成品 GUI |
| MacWhisper | 闭源 | 商业 | Whisper 系 | ✅（付费版） | 取决模型 | ✅ | 非开源，仅作体验参照；版本**未确认** |

**判断**：现成 GUI 全部围绕 Whisper 系模型，"分离 + 离线"能满足，但**中文准确率比 FunASR 系差一个量级**（§3.4）。要同时满足「分离 + 中英混说 + 离线 + 中文准确率」，正解是**自建管线**：sherpa-onnx（SenseVoice/Paraformer + CAM++ 分离）+ 声纹注册 + CT-Punc + 本地 LLM，GUI 只做壳。

来源：[Meetily](https://github.com/Zackriya-Solutions/meetily) · [Buzz](https://github.com/chidiwilliams/buzz) · [Vibe](https://github.com/thewh1teagle/vibe) · [Scriberr](https://github.com/rishikanthc/Scriberr) · [Whisper-WebUI](https://github.com/jhj0517/Whisper-WebUI)

---

## 10. 版本不确定项（未从抓取内容确认，禁止据此写死）

| 项 | 状态 | 原因 |
|---|---|---|
| **SenseVoice-Large** | **不存在**（不是"未确认"，是确认没有） | 官方 Model Zoo 只有 SenseVoiceSmall 234M |
| pyannote community-1 在 **CPU / MPS** 的实测 RTF | 未确认 | 官方仅公布 H100 数字；MPS 未列入支持设备 |
| pyannote 4.x 是否能在 MPS 上完整跑通 | 未确认 | README 只演示 cuda，无 Apple 相关说明 |
| **CT-Punc 在 M4 上的实测延迟** | 未确认 | 无公开 benchmark |
| **funasr-mlx 生产可用性** | 存疑 | 0.2.x、2026-10 新项目；仅 7 个语音族跑通 CLI，4 个 LLM-ASR 族的 pipeline 注册"pending"；流式 Paraformer 已知 CER 0.156 |
| **Scriberr 最新 release 号** | 未确认 | 只取到 repo 元数据（3,125 star / MIT / push 2026-09-20），未逐条列 release |
| **MacWhisper 版本号** | 未确认 | 闭源，无 API 可取 |
| **Qwen3-ASR / GLM-ASR-Nano 权重许可** | 未确认 | 权重在 HF，本机不可达；FunASR README 只写"各模型许可独立于工具包 MIT" |
| **FunASR 基准所用 CPU 型号** | 未确认 | Discussion #2947 只写 "CPU"，未给型号，故 17.2x 不能直接外推到 M4 |
| **CTranslate2 最新版本** | 未确认 | 本轮未查（faster-whisper 1.2.1 锁定的是其兼容区间） |
| **FluentWhisper（中文填充词）** | 未确认 | 仅搜索结果，HF 不可达，未见许可与效果数据 |

---

## 11. 对 16GB M4 的适配建议

### 11.1 各组件在本机的可用性

| 组件 | CPU | MPS/GPU | 结论 |
|---|---|---|---|
| sherpa-onnx（SenseVoice/Paraformer + CAM++ + VAD + 标点，ONNX int8） | ✅ 官方 arm64 wheel | 不需要 | **首选**，全 CPU 即可，无 PyTorch 依赖 |
| FunASR（PyTorch） | ✅ 可用 | ❌ MPS 非官方路径 | 可跑，但比 ONNX 慢、内存高 |
| funasr-mlx / mlx-whisper | — | ✅ Apple GPU 原生 | M4 上唯一有公开实测加速的路线（large-v3 RTF 0.058） |
| faster-whisper | ✅（int8） | ❌ 无 MPS | 可用，中文弱 |
| pyannote.audio | ✅ 慢 | ⚠️ MPS 非官方 | 能跑，长会议耗时显著 |
| Ollama（qwen3.5） | ✅ | ✅ Metal | 4b–9b 可用 |

### 11.2 内存与速度量级

- **常驻内存**【估算】：sherpa-onnx 全链路（SenseVoice int8 ≈250MB + FSMN-VAD 0.4M + CAM++ int8 ≈7MB + CT-Punc int8 ≈300MB）约 **1–2GB**；FunASR(PyTorch) 同链路约 **3–5GB**（torch 运行时开销大）；pyannote community-1 单独约 **2–3GB**。
- **速度**：
  - 官方 CPU 基准（型号未标注）：SenseVoice **17.2x 实时**、Paraformer **15.6x**、Fun-ASR-Nano **3.6x**。【实测抓取，不可直接外推到 M4】
  - M4 上按同工作流 **3–8x 实时**估算（1 小时会议 ≈ 8–20 分钟）。【估算】
  - MLX 路线实测：Whisper large-v3 **RTF 0.058（≈17x）**、turbo **0.061**；W4A16 量化在 M4 上降峰值内存 41–59%。【实测抓取：funasr-mlx】
  - pyannote CPU 分离约 **0.5–1x 实时**（1 小时会议 1–2 小时）。【估算，无官方数据】

### 11.3 推荐链路（按优先级）

**方案 A：准确率优先（推荐）**
```
音频 → sherpa-onnx FSMN-VAD 切段
     → SenseVoice int8（中文 CER 7.81%）或 Fun-ASR-Nano（8.06%，更慢）
     → CAM++ (zh_en_advanced) embedding + 聚类 = 匿名说话人
     → 声纹库余弦匹配 = 真实姓名（注册 3–5 条/人）
     → CT-Punc 加标点 → qwen3.5:4b/9b 纠错 + 纪要
```
内存峰值约 2GB（ASR 阶段）+ 5GB（LLM 阶段），**错峰加载即可稳在 16GB 内**。全链路离线，无 HF token 依赖。

**方案 B：Apple GPU 加速**
用 funasr-mlx 跑 SenseVoice/Paraformer + ct-punc（Apache-2.0，MLX 原生），GPU 空闲时 RTF 更优；风险是新项目（0.2.3）、流式不可用，需锁版本并自建回归测试。

**方案 C：多语/英文为主**
mlx-whisper（large-v3-turbo）或 faster-whisper + pyannote community-1（需先联网接受协议、缓存权重，CC-BY-4.0 需署名）。中文场景不推荐做主链路——CER 约为方案 A 的 2.7 倍。

### 11.4 明确不要做的事

- 不要指望 MPS 加速 pyannote 或 FunASR：两者都不在其官方支持设备列表。
- 不要在 16GB 上同时驻留 9b LLM 与 pyannote + Whisper 全链路（峰值易触发 swap，速度断崖）。
- 不要用 Whisper 系模型作为中文会议主 ASR——除非能接受 20%+ 的字符错误率。
- 不要用 SenseVoice 的 `spk_model` 输出当"声纹识别"：它只是录音内匿名聚类，跨会议不通用。

---

## 12. 一页速查

| 需求 | 选型 | 版本 | 许可 |
|---|---|---|---|
| 中文 ASR | SenseVoice-Small（234M）/ Fun-ASR-Nano（800M） | FunASR 1.4.17 | Apache-2.0 |
| 部署运行时 | sherpa-onnx | 2.0.1（macOS arm64 wheel 齐全） | Apache-2.0 |
| 说话人分离 | pyannote segmentation-3.0 + CAM++/NeMo（ONNX） | 随 sherpa-onnx | Apache-2.0 |
| 声纹识别 | CAM++（7.2M）/ ERes2NetV2（17.8M） | 3D-Speaker 权重 | Apache-2.0 |
| 标点 | CT-Punc（290M） | FunASR 1.4.17 | Apache-2.0 |
| ITN | SenseVoice 内建 / WeTextProcessing | — | Apache-2.0 |
| 纪要 LLM | qwen3.5:4b / 9b | Ollama 现役 | 见模型卡 |
| 参考成品 | Vibe / Scriberr / Buzz | — | MIT |
