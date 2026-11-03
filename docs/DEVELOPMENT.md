# 听记 TingJi · 开发说明

本地离线中文会议转写与纪要系统。**设计文档见 [docs/]()，本文件只讲怎么跑。**

当前版本：`0.33.0`（逐字稿/清洗稿支持按时间拆分段落；标签页可深链）

---

## 0. 目录结构（v0.28.0 起）

```
project/
├── code/             源码
│   ├── app/          程序
│   ├── bench/        基准与对比脚本
│   ├── tests/        单元测试
│   ├── scripts/      安装脚本
│   └── config/       config.yaml / rules.txt / access_token.txt
├── docs/             文档（本文件、选型、PRD、调研、模板）
├── input/            待处理音频
└── output/           产出 + .runtime/{data,work,logs,voiceprint}
```

- 代码里 `CODE_ROOT` 指 `code/`，`ROOT` 指项目根；配置里的相对路径按**项目根**解析
  （如 `data_dir: ./output/.runtime/data`、模型路径 `.cache/...`）。
- 版本号在 `code/app/__init__.py`，Web 页头与 `/settings` 显示，`/api/version` 也可查。
- `.cache/ .pythons/ .venv-*/` 是隐藏的环境目录，保留在根（移动会破坏 venv/缓存）。

## 1. 环境

| 项 | 要求 |
|---|---|
| 机器 | Apple Silicon（本项目在 M4 / 16GB / macOS 26.5 验证） |
| Python | **3.12**（系统 3.14 装不了 torch） |
| 依赖 | ffmpeg、Ollama（可选，仅纠错用） |
| 模型 | FunASR（ModelScope，首次自动下载）、CAM++ ONNX |

### 安装

```bash
export UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.pythons"
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

uv python install 3.12
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e .

# 运行期依赖（本项目仓库里已装好 .venv-funasr，可直接用）
uv pip install --python .venv-funasr/bin/python fastapi uvicorn python-multipart watchdog
```

### 模型

- **FunASR**：首次运行自动从 ModelScope 下载到 `.cache/modelscope`（约 2.1GB）。
  必须设置 `MODELSCOPE_CACHE` 到可写目录（`../code/app/settings.py` 已默认指向项目内）。
- **CAM++ ONNX**：放在 `.cache/sherpa/campplus.onnx`（27MB）。
- **Ollama**（可选，`correction.enabled: true` 时才需要）：`qwen3.5:9b-q4_K_M`。

自检：

```bash
PYTHONPATH=code .venv-funasr/bin/python -m app.cli selfcheck
```

---

## 2. 快速开始

### 处理一个文件

```bash
# 自动判断说话人数
PYTHONPATH=code .venv-funasr/bin/python -m app.cli run 会议录音.aac

# 已知人数时更稳（推荐）
PYTHONPATH=code .venv-funasr/bin/python -m app.cli run 会议录音.aac --speakers 4

# 强制重跑（忽略 sha256 去重）
PYTHONPATH=code .venv-funasr/bin/python -m app.cli run 会议录音.aac --force
```

产物在 `output/<文件名>/`：

```
transcript.md      # 按说话人分组、带时间戳的逐字稿
transcript.json    # 结构化（speaker/start_ms/end_ms/text/raw_text/quality/overlap）
preview.m4a        # 转码后的音频（供播放器使用）
```

中间产物在 `work/<job_id>/`（16k.wav、分窗 wav）。

### Web 后台

```bash
PYTHONPATH=code .venv-funasr/bin/python -m app.cli serve --port 8000 --with-worker
# 默认 bind=0.0.0.0（局域网可访问）；首次启动自动生成访问令牌并写入
# config/access_token.txt，终端会打印本机与局域网 URL
# 只本机用：--host 127.0.0.1（此时不校验令牌）
# 打开 http://127.0.0.1:8000
```

- `/` 任务列表（3 秒轮询）+ 上传
- `/job/{id}` 逐字稿，点段落跳转音频
- `/hotwords` 热词录入（单个/批量导入）

### Obsidian 归档

在 `config/config.yaml` 里设置 `obsidian.vault_path`（留空则只写 `output/`）：

```yaml
obsidian:
  vault_path: "/Users/you/Documents/MyVault"
  notes_dir: 会议纪要
  transcript_dir: 转写稿
  attachment_dir: attachments/audio
```

写出结构：

```
<vault>/
├── 会议纪要/2026/2026-10-31 0716 周会.md        # frontmatter + 概览 + 说话人表 + 音频嵌入
├── 转写稿/2026/2026-10-31 0716 周会-逐字稿.md   # 按说话人分块
└── attachments/audio/2026-10-31 0716 周会.{m4a,srt,txt}
```

- 原子写（临时文件 + rename），`job_id` 幂等，重跑覆盖同一篇；
- 未命名说话人写成纯文本（不制造 `[[人物/说话人1]]` 死链），已命名写成人物双链；
- 标签：`会议纪要 / 来源/本地录音 / 2026/10 / 时长/标准 / 说话人/张三`（≤12 个）。

### 声纹库

```bash
# 网页：http://127.0.0.1:8000/voices，上传 10–30 秒单人语音
# 或 API：
curl -X POST http://127.0.0.1:8000/api/voices -F "name=张三" -F "file=@zhangsan.wav"
```

注册后，下次转写会自动把匹配的说话人簇命名为"张三"（阈值 0.72 + margin 0.05），
逐字稿、Obsidian 笔记、`speakers.json` 都会带上名字。

### OpenAI 兼容 API

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@meeting.m4a" -F "model=tingji-asr" -F "response_format=verbose_json"
```

| 能力 | 说明 |
|---|---|
| `GET /v1/models` | `tingji-asr` / `tingji-asr-hq`（带说话人）/ `whisper-1` 别名 |
| `POST /v1/audio/transcriptions` | `json`/`text`/`srt`/`vtt`/`verbose_json` |
| 同步 | 音频 ≤30 分钟（`sync_max_audio_sec`），实测 8 秒音频 20 秒 |
| 异步 | >30 分钟返回 **202 + Location**，`GET /v1/audio/transcriptions/{id}` 查询 |
| 鉴权 | 设了 `openai_api.api_key` 才校验 Bearer；绑非回环地址时**强制要求**设 Key |

### 队列模式

```bash
# 终端 A：消费队列
PYTHONPATH=code .venv-funasr/bin/python -m app.cli worker
# 终端 B：把文件放入队列 / 扫描 input/
PYTHONPATH=code .venv-funasr/bin/python -m app.cli ingest ./会议录音.aac
PYTHONPATH=code .venv-funasr/bin/python -m app.cli scan
```

---

## 3. 热词

```bash
# 单个
python -m app.cli hotword add "SaaS" "sars,sas"
python -m app.cli hotword add "沈知微" "沈志威"

# 批量（YAML）
python -m app.cli hotword import hotwords.yaml
python -m app.cli hotword list
```

`hotwords.yaml` 格式：

```yaml
- term: SaaS
  type: english
  aliases: [sars, sas]
  weight: 1.5
- term: 河坊街
  type: place
  aliases: [和房街, 和坊街]
```

热词会同时用于：① ASR 解码偏置（`hotword` 参数，取权重最高的 50 条）；② 解码后拼音模糊替换（`postprocess_hotwords`，阈值 0.85）；③ 纠错 prompt 术语表。

---

## 4. 配置

`config/config.yaml`（完整示例见 `config/config.example.yaml`）：

| 段 | 关键项 | 说明 |
|---|---|---|
| `asr` | `window_speech_sec` | 每个 ASR 窗口的累计语音时长（默认 1500s） |
| `diarization` | `max_speakers` | **0=自动**；已知人数时填 2–10 更稳 |
| | `merge_threshold` | 簇合并阈值（默认 0.85） |
| | `overlap_floor` | 重叠检测绝对下限（默认 0.45） |
| `hotwords` | `l1_max` / `l2_threshold` | 模型级条数上限 / 文本级阈值 |
| `correction` | `enabled` | **默认关闭**；开启后调 Ollama 纠错（约 +9 分钟/小时音频） |
| `server` | `bind` | 默认 `127.0.0.1` |

---

## 5. 处理流程

```
ingest → normalize → plan(分窗) → transcribe(逐窗) → diarize(聚类) → identify(声纹)
      → clean(规则+ITN) → hotword(L2) → correct(可选逐句) → refine(段落顺滑)
      → roles(角色消歧) → summarize(纪要) → archive(输出+Obsidian)
```

**双轨输出**（对齐商业产品做法）：

| 轨 | 内容 | 时间精度 | 用途 |
|---|---|---|---|
| 逐字稿 `transcript.md/json` | ASR 原文 + 规则清理，**保留填充词** | 逐句 | 溯源、校对、引用 |
| 整理稿 `refined.md/json` | 段落化 + LLM 顺滑（去口水词、重断句、修术语） | 段落 | 阅读、分享 |
| 纪要 `summary`（DB/Obsidian/Web） | 摘要 + 章节 + 决议 + 待办 + 关键词 | 章节 | 直接发人 |

**6 分钟真实片段实测**

| 指标 | 逐字稿 | 整理稿 |
|---|---|---|
| 字数 | 2,180 | 1,586 |
| 逗号密度 | 4.4/百字 | **3.4/百字** |
| 单字被标点包夹 | 14 处 | **2 处** |
| 嗯 / 呃 / 那个 | 14 / 22 / 19 | **0 / 0 / 6** |
| LLM 润色覆盖率 | — | **22/23 段（96%）** |

> 剩余的"那个"多为指示代词（"那个方案"），按设计保留；2 处标点异常来自唯一一段未润色的回退段落。

- 每个阶段更新 `jobs.stage/progress`，Web 可见；
- **断点续跑**：`work/<id>/state.json` 记录已完成阶段；ASR 逐窗缓存 `asr/wN.json`、说话人缓存 `diarize.json`，重跑命中缓存（实测 37s → 4s）；
- **并发锁**：同一 job 目录加 `flock`，防止两个进程同时写坏中间文件（压测时踩过）；
- 低质量段（>10s 且 <1 字/秒）标 `low_quality`，不进正文统计；
- 重叠嫌疑标 `overlap`，在 md 里显示 ⚠️。

---

## 5.0 说话人校对（人工弱标注 + 准确率）

在此之前，所有聚类参数（F0 阈值、簇合并、平滑）**都没有真值可校准**。这一步用最少的人工投入
换一个可复现的数字。

**流程**（Web 详情页 →「校对说话人」）：

1. 按簇分层抽样：每簇取 2–3 段"内容较多、时长 ≥1.2s"的片段，随机打乱；
2. 逐条播放 5 秒，人工点选 `说话人1 / 说话人2 / 导航噪声 / 听不清 / 跳过`；
3. 提交后服务端做**最优映射**：每个自动簇映射到出现最多的人工标签（优先未占用的标签），
   再算整体准确率、每簇纯度、混淆矩阵。

接口：

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/api/jobs/{id}/labelqueue?per_cluster=3` | 抽样队列 |
| POST | `/api/jobs/{id}/labels` | 提交 `{items:[{idx,label}]}`，返回准确率 |
| GET | `/api/jobs/{id}/accuracy` | 当前准确率与映射 |
| DELETE | `/api/jobs/{id}/labels` | 清空重标 |

**为什么按簇抽样而不是随机抽样**：随机抽样会几乎全落在最大簇里，小簇永远验不到；
分层抽样能同时暴露"簇内混人"（纯度低）和"一人多簇"（两个簇映射到同一标签）。

## 5.0.1 帧级说话人分离（解决"一段两人"）

人工校对时发现：**一个 ASR 段里有两个人在说话**。实测 121 段中有 **17 段**跨两人（次说话人 ≥0.5s），
而之前按整段算一个声纹的做法无法发现。

试过但**失败**的方案（都记在这里，省得以后重复踩）：

| 方案 | 结果 |
|---|---|
| 段内前后半段声纹比对 | 0 命中——0.4s 窗的 CAM++ 向量在这段音频上相似度中位数只有 **0.19**，全是噪声 |
| 段内滑窗归类找切换点 | 0 命中——同上，窗太短 |
| 段内 F0 轨迹找跳变 | 不可用——F0 在 94/232/98/229 之间乱跳，是倍频错误 |

**可行方案**：sherpa-onnx 的 **pyannote 分割模型**（帧级），本机已有
`.cache/sherpa/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx`（1.5MB）。

- 实测：12.7 分钟音频 **37 秒**跑完（RTF 0.05），170 个 turn，2 人；
- 用它把 ASR 段在换人处切开，并给每段记 **`spk_purity`**（该段被主导说话人覆盖的比例）；
- 结果：121 段 → **166 段**，跨两人段 **17 → 7**；纯度 ≥0.8 的段占 77%。

配置：`diarization.backend: sherpa | embedding`（默认 sherpa；模型缺失时自动回退）。

**代价**：切分后段落更碎（166 段 → 段落数明显增加），可读性不如以前"一段一人"的粗粒度。
取舍是**先保证归属正确**，再靠段落合并/润色补可读性。

## 5.1 说话人标签平滑与跨说话人重切

借鉴 VoiceStudio `segmentation.py` 的 `_fill_and_smooth` / `_resplit_core`，但我们只有句级
时间戳，做了两处适配：

### 标签平滑（`segfix.smooth_labels`）

- **只平滑短段**（默认 <2.5s）：聚类噪声几乎都出在短段上；长段是可靠判断，
  A-B-A-B 的快速对白不能被抹平（第一版没加这条，把真实对白也压平了）。
- 规则：孤立 A-B-A 翻转改回 A；`smooth_min_run` 控制长度 ≤N 的孤立 run 是否合并。
- 实测：这段录音 80/41 → **87/34**。

### 跨说话人重切（`segfix.resplit_by_speaker`）

一句里两个人时按切点拆成两段。**不依赖静音**（拼接处往往没有停顿）：

1. 1.2s 窗 / 0.4s 跳，逐窗算嵌入并归到最近簇；
2. 枚举切点 k，要求左侧主导 A、右侧主导 B（A≠B）且两侧纯度 ≥0.7；
3. **按纯度打分**取最优（第一版按相似度打分，切点会偏到 7.2s；改用纯度后精确命中 11.6s，
   真值 11.65s）；
4. 文本按比例就近标点切开，两段继承原段的时间与情绪标签。

**真实录音上的表现**：这段 12 分钟录音 **0 次触发**——121 段里有 91 段本身就 <3s，
"一句两人"的情况几乎不存在。它更多是给**长段场景**（会议独白、40s 级 WhisperX 段）兜底。

## 5.2 说话人校正与非人声过滤

真实录音（车内对话 + 车机导航 + 停车场广播）暴露了两个问题，各加了一层校正：

**1) 基频辅助拆分说话人（`ascentics.split_by_f0`）**

情绪化/气声语音的 CAM++ 向量很不稳，曾把两个人并成 115:6。改用基频（自相关法）做二次拆分：
男女中位基频通常分别在 180–260 / 90–150 Hz，比声纹更稳。

- 大簇内 F0 呈双峰（两侧各 ≥3 段且占比 ≥20%）→ 按 165Hz 拆开；
- 再用 `cap_clusters` 限制到 `max_speakers`，超出的小簇并入基频最近的人。

效果：这段录音从 **115 : 6** 修正为 **80 : 41**，听感上终于像对话而非独白。

**2) 非人声过滤（`acoustics.looks_like_nav / looks_like_noise`）**

| 类型 | 判据 | 处理 |
|---|---|---|
| 导航/广播 | 导航词表（右转/前方/公里…）+ 停车场广播话术（照顾老人小孩/请勿/缴费…） | 标 `speaker_kind=nav`，不进整理稿 |
| 喇叭/提示音 | **明显更响（>1.8× 中位）+ 几乎没有浊音** + 文本极短 | 标 `noise`，不进整理稿 |

> 第一版把"嗯/是"这类应答误判成噪声（只看"短 + 无浊音"），已收紧成"必须同时更响"。

**已知不足**：词表法只能覆盖常见播报，换一种说法就漏；喇叭检测对小音量鸣笛无效。
彻底解决需要专门的音频事件模型（PANNs/AST 之类），当前未引入。

## 5.2.1 认名为什么不生效（排障记录）

现象：录了"刘益清"的声纹，重跑 12 分钟录音却完全没认出来。查到**四个叠加的问题**：

| # | 问题 | 后果 |
|---|---|---|
| 1 | 帧级分离分支**没有计算簇质心** | 认名拿不到比较对象 |
| 2 | 认名代码被缩进在"声纹聚类"分支里 | 帧级路径**根本不执行认名** |
| 3 | 自动模式 `num_clusters=-1` | 121 段裂成 **61 人**（阈值 0.9 仍有 17 人） |
| 4 | 阈值 0.72 按同信道设定 | 网页录音 vs 车内录音，相似度只有 **0.425**，永远过不了 |

修复：

1. 帧级分支补齐嵌入与质心；认名抽成两条路径共用的一段；
2. 自动人数改为**对 turn 做嵌入聚类 + 轮廓系数选 K**（实测稳定落到 2 人），
   并用 `max_auto_speakers` 兜底；
3. 阈值改为可配置：`match_threshold`（默认 **0.35**）、`match_margin`（默认 0.10），
   并在 `speakers.json` 记录 `confidence: high|low`（≥0.5 为 high）；
4. **顺手修掉一个真 bug**：查询向量没做归一化，模长会被当成相似度——单测抓到的。

修复后：**刘益清 = 簇1（165 秒，占 54%），相似度 0.425，领先第二名 0.387**，认名成功。

> 注意 0.425 属于**低置信度**（跨信道正常偏低）。若想更稳，建议用开会同款设备重录一遍声纹，
> 或在该设备上多录 1–2 段。

## 5.2.2 崩溃复盘：重跑后 KeyError 'spk'

**现象**：网页点重跑，worker 报 `KeyError: 'spk'`，任务失败。

**根因**：`work/<id>/diarize.json` 是上一次运行留下的（168 段标签），这次 ASR 只出 121 段。
旧代码的判断是：

```python
if len(cache["labels"]) == len(segs):   # 不相等时什么也不做
    ...
```

于是**整个分离阶段被静默跳过**，`spk` 从未赋值；后面角色消歧先报 `'spk'`（被 catch），
归档阶段直接崩溃。

**修复**：

1. 进入分离前 `s.setdefault("spk", -1)`，任何路径都不会缺字段；
2. 缓存校验抽成 `_load_diar_cache()`：段数不符或 JSON 损坏 → **删除缓存并重算**，
   同时打日志"说话人缓存段数不符（168≠121），已作废并重算"；
3. 所有 `s["spk"]` 改为 `s.get("spk")`（认名、角色、归档、Web 展示）。

**回归验证**：手动写入 999 条假标签的缓存再重跑 → 缓存被作废、分离重算、
认名成功、任务正常完成（497s）。新增 3 个单测覆盖"过期/正常/损坏"三种缓存。

## 5.2.3 为什么"新录了声纹却识别不出来"

**三个原因，都已修复**：

| 原因 | 说明 | 修复 |
|---|---|---|
| 认名缓存不失效 | `diarize.json` 命中直接复用旧 `names`，新声纹永远不参与比对 | 缓存写入 `vp_sig`（声纹条数+最新时间），变化即作废 |
| 质心被"稀释" | 用全部段算质心，安静/混合段拉低相似度 | `centroids_clean()`：只用 ≥2s 且 `spk_purity≥0.99` 的段，样本不足自动回退 |
| 阈值按同信道设定 | 跨信道相似度天然低 0.2–0.3 | `match_threshold` 0.35 / `match_margin` 0.08，并标 `confidence` |

**新功能**：任务详情页「重新认名」按钮 —— 不重新转写，只对已有结果重跑声纹比对（约 10 秒）。

实测（10月30日录音 + 刘益清/李彦昆两个声纹）：

| 簇 | 认名人 | 相似度 | 领先 | 说明 |
|---|---|---|---|---|
| 1 | 刘益清 | 0.452 | 0.22 | 明确 |
| 2 | 李彦昆 | 0.357 | 0.101 | 刚过线，标 low |

> 两人的**声纹样本本身很干净**（自一致性 0.906 / 0.867），低分来自**跨信道**：
> 网页麦克风录制 vs 车内/手机录音。想提高置信度：用录制该场景的设备重新登记，
> 或多录 2 段（尤其安静环境下的长句）。

## 5.3 情绪识别与叙事模式

**情绪识别**（`emotion.enabled`，默认开）：用 SenseVoiceSmall 一次拿到**文本 + 情绪 + 声学事件**。

| 情绪 | 事件 |
|---|---|
| HAPPY / SAD / ANGRY / NEUTRAL / FEARFUL / DISGUSTED / SURPRISED | Speech / BGM / Cry / Laughter / Cough / Sneeze / Breath / Applause |

> 两个实测坑：① `sentence_info[].text` **已剥离标签**，标签只在顶层 `text` 里，必须按 `<|zh|>` 切分后
> 顺序对齐；② 标签正则是大小写混合（`Cry`/`Speech`），用 `[A-Z_]+` 会漏掉所有事件。

**叙事模式**（`refine.style: auto|meeting|narrative`）：
- 说话人 ≤2 且（哭泣/难过占比 ≥3% 或句长中位 <3s）→ 自动 narrative；
- 段落合并窗口放宽到 **12 秒**，难过/哭时 9 秒，单段上限 600 字；
- 提示词改为"把被打断的碎片整理成连贯的 1–3 句，保留第一人称与情绪，不加心理描写"。

**引擎选择**（`asr.provider` 或 `--asr sensevoice`）：情绪化/叙事录音用 sensevoice，
会议仍用 funasr。同一段录音差异极大：

| 时间 | Paraformer | SenseVoice |
|---|---|---|
| 0:14 | 点难过怀孕不好玩 | 我有点难过，怀孕不好玩，为啥不好玩 |
| 0:20 | 到贪心那明 | 听不到胎心，那明天去听一下 |

## 5.4 ITN（数字规范化）与角色消歧

**ITN**（`text.itn`，默认开）：把口语数字写成阿拉伯数字，**只在有明确单位时转换**，避免改坏成语与口语。

| 输入 | 输出 |
|---|---|
| 接口响应时间从三百毫秒降到一百二十毫秒 | 从 **300毫秒** 降到 **120毫秒** |
| 增长了百分之十五，占比接近三十个百分点 | 增长 **15%**，接近 **30个百分点** |
| 二零二六年十月三十一号下午三点半 | **2026年10月31号下午3:30** |
| 预算是一千二百万美元 | **1200万美元** |
| 三心二意的方案不行 / 一个人做不完 | 不变（防误伤） |
| 三点五倍 | **3.5倍**（不是 3:05） |

**角色消歧**（`roles.enabled`，默认开）：从整理稿推断每个说话人是谁，**只在有依据时赋值**（自我介绍、别人称呼、职位上下文），无依据留空、不猜。

实测：6 分钟片段识别出「方案讲解人」「提问者」两个角色；声纹库已命名的人不会被覆盖。

## 5.5 纠错模式

```bash
python -m app.cli run a.wav --correct          # 开启 LLM 纠错
```

`correction.mode` 两种：

| 模式 | 文本 | 时间戳 | 适用 |
|---|---|---|---|
| `block`（默认） | 可读（允许合并断句） | 段落级 | 阅读、分享 |
| `segment` | 逐句（更碎） | **逐句精确** | 校对、引用 |

两种模式都做了：JSON/输出校验、元评论过滤、长度变化率限制、重复词兜底。

## 6.x Web 服务与访问控制

| 配置 | 默认 | 说明 |
|---|---|---|
| `server.bind` | `0.0.0.0` | 监听所有网卡；改 `127.0.0.1` 只允许本机 |
| `server.port` | `8000` | |
| `server.token` | 空 | 留空则启动时自动生成 `config/access_token.txt`（0600） |

**鉴权**（仅在绑定非回环时启用）：

- 页面：无令牌 → 302 跳 `/login`；浏览器输入一次后写 cookie（30 天）
- API：`?token=xxx`、`Authorization: Bearer xxx` 或 cookie，三者任一
- `/api/health` 与 `/login` 放行，其余全部拦截
- 令牌错误 → 401

> 之所以默认收紧：这个服务能下载全部录音、上传文件、触发重跑。以前只有一句
> "绑定非回环必须设 api_key" 的检查，但 `api_key` 在 Web 层**从未被校验**——等于没有防护。

## 7.x 多段声纹登记（Web）

**问题**：旧版只支持"一个姓名 + 一个文件"，而登记文本本身是 3 句（129 字，约 31 秒），
用户录完 3 段后无处上传。

**现在**：`/voices` 页面按文档 [04 §6.2](04-重叠说话与声纹登记设计.md) 列出 3 句 + 1 段自由说话，
每段独立录音（`MediaRecorder`）或选文件，可试听/重录，最后**一次提交全部**。

| 能力 | 实现 |
|---|---|
| 多段上传 | `POST /api/voices` 接收 `files: list[UploadFile]`（兼容旧的单 `file`） |
| 每段校验 | <1.5s 拒绝；逐段返回时长与错误原因 |
| 样本存储 | 每段一条 `voiceprints` 记录（`samples` 即段数） |
| **自一致性** | 两两余弦均值，<0.70 提示"可能有插话或环境变化"（文档 §6.4 步骤 6） |
| 不足 3 段 | 正常入库但提示补录 |
| 认名 | 同一人多条样本**按人取最大相似度**再比领先幅度，避免"自己比自己"吃掉 margin |

实测（用同一段语音切 3 份）：3/3 有效，自一致性 **0.936**，列表显示样本数 3；
只传 1 段时正确给出两条补录提示；重名被拒。

## 8. Web 功能清单（v0.14.0）

| 页面 | 能力 |
|---|---|
| `/` | 任务队列、进度、状态、上传 |
| `/job/{id}` | 四轨对照、摘要卡、导出 SRT/VTT/TXT、逐段改说话人/合并/删除、说话人校对、**声纹入库**、重新认名、重跑、删除任务 |
| `/hotwords` | 热词增删改、别名、权重、导入 |
| `/voices` | 多段声纹登记（3 句 + 自由说）、试听、删除 |
| `/settings` | 环境与模型自检（Python/ffmpeg/Ollama/CPU/内存）、关键配置一览 |
| `/logs` | 跨任务事件流、级别筛选、5 秒刷新 |
| `/api/jobs/{id}/events` | SSE 实时进度 |
| `*.md` 导出 | 中文文件名用 RFC 5987 编码 |

**手动干预后**：点「重跑」会从整理稿阶段重建（不重新识别），把人工修改带进整理稿与纪要。

## 5.6 ASR 幻觉过滤

灵感来自 [WhisprRT 调研](research/whisprt-realtime.md)：Whisper 在静音/音乐上会稳定编出套话
（"优优独播剧场""请不吝点赞"），FunASR 系则主要表现为**纯标点段**与**整段重复**。

`../code/app/hallucination.py` 的判定：

| 规则 | 处理 |
|---|---|
| 空 / 纯标点（"。"） | **整段丢弃**（在 ASR 之后、情绪/分离之前） |
| 套话黑名单（17 条中英正则） | 整段丢弃 |
| 连续重复（tandem repeat） | **不丢**，交给 `strip_repeats` 清理（真口吃也是内容） |
| 短句（"嗯。""是。"） | **不丢**（真实应答；空格率问题由 `quality_flag` 负责） |

> 真实语料 6,853 段回放：只命中 **287 段纯标点（4.2%）**，无短句误杀。
> 第一版把"短"和"重复"也算幻觉，误杀 14%，已修正。


## 5.7 词级时间戳

| 来源 | 精度 | 做法 |
|---|---|---|
| Paraformer | **精确** | `return_raw_text=True` 拿到空格分隔 token，与 `timestamp` 一一对应（实测 17/17 对齐） |
| SenseVoice | 估算 | 模型不返回 token 时间（`timestamp` 为空），按字符数在句内均摊，标 `approx=1` |

- 落库：`segments.words`（JSON `[{t,s,e,approx}]`）；
- 前端：**原文轨**每个字/词可点击，点了播放器跳到该词起点；
- 导出：`/api/jobs/{id}/export?format=srt&level=word` 生成逐词字幕（14KB / 60 秒音频）。

> 为什么只对原文轨做：清洗稿删了口水词、整理稿是 LLM 重写的，词与音频不再一一对应。
> WhisperX 等工具同理——词级对齐永远针对**逐字稿**。

## 9. 运维与健壮性（v0.17.0）

| 能力 | 实现 |
|---|---|
| 资源守卫 | `../code/app/resource.py` 在任务开始前检查磁盘/内存，不足直接失败并写明原因（不再跑到一半 OOM） |
| 任务通知 | `notify.webhook_url`：完成/失败 POST JSON，超时 5s，失败不影响主流程 |
| 接口限流 | `openai_api.rate_limit_per_min`（默认 0=关），按 IP 令牌桶，仅对 `/v1/` 生效 |
| 文件夹监听 | `python -m app.cli watch`：监听 `input/` 自动入队处理（补上 FR-A1 的常驻形态） |
| ASR 幻觉过滤 | 黑名单 17 条 + 纯标点段剔除（真实语料丢弃率 4.2%，短句零误杀） |
| 静音门控 | 整段有效语音占比 <2% 直接跳过（60s 纯噪声 20s → 1.2s） |

## 6.y 设置页（可编辑，不只是查看）

`/settings` 从只读状态页升级为**可编辑表单**：36 个字段分 7 组（识别 / 说话人 / 文本后处理 /
整理稿与纪要 / LLM / **Obsidian** / 接口与服务）。

| 能力 | 实现 |
|---|---|
| 保存 | `PATCH /api/settings` → 写 `config/overrides.yaml`（**不动 config.yaml，保留注释**） |
| 校验 | 类型/范围/枚举逐字段校验，错误直接标在字段下方 |
| 热更新 | 保存后 `reload_into(cfg)` 就地替换进程内 Config，**下一个任务立即生效** |
| 需重启项 | `server.bind` / `server.port` 标红提示，不静默失败 |
| 恢复默认 | 删除 `overrides.yaml` 并重新加载 |
| Obsidian 仓库 | 输入框 + **「测试写入」**：检查目录存在、是目录、可写（真写一个探针文件再删） |

**优先级**：`overrides.yaml` > `config.yaml` > 代码默认值。手工想改回去，删掉 overrides 即可。

实测：在设置页把仓库地址改成 `/tmp/obs-vault` → 校验通过 → 跑任务 → 笔记确实写到
`/tmp/obs-vault/会议纪要/2026/…md`，无需重启。

## 10. 部署：统一入口与开机自启

### 10.1 一个命令跑全部

```bash
PYTHONPATH=code .venv-funasr/bin/python -m app.cli service
```

一个进程里同时跑：**监听 input/**（新音频自动入队）→ **队列处理** → **Web 服务**。
`--no-web` 可只跑前两者；`--port` 覆盖端口。

> 监听器已改成**非阻塞**轮询：旧实现每个文件要 sleep 4 秒判断"大小稳定"，
> 目录里已有几个文件时新文件要排队等十几秒；现在并发等待，实测丢文件进 `input/`
> 约 4 秒后自动入队并开始处理。

### 10.2 开机自启（macOS launchd）

> 若命令提示"无权写入 ~/Library/LaunchAgents"（沙箱/权限受限），
> 改用一次性脚本：`bash scripts/install-autostart.sh`（在你自己终端执行，会生成 plist、
> 安装到 LaunchAgents 并 bootstrap）。

```bash
python -m app.cli autostart install      # 注册并立即加载
python -m app.cli autostart status       # 看是否安装/运行
python -m app.cli autostart restart      # 重启服务
python -m app.cli autostart uninstall    # 移除
```

生成 `~/Library/LaunchAgents/com.tingji.service.plist`：

| 键 | 值 |
|---|---|
| `ProgramArguments` | 当前 venv 的 python + `-m app.cli service --config <绝对路径>` |
| `WorkingDirectory` | 项目根目录（相对路径配置才不会错位） |
| `RunAtLoad` / `KeepAlive` | 开机启动；非正常退出自动拉起 |
| `StandardOutPath` | `logs/service.log`（另有 `.err.log`） |

`plutil -lint` 校验通过。日志：`tail -f logs/service.log`。

### 10.2.1 自启踩坑记录（v0.28.1）

**症状**：执行 `bash code/scripts/install-autostart.sh` 后，网页打不开了；`launchctl print`
显示 `last exit code = 78: EX_CONFIG`，日志里没有任何 Python 报错。

**根因**：生成的 plist 把 `--config` 放在了子命令**后面**：

```
python -m app.cli service --config .../config.yaml --port 8000   ❌
```

`-c/--config` 是 argparse 的**全局参数**，必须在子命令之前；放错位置 argparse 直接
`unrecognized arguments` 退出，launchd 于是不断重试 → 对外表现为"服务起不来"。

**修复**：

1. `build_plist()` 改为 `python -m app.cli -c <config> service [--host] [--port]`；
2. `install()` 增加**命令行预检**：拿要写进 plist 的命令跑一次 `--help`，非 0 就直接
   报错返回，不再等 launchd 崩溃；
3. 安装脚本增加**端口占用处理**（先停手工实例）与**健康检查**（`curl /api/health`），
   失败时直接打印 `service.err.log` 末尾。

> 教训：凡是"写配置文件给别的进程执行"的场景，都要在写之前把那条命令**自己跑一遍**。

**第二个坑（v0.28.2）**：改好 plist 后重装，仍然 `exit 78`。

根因不是新 plist 有问题，而是**旧任务根本没卸载**：`launchctl bootstrap` 在 label 已加载时
会失败，但脚本没检查返回码，于是 launchd 内存里继续跑**旧参数**（`service --config` 错位那版），
plist 文件换了也没用。

修复：重装前按 **label** 与 **path** 各 `bootout` 一次，确认 `launchctl print` 查不到再
`bootstrap`，并检查返回码；失败时打印 hint（`launchctl bootout gui/$(id -u)/<label>`）。

> 排查口诀：**plist 文件新 ≠ 任务新**。改完先 `launchctl print` 看它加载的是不是旧配置。

**第三个坑（v0.28.3）**：改成"按 label 卸载"后重装，仍然 `exit 78`，且 `launchctl print` 里是：

```
program = …/code/.venv-funasr/bin/python     ← venv 根本不在 code/ 里
stdout path = …/code/output/.runtime/logs/…  ← 日志也错位
```

根因是脚本生成 plist 时把"项目根"和"code 根"混用了：`root` 传的是 `code/`，
再用它拼 `root/.venv-funasr/bin/python`，于是路径全部多了一层 `code/`。
`launchd` 找不到可执行文件 → 78 EX_CONFIG。

修复：脚本里明确区分 `ROOT`（项目根）与 `CODE_ROOT`：

| 用途 | 基准 |
|---|---|
| python / venv / output / logs | `ROOT` |
| 工作目录 / config / plist 输出 | `CODE_ROOT` |

并新增**安装前命令预检**（②）：拿 plist 里的 `ProgramArguments` 加 `--help` 真跑一次，
失败就直接退出——路径写错会立刻暴露，不用等 launchd 反复崩溃。

> 三次踩坑的共同点：**配置写对了没用，要验证"真能跑"**。预检那一步是这次最值钱的补丁。

**第四个坑（v0.28.5）**：plist 参数、路径、卸载都正确，launchd 仍然 `exit 78`，
系统日志只有一句 `already handled failed init, ignoring`，而自检显示 `/bin/echo` 和
我们的 python **都能被 launchd 拉起**（`last exit code = 0`）。

差别只剩 plist 里的 **`StandardOutPath` / `StandardErrorPath`**：原来指向
`~/Documents/voice-to-text/output/.runtime/logs/`。macOS 的 TCC 保护 `~/Documents`，
launchd 打不开这个路径的日志文件 → 任务初始化失败 → `EX_CONFIG(78)`。

修复：自启日志改到 **`~/Library/Logs/TingJi/`**（不受保护），应用自身日志仍在
`output/.runtime/logs/`（由 Python 自己写，不受影响）。

### 10.3 任务流程图

任务详情页顶部新增「处理进度」节点条，13 个节点：

```
归一化 → 分窗 → 语音识别 → 情绪/事件 → 说话人分离 → 声纹认名 → 文本清洗
→ 热词修正 → LLM纠错 → 整理稿 → 角色推断 → 智能纪要 → 归档
```

- 状态：**绿=完成 / 蓝=进行中 / 灰=跳过 / 红=失败 / 暗=未开始**；
- 每节点显示耗时与关键指标（"121 句""2 人""章节 4"）；
- 数据来自 `GET /api/jobs/{id}/stages`：只取**最近一次运行**的事件，
  耗时按"上一阶段结束 → 本阶段结束"计算（顺序管线里就是该阶段真实耗时）。

## 7.y 声纹库管理（追加 / 改名 / 删样本）

`/voices` 从"只能新建+整人删除"升级为完整管理：

| 操作 | 接口 | 说明 |
|---|---|---|
| 追加录音 | `POST /api/voices/{sid}/samples` | 多文件一次传，逐段校验；≥1.5s 才收；返回新增数与**全部样本的自一致性** |
| 查看样本 | `GET /api/voices/{sid}/samples` | 每条的 quality / source / 时间 |
| 删除单条 | `DELETE /api/voices/{sid}/samples/{vid}` | 只剩 1 条时拒绝（否则无法认名） |
| 改名 | `PATCH /api/voices/{sid}` | 重名会拒绝 |
| 删除人物 | `DELETE /api/voices/{sid}` | 连同样本一起删 |

**自一致性**：每次追加后重算全部样本两两余弦均值，<0.70 时提示"建议删掉差异大的样本"。
实测拿错音频（模型音色不同）追加后一致性从 0.87 掉到 **0.32**，警告如期出现——
这正是"追加了错的录音"的信号。

> 认名用的是"同一人取最大相似度"，所以样本多一些通常更稳；
> 但**互相矛盾的样本不会互相抵消**，只会让最大值指向错误的人，所以低一致性要处理。

## 10.4 开机自启（登录项方案，v0.29.0 起）

LaunchAgent 在本机反复失败（`exit 78 EX_CONFIG`），原因是 macOS 的 TCC 保护 `~/Documents`：
launchd 拉起任务时读不到项目文件、也打不开日志，任务在初始化阶段就失败——即使
plist 参数、路径、日志位置全部正确，连 `/bin/echo` 能跑、我们的 python 单独也能跑。

**改用登录项**，由 **Terminal** 打开 `.command` 启动服务。Terminal 已经被授予
Documents 访问权限，因此不受这条限制。

```bash
bash code/scripts/install-login-item.sh     # 安装（会请求控制 System Events 的权限）
bash code/scripts/install-login-item.sh --remove   # 卸载
```

做了三件事：

| 文件 | 作用 |
|---|---|
| `~/Library/Application Support/TingJi/start.sh` | 启动服务（已在跑则跳过），日志写 `output/.runtime/logs/` |
| `~/Library/Application Support/TingJi/TingJi.command` | 登录项入口，调用 start.sh 后立刻退出 |
| 系统登录项 | 通过 `osascript` 注册，`hidden:true` |

若 `osascript` 报 `-10004 权限违例`，是没授予自动化权限：到
**系统设置 → 隐私与安全性 → 自动化** 允许 Terminal 控制 System Events；
或手动在 **通用 → 登录项 → 打开时 → +** 选中那个 `.command` 文件。

旧的 LaunchAgent 会被自动清理（`launchctl bootout` + 删除 plist），避免两套机制抢端口。

## 11. 排障：页面能打开但列表是空的 / 按钮没反应

**症状**：用 `http://ip:8000/voices?token=xxx` 能打开页面，但声纹库列表空、点"提交注册"无反应；
其它页面同样如此。

**根因**：鉴权中间件在 `?token=` 通过校验后**没有给浏览器种 cookie**。页面本身是服务端渲染的，
所以能打开；但页面里的 `fetch('/api/voices')` 不带 token、也没有 cookie → 全部 401 →
列表空、提交失败（前端只看到 `d.ok` 为 undefined）。

**修复**（v0.21.1）：

```python
if cand and cand == token:
    resp = await call_next(request)
    if not request.cookies.get("tingji_token"):
        resp.set_cookie("tingji_token", token, httponly=True, samesite="lax", max_age=30*86400)
    return resp
```

另外在公共 JS 里包了一层 `fetch`：任何 401 直接跳 `/login`，不再"静默空白"。

**回归测试**：`TestTokenCookiePropagation` —— 带 `?token` 请求页面后，用**同一个 client**
（只有 cookie）访问 `/api/voices` 必须是 200；清掉 cookie 后必须 401。

> 教训：这类"服务端渲染正常、客户端请求 401"的问题，用 curl 带 token 逐条验证是发现不了的，
> 必须模拟"页面 + 后续 XHR"的完整链路（TestClient 的 cookie jar 正好能做）。

### 11.1 第二个坑：`service` 入口没启用令牌（v0.21.2）

统一入口 `app.cli service` 是后加的，**没有复用 `serve` 的令牌准备逻辑**，
结果 `cfg.server.token` 为空 → 中间件判定"无需鉴权" → **8000 端口公开可访问**。

修复：抽出 `_prepare_server()` / `_print_urls()`，`serve` 与 `service` 共用一套；
`autostart` 注册的正是 `service`，所以这个洞会随开机自启一直存在。

**自查方法**：`curl -s -o /dev/null -w "%{http_code}" http://ip:8000/api/voices`
——非回环绑定应当是 **401**，返回 200 就说明鉴权没生效。

### 11.2 第三个坑：改了代码但进程没重启

表现：页面能开、某些新功能不出现，看不出原因。

修复：`/api/version` 返回 `started_at` / `code_mtime` / `stale`；
页面加载时如果发现代码比进程新，顶部显示一条黄色提示"请重启服务"。
旧进程（不含该接口）访问 `/api/version` 会 404，本身也是"该重启了"的信号。

### 11.3 第四个坑：Python 把 JS 里的 `\n` 变成了真换行

**症状**：`/voices` 页面能打开、服务端 200，但**列表空白、每段的"上传文件"按钮都没了**。

**根因**：JS 写在 Python 三引号字符串里：

```python
# app/web.py（错误写法）
+ (d.warnings.length ? ('\n⚠️ ' + d.warnings.join('\n')) : '')
```

Python 会把 `\n` 解释成**真实换行**，下发的 JS 变成：

```js
+ (d.warnings.length?('
              ⚠️ ...
```

单引号字符串里出现裸换行 → **整个 `<script>` 语法错误** → 后面的 `renderSlots()` / `load()`
一行都没执行 → 槽位（含文件选择框）和表格全都不渲染。

**修复**：JS 里不写反斜杠转义，改用 `String.fromCharCode(10)`。

**回归**：删除人物/样本会写 `voices` 事件（之前静默删除，无法追溯）；`TestServedJsParses` —— 用 TestClient 取 `/`、`/hotwords`、`/voices`、
`/settings`、`/logs`、`/job/{id}` 六个页面，提取内联 `<script>` 跑 `node --check`。
故意注入一个坏换行已验证该测试会失败。

> 教训：**服务端 200 不等于页面可用**。前端是内联 JS 时，任何字符串转义错误都会静默
> 毁掉整页交互，而 curl 只看状态码完全发现不了。必须真的执行或至少做语法检查。

## 5.8 按内容类型命名与归档

不再把所有笔记都叫「会议纪要」。纪要阶段会先判断**这段录音是什么**，再据此命名与归档。

| 类型 | 判定（LLM 优先，失败走启发式） |
|---|---|
| 会议纪要 | 多人轮流发言、有议题与结论；≥3 位说话人 |
| 日常对话 | 两人或少数人闲聊；≤2 人且 <30 分钟 |
| 重要时刻 | 有明显情绪/纪念意义（表白、生日、宝宝胎动、告别）；哭泣或难过占比 ≥15% |
| 访谈 | 一问一答的采访 |
| 讲座 | 单人长时间讲述；1 人且 ≥15 分钟 |
| 备忘 / 其他 | 简短记录 / 都不像 |

**命名**：`{日期} {标题}`，标题由 LLM 从内容概括（≤20 字），取不到时回退文件名。

**归档结构**（`notes_by_type` + `per_note_folder`，默认都开）：

```
录音笔记/
  会议纪要/2026/
    2026-10-31 2222 某客户亚太 AI 算力项目背景与需求梳理/
      同名.md      ← 总结 + 章节 + 待办 + 整理稿（正文）+ 说话人表
      同名.m4a     ← 音频附件
      同名.srt     ← 字幕
      同名.txt     ← 纯文本
  重要时刻/2026/
    2026-10-31 2212 孕妇因胎动减少产生厌孕情绪及家人安抚/
      同名.md / 同名.m4a / 同名.srt / 同名.txt
转写稿/2026/
  同名-逐字稿.md   ← 逐字稿单独放这里
  同名-整理稿.md
```

要点：
- **一个文件夹 = 一次录音**，音频与正文放在一起，方便整包拷贝/分享；
- **逐字稿独立**在 `转写稿/`，笔记里只放整理稿正文 + `[[…-逐字稿]]` 链接；
- 类型判定优先看**内容主题**（工作/业务/客户 → 会议纪要；生活/家庭 → 日常对话），
  而不是只看人数——之前"一对一客户沟通"会被误判成日常对话，已修正；
- 关掉 `per_note_folder` 可回到"音频统一放 attachments"的旧结构。

frontmatter 增加 `type: <类型>` 与 `doc_type: <类型>`，tags 里加 `类型/<类型>`。

实测（两条真实录音）：

| 录音 | LLM 类型 | 生成标题 | 落盘目录 |
|---|---|---|---|
| 会议片段（60s） | 会议纪要 | 客户接洽背景与 AI 算力需求分析 | `录音笔记/会议纪要/2026/` |
| 孕期情绪对话（12.7 分钟） | 重要时刻（启发式：难过占比 16%） | 见该次运行 | `录音笔记/重要时刻/2026/` |

> 目录根从「会议纪要」改名为「录音笔记」——否则会出现 `会议纪要/会议纪要/` 这种双层目录。
> 老笔记不会被移动；想统一可以把旧目录整体拖进新结构。

## 5.9 正则纠错规则（借鉴 CapsWriter-Offline，详见 research/three-projects.md）

热词表只能替换"词"，处理不了**结构化**的错。新增 `config/rules.txt`：

```
欧拉玛 = Ollama
(艾特|at)\s*([A-Za-z0-9_.-]+)\s*(点|dot)\s*(com|cn|net|org) = @\2.\4
(?<![@A-Za-z0-9])([a-z0-9-]{2,})\s*点\s*(com|cn|net|org|io) = \1.\2
\\[(breath|sil|noise)\\] =
```

- 每行 `查找 = 替换`，**等号两侧要有空格**（正则里带 `=` 时才不会被拆错）；
- 查找到用正则，替换支持 `\1` 反向引用；`#` 开头为注释；
- 坏规则只跳过并记日志，不会中断任务；
- 执行顺序：热词替换之后、整理稿之前。

实测：`邮箱是 艾特 zhangsan 点 com` → `邮箱是 @zhangsan.com`。

## 5.10 ITN 约数/范围（借鉴 CapsWriter-Offline）

| 输入 | 输出 |
|---|---|
| 十五六个人 | **15~16个人** |
| 三五百台服务器 | **300~500台服务器** |
| 二三十台机器 | **20~30台机器** |
| 三四个问题 | **3~4个问题** |
| 十五个 / 十个人 | **15个 / 10个人**（≥10 时才转普通量词） |
| 二十三岁 / 三心二意 / 一个想法 | 不变（防误伤） |

关键教训：范围转换必须**在量词转换之前**做，否则"三五百台"会先被当成数字 500。

## 5.11 Fun-ASR-Nano 引擎（借鉴 Brevia）

**为什么加**：Brevia 用 Fun-ASR-Nano，句子更完整、自带标点。我们确认 **sherpa-onnx 2.0.1 原生支持**
`OfflineRecognizer.from_funasr_nano`——不需要 torch/funasr，模型 1.0 GB（int8）。

```bash
python -m app.cli run 录音.m4a --asr funasr_nano
# 或在 /settings 里把「识别引擎」改成 Fun-ASR-Nano
```

**切分策略**（`../code/app/nano.py`，照抄 Brevia 的思路）：

| 机制 | 值 | 原因 |
|---|---|---|
| 目标块长 | 8 秒 | 太碎识别质量差 |
| 切点 | 6–10 秒内**能量最低点** | 不在一句话中间硬切 |
| 硬上限 | 22 秒 | Nano 的 KV 容量 ~512，实测 >28s 截断、~29.5s 整段为空 |
| 段合并 | 句末有 `。！？` 就切，否则并入下一块 | 对应 Brevia 的 `endpoint` 语义 |

**实测（M4 CPU / int8）**：RTF **0.13–0.20**（SenseVoice 约 0.015，Paraformer 约 0.047）。

**10.6 分钟真实对话三引擎对比**（关掉 LLM 层，只比识别）：

| 指标 | Paraformer | SenseVoice | Fun-ASR-Nano |
|---|---|---|---|
| 总字数 | 1270 | 1240 | **1841** |
| 段长中位数 | 1.16s | 1.41s | **1.55s** |
| 标点异常 | **5** | 14 | 12 |
| 耗时（10.6 分钟） | 155s | **104s** | 182s |

同段 00:00–01:00：

- Paraformer：`点难过怀孕不好，玩为啥`
- SenseVoice：`我有点难过，怀孕不好玩…听不到胎心`
- Nano：`我有点难过，怀孕不好玩。为啥不好玩？…听不到，担心那明天去听一下`

**Nano 的句子连贯性最好**（尤其安静/情绪化录音），代价是口水词更多、偶发编造连接词。
复现脚本：`bench/compare_engines.py`。

**单片段 A/B**：

| 片段 | Nano | 对比引擎 |
|---|---|---|
| 业务通话 | `跟mute了哦，我刚mute掉了…ICP落地接触` | Paraformer：`我我刚掉掉…ICT落地接触`（ICT 更准） |
| 情绪录音 | `…听我的，担心那明天去听一下…去食堂听一下` | SenseVoice：`…听不到胎心…去社康听一下`（关键语义更准） |

结论：**按场景选引擎**——会议默认 `funasr`，叙事 `sensevoice`，质量优先 `funasr_nano`。
详细分析见 [research/brevia-funasr-nano.md](research/brevia-funasr-nano.md)。

## 5.12 四项借鉴落地（源自 Brevia）

### 5.12.1 非语音掩蔽 + 安静语音兜底（`diar_sherpa.mask_non_speech`）

分离前把"VAD/ASR 判为非语音"的音频**置零**，只留下真正有人说话的片段：

- 好处：BGM、路噪、键盘声不再参与说话人聚类，短段声纹不再被污染；
- **兜底**：夹在两段语音之间、时长 1–20 秒、电平不低于整体语音 8% 的"空隙"**保留**——
  音乐里的轻声、电话音如果被整段抹掉，说话人会凭空少一个。
- 日志会写：`非语音掩蔽：N 段语音，其余置零`。

### 5.12.2 15 秒窗口声纹（`diar_sherpa.centroids_from_turns`）

不再"每个 ASR 段算一个向量求均值"，而是**每个 turn 切 15 秒窗口分别提声纹再平均**：

- 长 turn 不会因为整段平均而被稀释；
- 短于 1.5 秒的窗口直接跳过，不污染质心；
- 日志会写：`声纹质心：N 人（按 turn 15s 窗口）`。

配置：`diarization.embed_window_sec`（默认 15）、`diarization.mask_non_speech`（默认开）。

### 5.12.3 切点回补 + 已交付水位（`nano.split_chunks`）

- 每块交付区间 `start/end` **首尾相接**（水位）；解码时向前多带 `pad=0.3s`；
- `pad` **不得超过上一块的 `end`**，避免重复解码已交付内容；
- 重叠导致的重复文本用 `dedupe_seam()` 去掉（最长 12 字的前后缀匹配）。

### 5.12.4 边界可信度（`boundary` 字段）

每个 Nano 段带 `boundary`：

| 值 | 含义 | 句末标点 |
|---|---|---|
| `endpoint` | 切在真静音上 | **可信** |
| `pause` | 切在低能量但非静音 | 模型补的，不可信 |
| `cut` | 到 22 秒硬上限 | 不可信 |

`textproc.trim_uncertain_tail()` 在清洗阶段把 `pause/cut` 段末尾的补出来的标点去掉。
字段已落库（`segments.boundary`）并在 `/api/jobs/{id}` 返回，前端可据此显示/排查。

**实测**（slice60，Nano 引擎）：7 段，boundary 分布 `endpoint×6 + pause×1`；
分离阶段日志出现"非语音掩蔽：21 段语音"与"声纹质心：1 人（按 turn 15s 窗口）"。

## 5.13 自动选识别引擎（`asr.provider: auto`，默认）

识别前先做两件轻量探测，再决定用哪个引擎：

1. **有效语音占比**（静音门控顺带算的）；
2. **说话人数**（在开头 90 秒上跑一次 pyannote 分离，约 3–5 秒）。

| 条件 | 引擎 | 理由 |
|---|---|---|
| 时长 ≥ 20 分钟 | `funasr` | 长会，Paraformer 快且词级时间戳精确 |
| 探测到 ≥3 位说话人 | `funasr` | 多人会议 |
| 只有 1 位说话人 | `funasr` | 讲座/独白，术语准确性更重要 |
| **2 位说话人**（一对一） | **`funasr_nano`** | 对话类，Nano 句子更连贯 |
| 探不到人声/其他 | `funasr_nano` | 碎片化、安静录音 |
| Nano 模型缺失 | `sensevoice` | 回退 |

**为什么不用语音占比区分**：实测那段 12.7 分钟情绪对话的语音占比是 **98%**，比很多会议还高——
重度说话、密集停顿反而是私人录音的特征。**说话人数**才是可靠信号。

**实测**

```
情绪对话（2 人 / 12.7 分钟）：
  说话人数探测：2 人
  自动选引擎 → funasr_nano（一对一对话，按叙事处理）

业务片段（1 人 / 60 秒）：
  自动选引擎 → funasr（单人长讲，按讲座/独白处理）
```

想强制指定引擎：`--asr funasr | sensevoice | funasr_nano`，或在 `/settings` 里选。

## 5.14 OpenAI 兼容接口：客户端怎么填（v0.29.1）

服务已提供 `/v1/models` 与 `/v1/audio/transcriptions`，可以直接接到 Memos / Open WebUI /
任何支持 OpenAI Whisper 的客户端。

| 字段 | 填什么 |
|---|---|
| Provider 类型 | `OPENAI` |
| 端点 | `http://<本机IP>:8000/v1`（本机用 `http://127.0.0.1:8000/v1`） |
| API 密钥 | `config.yaml` 里的 `openai_api.api_key`（空 = 不校验） |
| 模型 | `tingji-asr`（快）/ `tingji-asr-hq`（带说话人）/ `whisper-1`（兼容别名） |
| 说话人 | 只有 `-hq` 返回；`json`/`text` 会把「说话人1：…」写进 `text`，`verbose_json` 另带 `segments[].speaker` |
| 语言 | `zh`（仅透传，识别本身以中文为主） |
| 提示词 | 热词，如 `人名：刘益清、李彦昆。术语：胎心、社康。` |

**三个修正**（都是接客户端时暴露的）：

1. `/v1/*` 之前被**网页访问令牌**拦截，返回 302 跳登录页——客户端完全没法用。
   现在 `/v1/*` 只走 `openai_api.api_key` 鉴权，`/v1/models` 也补上了鉴权。
2. `prompt`（Whisper 的热词提示）之前被忽略；现在会拆成词条作为**请求级热词**，
   权重高于热词库，只影响本次请求。
3. 模型路径（`campplus.onnx` 等）是相对路径，服务以 `code/` 为工作目录时会找不到；
   现在 `load_config()` 统一解析成**绝对路径**。

```bash
curl -H "Authorization: Bearer $KEY" \
  -F file=@a.wav -F model=whisper-1 -F language=zh \
  -F "prompt=人名：刘益清" -F response_format=verbose_json \
  http://127.0.0.1:8000/v1/audio/transcriptions
```

**支持的音频格式**（v0.29.2 起）：

```
wav mp3 aac m4a m4b mp4 mov mkv flac
webm weba opus ogg oga wma amr 3gp caf aiff aif
```

后端用 ffmpeg 解码，所以**扩展名不再作为门槛**：不在列表里的名字也会先落盘再用
`ffprobe` 试解，能解就转写，解不了才回 400（报错里带扩展名与原因）。
浏览器 `MediaRecorder` 默认的 `webm/opus`、iOS 的 `m4a`、安卓的 `3gp/amr` 都已覆盖。

**hq 模型的说话人输出**（v0.29.3）：

- `json` / `text`：`text` 字段直接是分行正文 `说话人1：…\n说话人2：…`（相邻同人合并）——
  大多数客户端只读 `text`，不带进正文的话等于没返回；
- `verbose_json`：额外给 `segments[].speaker` 与 `x_tingji.speakers`；
- 若该说话人的声纹在声纹库里匹配上（阈值 `diarization.match_threshold`），
  名字会替换成真人名（如 `刘益清：…`），否则用 `说话人N`。

同步上限 `openai_api.sync_max_audio_sec`（默认 1800 秒）；超过返回 `202` +
`Location: /v1/audio/transcriptions/{id}`，客户端需要轮询——多数客户端不认 202，
长音频建议先在网页端跑。

## 5.15 模型路径与加载失败的坑（v0.29.4）

服务改成以 `code/` 为工作目录后，**所有模型路径都必须是绝对路径**，否则会以
`code/` 为基准去找 `.cache/sherpa/...`，一份都找不到。踩到两处：

| 位置 | 症状 |
|---|---|
| `diar_sherpa.SEG_MODEL` / `EMB_MODEL` | 日志 `帧级分离失败，说话人未知`，整篇没有说话人 |
| `nano.MODEL_DIR`、`web` 的模型自检列表 | Fun-ASR-Nano 不可用、设置页模型状态显示缺失 |

修复：模块加载时用 `settings.ROOT` 拼绝对路径（`ROOT = code/` 的上一级）。

另一处是 FunASR 的**加载状态污染**：某个 46 分钟任务先报
`model 'paraformer-zh' is not registered`，随后同进程再报
`TypeError: None argument after ** must be a mapping`（模型名退化去找 `Conformer`，
缓存里真的出现了一个 `Conformer/` 目录）。这是长期运行进程里初始化失败后的残留状态。

修复（`engines.AsrEngine._load`）：

1. 模型名为空 → 兜底 `paraformer-zh`；
2. 首次加载抛错 → **清空 `_model` 缓存**，用完整 ModelScope id
   （`iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`）重试一次；
3. 仍失败才抛出，错误信息原样带到任务页。

验证：同一份 46 分钟 `会议2.aac` 修复后转写 **700 句**、完成归档
（`录音笔记/会议纪要/2026/2026-11-02 2103 带宽费用异常分析与管控会议/`）。

> 教训：**长驻进程里模型加载失败后必须重置缓存再重试**，否则一次失败会污染后续所有任务。

## 5.16 任务归档/删除与登录流程（v0.30.0）

### 任务归档

任务列表每行新增「归档 / 恢复 / 删除」：

| 操作 | 行为 |
|---|---|
| 归档 | `jobs.archived=1`，默认列表不再显示；文件、逐字稿、Obsidian 笔记都保留 |
| 恢复 | `archived=0`，重新出现在列表 |
| 删除 | 删 DB 记录 + `output/<任务名>/` + `work/<id>/`；**Obsidian 里已归档的笔记不动** |

- API：`POST /api/jobs/{id}/archive` body `{"archived": true|false}`；`GET /api/jobs?archived=1` 含已归档；
- 列表页有「显示已归档」勾选；
- 老库自动迁移：`ALTER TABLE jobs ADD COLUMN archived INTEGER DEFAULT 0`。

> 注意：`archived` 列必须加在 **jobs** 表。第一次误加到了 segments 的迁移列表里，
> 结果接口 500 —— 迁移代码要按表分开写。

### 登录流程

原实现有三个小问题，都已修：

1. 跳转用 `302` + 原样 `next`，遇到 `next` 被编码成 `%2F` 或被改写成 `+` 时会跳到怪路径
   （`/+`、`/%2F`）→ 现在 **先 `unquote`，只接受以 `/` 开头且不以 `//` 开头的路径**，否则回 `/`；
2. 改成 **`303 See Other`**，浏览器用 GET 打开目标页，token 不留在地址栏；
3. 已登录（cookie 有效）再访问 `/login` 直接跳目标页，不再显示表单。

```
GET /            → 302 /login?next=/
GET /login?token=…&next=%2F  → 303 /
GET /            → 200
```

## 5.17 Web 界面改版：Memphis（v0.31.0）

按 `docs/08-界面设计规范.md`（DESIGN.md v1.0）重做了整站样式，功能与交互逻辑不变，只换视觉层。

| 令牌 | 值 | 用在哪 |
|---|---|---|
| 背景 | `#FFF8E8` + 点阵纹理 | 页面底 |
| 卡片 | `#FFFFFF`，`3px` 黑边，`20px` 圆角 | 所有 `.card` |
| 硬阴影 | `8px 8px 0 #00C2FF` | 卡片 |
| 主按钮 | `#FFE45E`，胶囊，`4px 4px 0 #FF4FA3` | 上传/保存/重跑 |
| 强调 | `#FF4FA3` 粉 | 链接、进度条、当前标签、说话人 |
| 标签/徽章 | 黑边胶囊，按状态换底色 | 任务状态、类型徽章 |
| 字体 | Space Grotesk（标题）/ Inter（正文） | 本机没有时回落 Avenir Next / PingFang SC |

细节：

- 表头 `13px` 大写 + `0.05em` 字距，行间用 2px 暖色分隔线，不再用细灰线；
- 按钮有 hover 上浮、active 下压（阴影同步收放），保留可点感；
- 音频文件选择框改成虚线卡片，`::file-selector-button` 也套了胶囊样式；
- 流程图的节点改成 3px 描边圆点，完成是粉色、运行是青色、失败是红色；
- `prefers`/窄屏：卡片阴影减到 5px、表格字号 13px、字段单列。

验证：用 headless Chrome 截图四张（首页/任务/设置/登录）逐页对照规范，
并跑 `TestServedJsParses` 确保注入的 CSS 没有破坏页面脚本。

### 字体（v0.31.1）

规范要求 Space Grotesk / Inter，本机没有安装。现在把字体**打包进仓库**，完全离线：

```
code/app/static/fonts/
├── space-grotesk-700.woff2   13 KB
├── space-grotesk-500.woff2   13 KB
├── inter-400.woff2           23 KB
└── inter-600.woff2           24 KB
```

- 来源 `@fontsource/*`（SIL OFL），只取 **latin** 字形——中文仍由苹方渲染，避免几 MB 的 CJK 字体；
- 通过 `app.mount("/static", StaticFiles(...))` 提供；
- **`/static/` 不鉴权**：登录页也要加载字体，否则首屏会闪一次回落字体；
- `@font-face` 用 `font-display:swap`，加载失败自动回落到 Avenir Next / 苹方；
- 顺手加了 SVG favicon（黄底粉块，内联 data URI，无额外请求）。

## 5.18 声纹认名：按信道聚类（v0.32.0）

### 问题

录音设备不同时，同一个人声纹差异可能比不同人还大。实测（刘益清）：

| 对比 | 余弦 |
|---|---|
| 库内 3 条样本之间（同一网页麦克风） | 0.90–0.92 |
| 网页样本质心 vs 车机录音片段 | **0.25–0.39** |
| 同录音内部录入 vs 留出片段 | 0.50–0.77 |

原来"每人一个质心"（把所有样本平均）会把两种信道都拉糊：平均后的向量既不像麦克风录的、
也不像车机录的，跨设备查询只能拿到 0.3 上下，正好卡在认名阈值 0.35 附近。

### 改法

`pipeline.group_channels()`：**同一人的样本先按相似度聚成若干"信道簇"**（默认阈值 0.75、
每人最多 4 簇），匹配时与每个簇质心比对、**取最高分**：

```
刘益清
├─ 簇1（网页麦克风）  质心 A ── 0.91
├─ 簇2（车机）        质心 B ── 0.88   ← 取这个
└─ 簇3（手机）        质心 C ── 0.55
```

- 只有一种设备时退化为 1 簇，行为与原来一致（实测 rematch 分数 0.438，无回归）；
- `detail` 里新增 `channels` 字段，可看出某人被拆成几个信道；
- 配置：`diarization.channel_threshold`（0.75）、`diarization.max_channels`（4），
  设置页「说话人」分组可调。

### 声纹库页面（v0.32.1）

`/voices` 的「声纹库管理」表新增 **信道** 列：

```
姓名    样本  信道                创建时间
刘益清   5    3 簇  1:3 2:1 3:1   2026-10-31…
李彦昆   4    单信道  1:4          2026-10-31…
```

- `N 簇` 表示这个人的样本按设备分成了几组，后面是每组条数；
- 展开「样本」可以看到每条样本的**簇号**、来源、质量、时间；
- 后端 `/api/voices` 增加 `channels` / `channel_sizes`，`/api/voices/{sid}/samples`
  每条增加 `channel`（1 起）。

这样能直观看出：某人是否只在一种设备上录过（单信道），以及缺哪类设备的样本。

### 使用建议

- 每个常用设备各录 1–2 段，比单一设备录 10 条有用；
- 最好的样本是**真实会议录音回写**（任务页「声纹入库」），环境和实际使用一致；
- 跨设备样本多、仍频繁认不出时，把认名阈值降到 0.30、margin 0.05。

## 5.19 逐字稿拆分（v0.33.0）

之前编辑区只有「合并下一段」，遇到**一句里有两个说话人**（重叠、抢话）无法拆开，
只能整段改判给一个人。现在每段后面多了「拆分」。

### 交互

点「拆分」后在弹窗里选切点：

- 自动给出 **2–3 个候选**：从词级时间戳里找停顿最大的位置（≥60ms），
  正好对应"两个人说话的缝隙"；
- 也可以手动填毫秒（默认段中点）。

拆完是两段，可以分别「改说话人」，再点「重跑」重建整理稿。

### 切点怎么切文字

`textproc.split_by_time()`：

1. **有词级时间戳**（Paraformer）：按词切，左边词 `end ≤ 切点`，右边其余——
   文字和音频严格对齐；
2. **没有词级时间**（SenseVoice 等）：按时间比例切字符，并尽量落在标点上。

### 后端

- `PATCH /api/jobs/{jid}/segments/{idx}` 新增 `{"split_at_ms": 12345}`；
- `db.split_segment()` 在 `UNIQUE(job_id, idx)` 约束下先整体后移 10 万再回填，
  避免插入时撞号（跟删除段落用的是同一套技巧）；
- 新段继承原段的说话人、质量、重叠、情绪等字段，`words` 也按切点拆开；
- 返回 `{"ok":true,"new_idx":N,"left":...,"right":...}`。

### 附带

任务详情页的标签页支持深链：`/job/<id>?track=final` 直接打开逐字稿，
`para / final / clean / raw` 四个值都行（切换时也会写回 URL）。

实测：把 18.2 秒的「/sil我有六只小猪」在 76.2s 处拆成两段后，再点「合并下一段」
可以完全还原——拆分是可逆的。

## 6. 测试

```bash
PYTHONPATH=code .venv-funasr/bin/python -m unittest discover -s tests -v
```

当前 22 个用例，覆盖：填充词规则、重复清理、质量标记、时间戳守卫、发言块合并、热词拼音匹配、聚类与簇合并。

---

## 7. 已知限制（如实记录）

| 项 | 现状 |
|---|---|
| **说话人聚类** | 真实远场录音的 CAM++ 向量偏弱，自动聚类容易偏少或偏多。**建议用 `--speakers N` 指定人数**。阈值需用带标注的基准集校准（目前只有 2 段片段验证）。 |
| **重叠检测** | 自适应阈值（P3 + MAD + 下限），真实录音上仍会有偏差；仅作提示，不阻断。 |
| **断点续跑** | 已实现（阶段状态 + ASR 逐窗缓存 + 说话人缓存 + job 目录 flock）；纠错/润色/纪要阶段未缓存，重跑会重新调 LLM。 |
| **Obsidian 归档** | 已实现（frontmatter/标签/附件/SRT/原子写/幂等）；**智能纪要区仍是占位**（未接 LLM 摘要）。 |
| **声纹库** | 已实现注册（上传音频）+ 认名 + 删除；**尚无"从任务一键登记"与多样本管理**。 |
| **OpenAI 兼容 API** | 已实现 `/v1/models`、`/v1/audio/transcriptions`（5 种格式）、202 异步、鉴权。已用官方 `openai` SDK 实测通过（json/verbose_json/srt 三种格式）。 |
| **润色覆盖率** | 96%（22/23）；失败的块走"单段重试"，仍失败则回退规则清理稿并再过一遍去口水/ITN 兜底。 |
| **SenseVoice 词级精度** | 该引擎不返回 token 时间，只能字符均摊。遇到 54 秒长句时会生成 269 个均匀分布的词（`approx=1`），仅适合粗定位。**要精确词级请用 Paraformer**（默认引擎）。 |
| **纠错** | 两种模式：`block`（段落级时间）/ `segment`（逐句精确）。逐句模式文本质量弱于块模式，建议校对场景才用。 |

---

## 8. 与设计文档的差异

| 设计 | 当前实现 | 原因 |
|---|---|---|
| 模块拆成 20+ 文件 | 收敛为 9 个模块 | MVP 优先跑通；边界已按设计划分，可平滑拆分 |
| 分窗用独立 VAD 模型 | 用 ffmpeg `silencedetect` 找切点 | 少加载一个模型、少一次全量读取；VAD 方案在 5h 音频上更精确，待替换 |
| 纠错按 seg_id 逐句 | 按块整段重写后按长度比例回填 | 先验证效果；严格版需要模型输出 seg_id 映射 |
| 声纹库 + 认名 | 未实现 | 下一阶段 |

---

## 9. 目录

```
app/                 # 应用代码（9 个模块）
config/              # 配置
tests/               # 单元测试
input/               # 待处理音频（队列模式）
work/                # 中间产物
output/              # 最终逐字稿
data/app.db          # SQLite（任务/段落/热词/事件）
docs/                # 产品与技术文档
```
