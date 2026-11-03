# 调研 · 会议转写产品的后处理链路（顺滑 / 标点 / ITN / 书面化）

> 调研日期 2026-10-31｜目的：解释"为什么我们的转写读起来不像妙记"，并据此调整实现
> 关联：[01 技术选型 §3.15](../01-技术选型报告.md) · [06 开发方案 §4.4](../06-开发方案.md) · [真实会议基线](real-meeting-baseline.md)

---

## 0. TL;DR

1. **中文"顺滑"= 删除型去口语**（语气词/重复/口吃），**不等于整段重写**。讯飞做成 ASR 开关（`has_smooth` 默认开，词级属性 `wp=s` 标顺滑词），腾讯把"口语转书面语"做成**付费档**（`ResTextFormat=5`），Otter 干脆把 filler words **程序化忽略**（不显示）。
2. **标点、ITN、分段、顺滑在工业界是"一次打标多属性"**（讯飞 `wp` 字段同时含 `p` 标点 / `d` 数字 / `s` 顺滑词 / `g` 分段），而不是四个模型串行。
3. **所有厂商都保留逐字稿原稿**，可读稿是另一层（腾讯 `=5` 仍返回基础结果；讯飞"纪要溯源"点回原文；通义"口语书面化"是独立能力项）。**双轨是信任基础，不是可选项。**
4. **FunASR 没有中文"顺滑"模型**，也**没有中文 ITN**（ITN 仅 en/ru/ja/ko/id/vi/tl/es/pt/fr/de）。中文 ITN 要用 WeTextProcessing（Pynini），macaOS/ARM 上编译困难——本次实测**装不上**（缺 OpenFST 头文件）。
5. 学术上 LLM 后编辑的**已知失败模式是"过度删除"与"保真度下降"**，所以工业界普遍用"删除/替换 span"而非自由重写。

---

## 1. 各家链路对照

| 产品 | 独立顺滑 | 标点 | ITN | 口语→书面 | 说话人命名 | 章节 | 摘要输入 |
|---|---|---|---|---|---|---|---|
| 飞书妙记 | 未确认 | △ 可导出带句读 srt/txt | 未确认 | 未确认 | ✓ 分离；命名未确认 | ✓ `chapters` | 逐字稿（反推） |
| 讯飞听见/录音笔 | ✓ `has_smooth` 默认开 | ✓ `wp=p` | ✓ `wp=d/m` | △ 仅纪要侧 | ✓ **声纹库自动命名** | ✓ 议题速览 | **转写原文**（可溯源） |
| 通义听悟 | △ | ✓ | 未确认 | ✓ 独立能力项 | ✓ 不分离/2人/多人 | ✓ 章节速览 | 转写（反推） |
| 腾讯云 ASR | ✓ `=5` | ✓ `=2/3`，`=4` 语义分段 | ✓ `ConvertNumMode` 默认 1 | ✓ `=5`（仍保留逐字稿） | ✓ `SpeakerRoles` | ✓ `=4` | 大模型后处理（付费） |
| Otter | ✓ 语气词程序化忽略 | △ | 未确认 | ✗ | ✓ label+手动命名 | △ 按说话人 | 逐字稿 |
| Google Recorder | 未确认 | 未确认 | 未确认 | ✗ | Pixel 有 speaker labels | 未确认 | 转录 |
| iOS 语音备忘录 | 未确认 | 未确认 | 未确认 | ✗ | iOS 未见分离 | 未确认 | Apple Intelligence 摘要 |

来源见 §5。

**关键证据**

- 讯飞文档：`has_smooth` = "开启或关闭**顺滑词**"，仅中/英支持；结果按句返回 `bg/ed/onebest/speaker`，词级 `wp` 含 `n 普通 / r 人名 / d 数字 / m 量词 / s 顺滑词 / t 地名多音字 / p 标点 / g 分段标识`。
- 腾讯云 `ResTextFormat`：2=带标点词级时间戳、3=按标点分段、4=按 NLP 语义分段、**5=口语转书面语（去语气词、去重复、精简冗余、修正口误）**、7=大模型后处理。4/5 仅 8k_zh/16k_zh 且增值付费。
- Otter FAQ：filler words / interjections / hesitation markers "are programmatically ignored"，即使加进 custom vocabulary 也被忽略。

---

## 2. 学术与工业的三种技术路线

| 路线 | 做法 | 优点 | 失败模式 |
|---|---|---|---|
| 端到端 seq2seq 顺滑 | 音频/文本直接出流利稿 | 无二次对齐损失 | 需配对语料、**难保时间戳**、易过度删除 |
| **联合 tagging + WFST**（工业主流） | 一个 transformer 同时打 ITN/标点/大小写/不流畅 tag，再规则落地 | 可解释、保 word 对齐、可控 | 只能局部删改，处理不了长程改写 |
| **LLM 后编辑** | 文档级口语→书面 | 语法性与正式度最好 | **保真度差**：reasoning 模型系统性过度删除；聚合指标掩盖局部失败 |

**评价方式**：没有统一的"顺滑后 CER"。主流是 **verbatim（逐字）与 intended（意图稿）双参考分开报**——混用会奖励"乱删"的系统。删除质量用 span 级 F1 按 EDITED/INTJ/PRN 分型统计。

---

## 3. 我们踩的坑 vs 商业做法

| 我们的现状（73 分钟会议） | 商业做法 | 差距 |
|---|---|---|
| 逗号密度 **6.2/百字**（正常散文 2–3） | 按语义断句 + `SentenceMaxLength` 控制（腾讯 6–40 字） | CT-Punc 按长段打标 → 逗号成灾 |
| **163 处**单字被标点包夹（"这，个"） | 词级标点属性 + 分段融合 | 我们缺"句边界融合" |
| 填充词 880+ 次（嗯154/呃82/啊115/那个255/就是188/然后89） | 删除型顺滑默认开 | 我们的规则层只删句首尾 |
| 句长中位 **1.6s**、1750 段 | 段落化 + 章节 | 缺"段落"这一层 |
| 无 ITN（"三百毫秒"保持汉字） | 默认开 `ConvertNumMode` | FunASR 无中文 ITN |
| 摘要缺失 | 摘要是标配 | 已补（见下） |

---

## 4. 据此做的实现调整（本轮已落地）

| # | 调整 | 依据 |
|---|---|---|
| 1 | **新增"整理稿"轨**（段落级，LLM 顺滑），逐字稿原样保留 | 双轨是三家共同做法；删除/轻改保可溯源 |
| 2 | 段落按 **speaker turn + 时间间隔 + 320 字** 切分，前后带只读上下文 | 调研 D3：按 turn 切 + 重叠上下文防指代断裂 |
| 3 | 润色输出 **JSON 批量对齐**（4 段/次），失败块回退原文 | 降低长音频 LLM 成本；逐块校验可回滚 |
| 4 | **硬闸门**：长度比 >1.25 拒绝、数字集合不得新增、不许出现时间戳、剥掉解释性括号 | 对应"保真度是 LLM 后编辑的短板" |
| 5 | **智能纪要**：摘要 + 3–6 章节（带时间）+ 决议/待办/关键词，写入 Obsidian | 妙记/听悟/腾讯都做；章节粒度直接影响可用性 |
| 6 | 声纹认名（已有）保持 | 讯飞/腾讯都做声纹命名 |

**效果（6 分钟真实会议片段）**

| 指标 | 逐字稿 | 整理稿 |
|---|---|---|
| 字数 | 2,180 | 1,586（−27%） |
| 逗号密度 | 4.4/百字 | **3.4/百字** |
| 单字被标点包夹 | 14 处 | **2 处** |
| 嗯/呃/那个 | 14 / 22 / 19 | **0 / 0 / 6** |
| LLM 润色覆盖率 | — | **22/23 段（96%）** |

---

## 5. 未采用 / 未解决的

| 项 | 原因 | 后续 |
|---|---|---|
| 中文 ITN（WeTextProcessing） | Pynini 需 OpenFST 头文件，macOS ARM 装不上 | **已改为自写保守规则**（`app/itn.py`）：单位白名单 + 小数/时刻消歧 + 大数 万/亿 格式化 |
| FunASR 顺滑模型 | 官方 model zoo 里**没有**；网传的 `ct-transformer 顺滑` ID 多路检索无果 | 不引用该说法；用我们的规则 + LLM 段落顺滑替代 |
| 词级主表（word_id） | 本轮用段落级；word 级需重构 segments 表 | P2：配合"编辑操作而非全文"的改写方式 |
| 章节摘要的 LLM 忠实度 | 未做人工评测 | 用双参考（verbatim/intended）思路建评测集 |

---

## 6. 来源

- 讯飞转写 API（`has_smooth`、`wp` 词属性）— https://www.xfyun.cn/doc/asr/lfasr/API.html
- 腾讯云 CreateRecTask（`ResTextFormat` 2/3/4/5/7、`ConvertNumMode`、`SentenceMaxLength`）— https://cloud.tencent.com/document/product/1093/37823
- 通义听悟功能特性（口语书面化为独立能力项）— https://help.aliyun.com/zh/tingwu/features
- 飞书妙记 transcript 接口（`need_speaker`/`need_timestamp`/txt|srt）— https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/minutes-v1/minute-transcript/get.md
- Otter FAQ（filler words 程序化忽略）— https://help.otter.ai/hc/en-us/articles/360048322533-Speech-transcription-accuracy-FAQ
- FunASR model zoo（无中文 ITN、无顺滑模型）— https://github.com/modelscope/FunASR/blob/main/model_zoo/readme_zh.md
- CT-Punc 模型卡（只补标点、不删改、无时间戳）— https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary
- WeTextProcessing（中文 TN/ITN，Pynini/OpenFst）— https://github.com/wenet-e2e/WeTextProcessing
- Four-in-One（标点/ITN/大小写/不流畅联合打标 + WFST）— https://arxiv.org/abs/2210.15063
- CoS2W / SWAB（LLM 口语→书面，保真度短板）— https://arxiv.org/abs/2408.09688
- 过度删除的实证 — https://arxiv.org/abs/2509.20321 ；span 级评测 — https://arxiv.org/abs/2509.20319
- 双参考（verbatim vs intended）评测 — https://arxiv.org/abs/2606.31112
- E2E 顺滑 vs 两级 pipeline — https://arxiv.org/abs/2009.10298 ；https://arxiv.org/abs/2309.04516
- AASR-Bench（rubric 基准，较新）— https://arxiv.org/abs/2607.28175
