# 听记 · 文档索引

本地离线中文会议转写与纪要系统。项目入口见 [根目录 README](../README.md)。

> 当前版本：**v0.33.0**（2026-11-03）｜ 状态：可日常使用

---

## 文档导航

| 文档 | 内容 | 适合谁 |
|---|---|---|
| [10-使用手册.md](10-使用手册.md) | 安装、Web 界面、声纹/热词/设置、Obsidian 归档、故障排查 | **使用者** |
| [11-API文档.md](11-API文档.md) | OpenAI 兼容接口 + 内部 REST API + 鉴权 | 集成方 / 脚本调用 |
| [01-技术选型报告.md](01-技术选型报告.md) | 为什么选 FunASR / sherpa-onnx / Ollama，含本机实测 | 想了解选型与数据 |
| [02-PRD.md](02-PRD.md) | 需求与验收标准（FR-A…FR-I） | 产品 / 验收 |
| [03-实施计划.md](03-实施计划.md) | M0–M5 里程碑 | 项目管理 |
| [04-重叠说话与声纹登记设计.md](04-重叠说话与声纹登记设计.md) | 重叠处理、声纹登记句设计 | 深入说话人 |
| [05-热词登记设计.md](05-热词登记设计.md) | 热词两级匹配 | 提高专名准确率 |
| [06-开发方案.md](06-开发方案.md) | 架构、模块契约、状态机 | 开发者 |
| [07-需求对账.md](07-需求对账.md) | 需求 ↔ 实现逐条对账 | 验收 |
| [08-界面设计规范.md](08-界面设计规范.md) | Memphis 设计令牌（颜色/字体/组件） | 改 UI 的人 |
| [DEVELOPMENT.md](DEVELOPMENT.md) | 开发说明：流水线细节、版本历史、踩坑记录 | 开发者 |
| [../CHANGELOG.md](../CHANGELOG.md) | 版本演进 | 所有人 |
| [research/](research/) | 同类项目调研（Meetily / CapsWriter / Brevia / WhisprRT / MOSS） | 找灵感 |
| [templates/](templates/) | Obsidian 笔记模板（会议纪要/看板/示例） | Obsidian 用户 |

---

## 30 秒了解

```
input/会议.m4a
   ├─ 归一化 16k 单声道
   ├─ 转写（Paraformer / SenseVoice / Fun-ASR-Nano，按场景自动选）
   ├─ 说话人分离（pyannote 分割 + CAM++ 声纹，帧级）
   ├─ 声纹认名（按设备聚类，跨设备也能认）
   ├─ 清理（口水词/重复/ITN/规则）
   ├─ 整理稿（LLM 分段润色）+ 智能纪要（摘要/章节/决议/待办）
   └─ 归档 Obsidian（笔记 + 音频 + 字幕；逐字稿单独存）
```

词级时间戳、逐句校对、拆分/合并、导出 SRT/VTT、OpenAI 兼容接口都具备。

## 本机实测（Apple M4 / 16GB）

| 项 | 结果 |
|---|---|
| 转写 + 标点 + 分离（40s 音频） | 1.86s（约 21× 实时） |
| 73 分钟会议端到端 | 341–401s（约 12× 实时），峰值内存 4.9GB |
| 2 小时压力测试 | 793s，峰值 3.35GB |
| 本地 LLM 纪要（qwen3.5:9b） | 15.5 tok/s |

## 目录结构

```
tingji/
├── code/       源码（app/ config/ scripts/ tests/ bench/）
├── docs/       本目录
├── input/      待处理音频
└── output/     产出 + .runtime/{data,work,logs,voiceprint}
```

## 快速开始

```bash
PYTHONPATH=code .venv-funasr/bin/python -m app.cli service --port 8000
```

详细步骤见 [使用手册](10-使用手册.md)。

## 约定

- **配置分层**：`config/overrides.yaml`（Web 写入） > `config/config.yaml` > 代码默认；
- **路径基准**：代码里 `CODE_ROOT` = `code/`，`ROOT` = 项目根；配置里的相对路径按 `ROOT` 解析；
- **测试**：`PYTHONPATH=code .venv-funasr/bin/python -m unittest discover -s code/tests`（130+ 用例）；
- **版本号**：`code/app/__init__.py`，Web 页头与 `/api/version` 同步显示。
