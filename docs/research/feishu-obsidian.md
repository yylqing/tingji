> 调研时间 2026-10-30｜方法：飞书帮助中心为前端渲染，直抓只得标题，正文系解析页面内嵌 JSON 所得（属官方正文）；Obsidian 取官方帮助文档。未获官方说明的条目标注"未确认"。

# 调研原始材料 · 飞书妙记功能拆解 & Obsidian 本地集成

用途：为 [PRD](../02-PRD.md) 的竞品对标与 [Obsidian 输出规范](../02-PRD.md#7-obsidian-输出规范) 提供依据。

---

## A. 飞书妙记（Feishu Minutes）

### A.1 生成与导入

- 四种入口：录制会议、录音、上传本地文件、导入飞书云端文件。[1]
- 大小/时长限制：
  - 本地文件 ≤ **4GB**；会议录制 ≤ 16 小时；录音 ≤ 4 小时。[1]
  - 开放平台接口口径：≤ **6GB** 且 ≤ **6 小时**。[2]
  - ⚠️ 两套口径并存（网页端 vs 开放平台），对外宣传时以网页端为准。
- 支持格式（接口口径）：音频 `wav/mp3/m4a/aac/ogg/wma/amr`；视频 `avi/wmv/mov/mp4/m4v/mpeg/ogg/flv`。[2]
  - 网页端的格式清单未见官方说明（**未确认**）。
- 额度：基础版有转写额度；商业/企业版不限时长，仅受企业存储限制。[1]

### A.2 转写速度与准确率

- **转写速度：官方无公开口径（未确认）**。转写为异步任务，开放接口需轮询就绪状态。[2]
- 准确率：官方内容页称"支持 19 种语言互译，识别准确率高达 **98%**"——属营销口径，非第三方评测。[3]

> 对本项目的含义：**不要拿"98%"作为对标基线**，它没有评测方法支撑。我们的验收应基于自建基准集的 CER。

### A.3 说话人分离与命名

- 自动区分说话人；会议室共用设备可自动拆分。
- 个人设备多人共用时，需使用"**重新识别说话人**"功能（创建后 30 天内、每人仅可拆一次）。
- 默认命名"说话人 1/2/3"，支持手动改名。[4]

> 对本项目的含义：妙记的"重新识别"是一次性、受限的人工修正；我们的**注册式声纹库**在固定团队场景是结构性优势（一次注册、长期认名）。

### A.4 逐字稿与时间轴

- 点击任意文字定位视频；点击发言人时间轴定位其发言；搜索高亮定位。
- 支持倍速、CC 字幕、跳过空白、片段分享。[5]

### A.5 关键词与高亮

- 选中文字可高亮；关键词可定位。
- **识别说话人期间会锁定编辑与导出**。[4][5]

> 对本项目的含义：编辑态与处理态需互斥——我们的做法是任务 `running` 时逐字稿只读，`done` 后可编辑。

### A.6 AI 智能纪要

智能总结、章节、AI 待办（可增删改，@负责人是纯文本字段）、关键词；需智能纪要权限与额度。[6]

### A.7 多语言

录音实时转写中/英/日等并可翻译；官方称可听懂中英混说、方言词。[1][3]

### A.8 导出

飞书文档 / TXT / SRT，可选"包含说话人""包含时间戳"；受导出权限控制。[3][5]

### A.9 协作

选中片段评论、@、片段分享；权限分可阅读/可编辑/可管理，支持组织外分享与加密链接。[7]

### A.10 生态联动

生成智能纪要文档（飞书文档）、待办可同步飞书任务、日程会议录制归日程组织者、会议信息含相关文档链接。[1][3][5]

### A.11 离线

**未确认存在离线模式**；上传—转写—查看全在云端。

> 这是本项目最硬的差异化点：**全链路离线**是妙记结构上做不到的。

### A.12 竞品横向

| 产品 | 关键事实 | 与本项目关系 |
|---|---|---|
| 通义听悟 | 实时记录免费 48h/天、上传 2h/天（90 天试用）；ASR 0.6 元/小时；章节速览/摘要/待办/关键词/思维导图；含说话人分离；支持中英自由说。[8] | 云端，额度制；功能最接近妙记 |
| 腾讯会议 | 转写绑定云录制，需专业版及以上；仅中文普通话，不支持纯英文与方言；支持关键词定位、重新识别发言人、导出逐字稿。[9] | 场景绑定会议，不适合离线归档 |
| Notta | 免费 120 分钟/月、单次 3 分钟、50 次上传、10 次 AI 摘要；Pro 1800 分钟/月；多语种 + 说话人识别。[10] | 海外服务，合规不适用 |
| Otter.ai | Basic 300 分钟/月、3 次（终身）文件导入；Pro 单场 90 分钟、Business 4 小时；英文为主，集成 Zoom/Teams/Meet。[11] | 英文场景，不适用 |

---

## B. Obsidian 本地集成

### B.1 语法要素

| 要素 | 语法 | 来源 |
|---|---|---|
| frontmatter（properties） | 首行 `---` 包裹 YAML；类型含 text / list / number / checkbox / date / datetime / tags；**内部链接作为值需加引号** | [12] |
| 内部链接 | `[[note]]`、`[[note#标题]]`、`[[note#^block]]` | [14] |
| 嵌入 | 前面加 `!`，音频 `![[file.mp3]]` | [13] |
| Callout | `> [!note] 标题`；折叠写 `[!tip]-` | [15] |
| Dataview | ` ```dataview TABLE ... FROM "目录" WHERE type = "..." ``` ` | [16] |
| Templater | `<% tp.date.now("YYYY-MM-DD") %>` | [17] |
| Tasks | ` ```tasks not done due before tomorrow ``` ` | [18] |

### B.2 外部程序写入最佳实践

- **命名**：避开 `# | ^ : %% [[ ]]`（官方明确会破坏链接）；建议 `YYYY-MM-DD-主题.md`。[14]
- **附件**：Settings → Files & Links → 新附件默认位置（根/指定目录/同笔记目录/子目录）。[19]
- **同步冲突**：Obsidian Sync 对 md 用 diff-match-patch 自动合并，其他文件后写覆盖；1.9.7+ 可改为生成 `原名 (Conflicted copy 设备名 时间).md`；**勿与第三方网盘双同步**。[20]
- **写入**：先写临时文件再原子 rename，避免半写被索引。
- **索引**：桌面端监听文件系统，保存后自动出现；异常时重载；`obsidian://open?vault=X&file=Y` 可直开笔记（macOS/Windows 首启即注册协议）。[21]

### B.3 未安装 Obsidian 的影响

- **不影响 Markdown 生成**——纯文本，任意编辑器可读写。
- 但 **Dataview / Tasks / Templater、callout 渲染、音频嵌入播放** 必须装 Obsidian 才能呈现。

> 对本项目的含义：把"无插件硬依赖"写进 PRD（已写入 §7.6 设计要求）——笔记在纯文本状态下也要完整可读。

### B.4 会议卡片常见实践

```yaml
type: meeting
date: 2026-10-30
participants: ["[[张三]]"]
source: feishu-minutes
duration: 00:42:10
tags: [meeting/transcript]
audio: "[[attachments/2026-10-30-周会.m4a]]"
```

**音频时间戳**：Markdown 本身不支持 seek，替代方案：

1. 插件把时间戳变成可点链接，如 `Audio Timestamp Player` 的 `[00:13](#audio-time=00:13)`（多音频加 `&audio=part1`）。[22]
2. `Media Extended` 的 `?t=` 参数。
3. 轻量做法：时间戳旁加 `[[note#^block]]` 指向对应段落（不定位音频，但可定位文本）。

---

## C. 对本项目的 4 条直接结论

1. **功能对标**：妙记的"总结/章节/待办/关键词/导出（含说话人与时间戳）/逐字稿联动"必须全覆盖，否则用户没有迁移理由。
2. **差异化**：离线 + 0 边际成本 + Obsidian 原生归档（妙记做不到）。
3. **不要抄**：云端协作、评论、分享链接——本地单机做这些只会增加复杂度。
4. **Obsidian 侧必须零插件可用**：`Dataview`/`Tasks` 是加分项，不能是前置条件；未装 Obsidian 也应能产出可读 Markdown。

---

## 来源

[1] [生成妙记（大小/时长/入口）](https://www.feishu.cn/hc/zh-CN/articles/386045971891)
[2] [lark-cli minutes upload（接口口径 6GB/6h 与格式）](https://github.com/larksuite/cli/blob/main/skills/lark-meeting/references/lark-minutes-upload.md)
[3] [飞书内容页（19 语言/98% 准确率）](https://www.feishu.cn/content/article/7578773484596153570)
[4] [重新识别说话人](https://www.feishu.cn/hc/zh-CN/articles/812241214493)
[5] [查看妙记（联动/倍速/CC/片段）](https://www.feishu.cn/hc/zh-CN/articles/022111234449)
[6] [minutes detail（智能纪要字段）](https://github.com/larksuite/cli/blob/main/skills/lark-meeting/references/lark-minutes-detail.md)
[7] [妙记权限](https://www.feishu.cn/hc/zh-CN/articles/821155971912)
[8] [通义听悟计费](https://help.aliyun.com/zh/tingwu/pricing-and-billing-rules)
[9] [腾讯会议录制转写](https://cloud.tencent.com/document/product/1095/53483)
[10] [Notta Pricing](https://www.notta.ai/en/pricing)
[11] [Otter Pricing](https://otter.ai/pricing)
[12] [Obsidian Properties](https://help.obsidian.md/properties)
[13] [Obsidian Embeds](https://help.obsidian.md/embeds)
[14] [Obsidian Internal links](https://help.obsidian.md/links)
[15] [Obsidian Callouts](https://help.obsidian.md/callouts)
[16] [Dataview Query Structure](https://blacksmithgu.github.io/obsidian-dataview/queries/structure/)
[17] [Templater](https://silentvoid13.github.io/Templater/)
[18] [Tasks](https://publish.obsidian.md/tasks/)
[19] [Obsidian Attachments](https://help.obsidian.md/attachments)
[20] [Obsidian Sync 冲突处理](https://help.obsidian.md/sync/troubleshoot)
[21] [Obsidian URI](https://help.obsidian.md/uri)
[22] [Audio Timestamp Player](https://github.com/itnq18/obsidian-audio-timestamp-player)
