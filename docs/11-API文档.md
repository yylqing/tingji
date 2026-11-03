# 听记 · API 文档

> 适用版本：v0.33.x　｜　最后更新：2026-11-03

服务启动后同时暴露两套接口：

1. **OpenAI 兼容接口** `/v1/*` —— 给 Memos、Open WebUI、任意 Whisper 客户端用；
2. **内部 REST API** `/api/*` —— Web 页面用，也可脚本调用。

鉴权、地址、错误格式见文末。

---

## 1. OpenAI 兼容接口

### 1.1 `GET /v1/models`

返回可用模型列表：

```json
{"object":"list","data":[
  {"id":"tingji-asr","object":"model","owned_by":"tingji"},
  {"id":"tingji-asr-hq","object":"model","owned_by":"tingji"},
  {"id":"whisper-1","object":"model","owned_by":"tingji"}
]}
```

| 模型 | 行为 |
|---|---|
| `tingji-asr` / `whisper-1` | 快速转写，只返回文本 |
| `tingji-asr-hq` | 额外做说话人分离，正文按「说话人N：」分行 |

### 1.2 `POST /v1/audio/transcriptions`

`multipart/form-data`：

| 字段 | 必填 | 说明 |
|---|---|---|
| `file` | ✅ | 音频文件，支持 wav/mp3/m4a/aac/flac/webm/opus/ogg/3gp…（后端 ffmpeg 解码） |
| `model` | | 默认 `tingji-asr`；`tingji-asr-hq` 带说话人 |
| `language` | | 语言代码（目前仅透传，识别以中文为主） |
| `prompt` | | 热词提示，会拆成词条作为**本次请求**的热词 |
| `response_format` | | `json`（默认）/ `text` / `srt` / `vtt` / `verbose_json` |
| `temperature` | | OpenAI 兼容占位，忽略 |

示例：

```bash
curl -H "Authorization: Bearer $KEY" \
  -F file=@会议.m4a \
  -F model=tingji-asr-hq \
  -F language=zh \
  -F "prompt=人名：张三、李四。术语：交付、预算。" \
  http://127.0.0.1:8000/v1/audio/transcriptions
```

`json` 返回：

```json
{"text":"说话人1：今天先对一下预算。\n说话人2：好的，我这边…",
 "x_tingji":{"model":"tingji-asr-hq","elapsed_sec":18.6,"hotwords":4,
             "speakers":["说话人1","说话人2"]}}
```

`verbose_json` 额外返回：

```json
{"task":"transcribe","language":"zh","duration":46.1,"text":"…",
 "segments":[{"id":0,"start":0.05,"end":3.5,"text":"…","speaker":"说话人1"}],
 "x_tingji":{"model":"tingji-asr-hq","elapsed_sec":…,"hotwords":4,"speakers":[…]}}
```

`text` 直接返回纯文本；`srt` / `vtt` 返回字幕（带 `[说话人]` 前缀）。

> **说话人只来自 `model=tingji-asr-hq`**（或任何以 `-hq` 结尾的名字）。
> 用 `tingji-asr` / `whisper-1` 时不会做分离，这是有意的——快模型不跑分离，省 3–5 秒。

**长音频（异步）**：超过 `openai_api.sync_max_audio_sec`（默认 1800 秒）返回：

```
HTTP 202
Location: /v1/audio/transcriptions/{job_id}
{"id":"20261103-xxxx","object":"transcription.job","status":"queued","estimated_sec":210}
```

随后轮询 `GET /v1/audio/transcriptions/{job_id}`：

```json
{"id":"…","status":"done","text":"…","segments":[{"start":0.0,"end":2.1,"text":"…"}]}
```

进行中返回 `status/progress/stage`；`DELETE` 同路径可取消。

完成后（v0.33.4+）：

```json
{"id":"…","status":"done",
 "text":"张三：我们先过一下进展。\n李四：风控那边有个依赖…",
 "segments":[{"start":0.05,"end":3.5,"text":"…","spk":0,"speaker":"张三"}],
 "x_tingji":{"speakers":["张三","李四"]}}
```

- `segments[].speaker` 与 `text` 都可直接使用；说话人来自流水线的分离结果，
  匹配到声纹库则显示真名，否则为「说话人N」；
- 短碎片（<350ms 且 ≤6 字）会并入相邻同一人的轮次，避免 A/B/A 抖动。

---

## 2. 内部 REST API

所有接口在 `/api/` 下，需带访问令牌（见 §3）。

### 2.1 任务

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/jobs?archived=0` | 任务列表；`archived=1` 含已归档 |
| POST | `/api/jobs` | 上传音频入队，form-data `file`；返回 `{"id","queued"}` |
| GET | `/api/jobs/{jid}` | 任务详情：`job / segments / paragraphs / summary / export` |
| DELETE | `/api/jobs/{jid}?keep_archive=0` | 删除任务及输出（Obsidian 笔记不动） |
| POST | `/api/jobs/{jid}/archive` | body `{"archived":true|false}` |
| POST | `/api/jobs/{jid}/rerun` | 重跑（`{"force":true}` 全量） |
| POST | `/api/jobs/{jid}/rebuild` | 只重建整理稿/纪要 |
| POST | `/api/jobs/{jid}/rematch` | 用声纹库重新认名（不重转写） |
| POST | `/api/jobs/{jid}/enroll` | 把任务里的说话人样本入库（body 可选 `speaker`） |
| GET | `/api/jobs/{jid}/stages` | 流水线各阶段状态与耗时 |
| GET | `/api/jobs/{jid}/events` | 事件日志（分页 `limit/offset`） |
| GET | `/api/jobs/{jid}/export?fmt=srt` | 导出 `srt/vtt/txt/json` |
| GET | `/api/audio/{jid}` | 试听音频（Range 支持） |

### 2.2 段落校对

`PATCH /api/jobs/{jid}/segments/{idx}`，body 四选一：

```json
{"text": "改后的文字"}
{"speaker": 1}
{"merge_next": true}
{"split_at_ms": 76200}
{"delete": true}
```

- `split_at_ms` 必须在该段 `[start_ms, end_ms]` 之间；有词级时间戳时按词切，
  否则按时间比例切；返回 `{"ok":true,"new_idx":N,"left":…,"right":…}`。
- 改动后建议调用 `/rebuild` 重建整理稿。

### 2.3 说话人校对（抽样标注）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/jobs/{jid}/labelqueue?per_cluster=3` | 取待标注片段 |
| POST | `/api/jobs/{jid}/labels` | 提交标注 `{"items":[{"idx":1,"label":"张三"}]}` |
| DELETE | `/api/jobs/{jid}/labels` | 清空标注 |
| GET | `/api/jobs/{jid}/accuracy` | 标注准确率统计 |
| GET | `/api/jobs/{jid}/mixed` | 疑似多人混说的片段 |

### 2.4 声纹库

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/voices` | 人物列表（含 `samples`、`channels`、`channel_sizes`） |
| POST | `/api/voices` | 登记：form-data `name` + `files`（1–4 段音频） |
| PATCH | `/api/voices/{sid}` | 改名 `{"name":"…"}` |
| DELETE | `/api/voices/{sid}` | 删除人物及其全部样本 |
| GET | `/api/voices/{sid}/samples` | 样本列表（含 `channel`、`quality`、`source`、`created_at`） |
| POST | `/api/voices/{sid}/samples` | 追加样本：form-data `files` |
| DELETE | `/api/voices/{sid}/samples/{vid}` | 删除单条样本 |
| GET | `/api/voices/export` | 导出全部声纹 |

### 2.5 热词

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/hotwords` | 列表（`term/weight/type/aliases`） |
| POST | `/api/hotwords` | 新增/更新 `{"term":"张三","weight":1.0,"type":"term"}` |
| DELETE | `/api/hotwords/{term}` | 删除 |

### 2.6 设置

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/settings` | 当前值 + 字段定义（分组、类型、范围） |
| PATCH | `/api/settings` | `{"values":{"diarization.match_threshold":0.3}}` |
| POST | `/api/settings/reset` | 清空 `overrides.yaml`，回到 config.yaml |
| POST | `/api/settings/check-vault` | `{"path":"/…"}` 校验 Obsidian vault |

`server.*` 改动需重启服务，其余对新任务立即生效。

### 2.7 其它

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查（免令牌）`{"ok":true,"jobs":N,"voices":M}` |
| GET | `/api/version` | 版本、启动时间、代码 mtime、是否 stale |
| GET | `/api/status` | 队列、监听目录、磁盘、内存等运行状态 |
| GET | `/api/logs?limit=200&level=error&job=` | 事件日志过滤 |

---

## 3. 鉴权

| 接口 | 鉴权方式 |
|---|---|
| `/v1/*` | `Authorization: Bearer <openai_api.api_key>`；`api_key` 为空则不校验 |
| `/api/*`、页面 | 访问令牌：`?token=` / `Cookie: tingji_token` / `Authorization: Bearer` |
| `/api/health`、`/static/*`、`/login` | 免鉴权 |

- 访问令牌在 `code/config/access_token.txt`，首次启动生成；
- 服务绑定 `127.0.0.1` 时**不启用**令牌（本机免登录）；
- 绑定 `0.0.0.0` 时强制令牌，浏览器首次带 `?token=` 会写入 cookie；
- `/v1/*` 不受网页令牌影响，避免客户端收到 302。

## 4. 错误格式

```json
{"detail":"错误说明"}                      // /api/*（FastAPI 默认）
{"error":{"message":"unsupported format .webm（ffmpeg 无法解码）"}}   // /v1/*
```

常见状态码：`400` 参数/解码错误，`401` 鉴权失败，`404` 不存在，`413` 文件过大，
`429` 触发限流（`openai_api.rate_limit_per_min`），`202` 异步任务已入队。

## 5. 限流

仅对 `/v1/*` 生效，按来源 IP、每分钟计数；`openai_api.rate_limit_per_min: 0` 表示不限。
超限返回 `429 {"detail":"rate limited"}`。
