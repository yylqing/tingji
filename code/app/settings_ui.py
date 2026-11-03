"""设置页的可编辑 schema：字段、类型、取值范围与说明。

只暴露"改了有意义、且不会把服务弄挂"的项；改完写 `config/overrides.yaml`，
并热更新进程内的 Config，无需重启（`server.*` 除外，页面会提示重启）。
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

SCHEMA: list[dict] = [
    {"group": "识别", "items": [
        {"key": "asr.provider", "label": "识别引擎", "type": "select",
         "options": [["auto", "自动（会议→Paraformer，叙事→Nano）"],
                     ["funasr", "Paraformer（会议 / 精确词级）"],
                     ["sensevoice", "SenseVoice（叙事 / 情绪）"],
                     ["funasr_nano", "Fun-ASR-Nano（自带标点 / 更慢）"]],
         "help": "auto：≥20 分钟或连续语音多→会议；否则按叙事走 Nano"},
        {"key": "asr.model", "label": "Paraformer 模型", "type": "text"},
        {"key": "asr.window_speech_sec", "label": "分窗时长（秒）", "type": "number",
         "min": 60, "max": 3600, "step": 30, "help": "默认 1500（25 分钟/窗）"},
        {"key": "asr.min_speech_ratio", "label": "静音门控", "type": "number",
         "min": 0, "max": 1, "step": 0.005,
         "help": "有效语音占比低于此值直接跳过识别（默认 0.02）"},
    ]},
    {"group": "说话人", "items": [
        {"key": "diarization.enabled", "label": "启用说话人分离", "type": "bool"},
        {"key": "diarization.backend", "label": "分离方式", "type": "select",
         "options": [["sherpa", "帧级（pyannote，推荐）"], ["embedding", "声纹聚类"]],
         "help": "改了需重跑任务才生效"},
        {"key": "diarization.max_speakers", "label": "指定人数", "type": "number",
         "min": 0, "max": 20, "step": 1, "help": "0 = 自动；已知人数时填 2–10 更稳"},
        {"key": "diarization.max_auto_speakers", "label": "自动人数上限", "type": "number",
         "min": 1, "max": 20, "step": 1},
        {"key": "diarization.mask_non_speech", "label": "分离前掩蔽非语音", "type": "bool",
         "help": "把 VAD 判为非语音的部分置零，说话人更稳（保留轻声兜底）"},
        {"key": "diarization.embed_window_sec", "label": "声纹窗口（秒）", "type": "number",
         "min": 5, "max": 30, "step": 1, "help": "每个 turn 按此长度切窗提声纹，默认 15"},
        {"key": "diarization.channel_threshold", "label": "信道聚类阈值", "type": "number",
         "min": 0.5, "max": 0.95, "step": 0.01,
         "help": "同一人的样本按设备分簇：相似度高于此值算同一信道"},
        {"key": "diarization.max_channels", "label": "每人群数上限", "type": "number",
         "min": 1, "max": 8, "step": 1, "help": "每人最多保留几个设备/信道簇，默认 4"},
        {"key": "diarization.match_threshold", "label": "认名阈值", "type": "number",
         "min": 0, "max": 1, "step": 0.01,
         "help": "同信道可 0.6+；跨信道（网页录 vs 车内录）实测 0.35–0.45"},
        {"key": "diarization.match_margin", "label": "领先幅度", "type": "number",
         "min": 0, "max": 1, "step": 0.01},
    ]},
    {"group": "文本后处理", "items": [
        {"key": "text.filler_removal", "label": "去口语填充词", "type": "bool"},
        {"key": "text.repeat_removal", "label": "去重复/结巴", "type": "bool"},
        {"key": "text.itn", "label": "数字规范化（ITN）", "type": "bool",
         "help": "三百毫秒→300毫秒；十五六个→15~16个"},
        {"key": "text.rules_enabled", "label": "正则纠错规则", "type": "bool",
         "help": "config/rules.txt：单位/邮箱/固定错法"},
        {"key": "correction.enabled", "label": "启用逐句 LLM 纠错", "type": "bool",
         "help": "更准但更慢；一般用「整理稿」就够了"},
        {"key": "correction.mode", "label": "纠错粒度", "type": "select",
         "options": [["block", "按块（可读优先）"], ["segment", "逐句（时间戳精确）"]]},
    ]},
    {"group": "整理稿与纪要", "items": [
        {"key": "refine.enabled", "label": "生成整理稿", "type": "bool"},
        {"key": "refine.style", "label": "文风", "type": "select",
         "options": [["auto", "自动判断"], ["meeting", "会议书面语"], ["narrative", "个人叙事（保留情绪）"]]},
        {"key": "summary.enabled", "label": "生成智能纪要", "type": "bool",
         "help": "摘要 + 章节 + 待办"},
        {"key": "roles.enabled", "label": "推断说话人角色", "type": "bool"},
        {"key": "emotion.enabled", "label": "情绪/事件识别", "type": "bool"},
        {"key": "hotwords.enabled", "label": "启用热词", "type": "bool"},
    ]},
    {"group": "LLM", "items": [
        {"key": "llm.base_url", "label": "Ollama 地址", "type": "text",
         "help": "默认 http://127.0.0.1:11434"},
        {"key": "refine.model", "label": "整理稿模型", "type": "text"},
        {"key": "summary.model", "label": "纪要模型", "type": "text"},
        {"key": "roles.model", "label": "角色模型", "type": "text"},
    ]},
    {"group": "Obsidian", "items": [
        {"key": "obsidian.vault_path", "label": "仓库地址", "type": "text", "wide": True,
         "placeholder": "/Users/you/Documents/Obsidian/MyVault",
         "help": "填 Obsidian 仓库根目录的绝对路径；留空则不归档"},
        {"key": "obsidian.notes_by_type", "label": "按类型分目录", "type": "bool",
         "help": "会议纪要 / 日常对话 / 重要时刻 … 各自一个子目录"},
        {"key": "obsidian.notes_dir", "label": "笔记根目录", "type": "text"},
        {"key": "obsidian.transcript_dir", "label": "转写稿目录", "type": "text"},
        {"key": "obsidian.attachment_dir", "label": "附件目录", "type": "text"},
        {"key": "obsidian.keep_source", "label": "归档后保留源文件", "type": "bool"},
    ]},
    {"group": "接口与服务", "items": [
        {"key": "openai_api.enabled", "label": "开启 OpenAI 兼容接口", "type": "bool"},
        {"key": "openai_api.rate_limit_per_min", "label": "限流（次/分钟）", "type": "number",
         "min": 0, "max": 10000, "step": 10, "help": "0 = 不限流"},
        {"key": "notify.enabled", "label": "任务完成通知", "type": "bool"},
        {"key": "notify.webhook_url", "label": "Webhook 地址", "type": "text", "wide": True},
        {"key": "server.bind", "label": "监听地址", "type": "text", "restart": True},
        {"key": "server.port", "label": "端口", "type": "number", "min": 1, "max": 65535,
         "step": 1, "restart": True},
    ]},
]

RESTART_KEYS = {i["key"] for g in SCHEMA for i in g["items"] if i.get("restart")}
_INDEX = {i["key"]: i for g in SCHEMA for i in g["items"]}


def _get(cfg, key: str):
    group, name = key.split(".", 1)
    return getattr(getattr(cfg, group), name)


def values_of(cfg) -> dict:
    return {k: _get(cfg, k) for k in _INDEX}


def _coerce(item: dict, raw):
    t = item["type"]
    if t == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() in ("1", "true", "yes", "on")
    if t == "number":
        val = float(raw)
        if "min" in item and val < item["min"]:
            raise ValueError(f"不能小于 {item['min']}")
        if "max" in item and val > item["max"]:
            raise ValueError(f"不能大于 {item['max']}")
        return int(val) if float(val).is_integer() and item.get("step", 1) >= 1 else val
    if t == "select":
        allowed = [o[0] for o in item.get("options", [])]
        if raw not in allowed:
            raise ValueError(f"只能是 {allowed}")
        return raw
    return str(raw)


def validate(payload: dict) -> tuple[dict, dict]:
    """返回 (要保存的嵌套 dict, 每字段错误)。未知 key 直接忽略。"""
    out: dict = {}
    errors: dict = {}
    for key, raw in (payload or {}).items():
        item = _INDEX.get(key)
        if not item:
            continue
        try:
            val = _coerce(item, raw)
        except Exception as e:
            errors[key] = str(e)
            continue
        group, name = key.split(".", 1)
        out.setdefault(group, {})[name] = val
    return out, errors


def check_vault(path: str) -> tuple[bool, str]:
    """检查 Obsidian 仓库地址；存在且可写才算通过。"""
    p = Path(path).expanduser()
    if not str(path).strip():
        return False, "留空表示不归档"
    if not p.exists():
        return False, f"目录不存在：{p}"
    if not p.is_dir():
        return False, f"不是目录：{p}"
    try:
        probe = p / ".tingji-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as e:
        return False, f"不可写：{e}"
    return True, str(p)


def reload_into(cfg, path=None) -> None:
    """把最新配置热替换进进程内的 Config 对象（后续任务立即生效）。"""
    from .settings import load_config
    fresh = load_config(path)
    for f in dataclasses.fields(cfg):
        setattr(cfg, f.name, getattr(fresh, f.name))
