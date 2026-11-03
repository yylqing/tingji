"""配置加载：YAML → dataclass，带默认值与相对路径解析。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 目录结构：项目根下只有 code/ docs/ input/ output/（外加隐藏的虚拟环境与缓存）
CODE_ROOT = Path(__file__).resolve().parent.parent   # …/code
ROOT = CODE_ROOT.parent                              # 项目根
# 模型缓存必须落在项目内（沙箱/权限受限环境下 ~/.cache 不可写）
os.environ.setdefault("MODELSCOPE_CACHE", str(ROOT / ".cache" / "modelscope"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))


@dataclass
class AppCfg:
    data_dir: str = "./output/.runtime/data"
    work_dir: str = "./output/.runtime/work"
    input_dir: str = "./input"
    output_dir: str = "./output"
    log_level: str = "INFO"
    max_concurrency: int = 1


@dataclass
class AudioCfg:
    target_sr: int = 16000
    loudness_normalize: bool = True
    preview_m4a: bool = True


@dataclass
class AsrCfg:
    provider: str = "auto"   # auto|funasr|sensevoice|funasr_nano
    model: str = "paraformer-zh"
    device: str = "cpu"
    window_speech_sec: int = 1500
    min_speech_ratio: float = 0.02   # 低于此比例视为无人声，跳过识别
    batch_size_s: int = 300


@dataclass
class DiarCfg:
    enabled: bool = True
    campplus_model: str = ".cache/sherpa/campplus.onnx"
    min_cluster_segment_sec: float = 1.0
    cluster_threshold: float = 0.40
    merge_threshold: float = 0.85
    backend: str = "sherpa"  # sherpa=帧级分离 | embedding=声纹聚类
    max_speakers: int = 0   # 0=自动；指定后按人数强制切分
    max_auto_speakers: int = 8  # 自动模式下允许的最大人数（超过则按质心合并）
    match_threshold: float = 0.35   # 认名声纹相似度阈值（跨信道 0.35~0.45）
    match_margin: float = 0.08      # 与第二名的领先幅度
    channel_threshold: float = 0.75  # 同一人样本按信道聚类：相似度超过此值归为一簇
    max_channels: int = 4            # 每人最多保留几个信道簇
    smooth_min_run: int = 1  # 标签平滑：≤N 的孤立 run 合并
    resplit: bool = True     # 跨说话人重切
    resplit_max: int = 40
    mask_non_speech: bool = True      # 分离前把非语音置零（Brevia 做法）
    embed_window_sec: float = 15.0    # 每个 turn 按 15 秒窗口提声纹
    overlap_detect: bool = True
    overlap_threshold: float = 0.62
    overlap_floor: float = 0.45


@dataclass
class TextCfg:
    filler_removal: bool = True
    repeat_removal: bool = True
    itn: bool = True
    rules_enabled: bool = True
    rules_file: str = "config/rules.txt"


@dataclass
class HotwordCfg:
    enabled: bool = True
    l1_max: int = 50
    l2_threshold: float = 0.85
    l2_min_len: int = 2


@dataclass
class CorrectionCfg:
    enabled: bool = False
    mode: str = "block"            # block=可读优先（默认）| segment=逐句严格（时间戳精确）
    model: str = "qwen3.5:9b-q4_K_M"
    batch_chars: int = 700
    selective: bool = True
    max_length_ratio: float = 0.6


@dataclass
class RefineCfg:
    enabled: bool = True
    model: str = "qwen3.5:9b-q4_K_M"
    max_gap_ms: int = 2500
    sad_gap_ms: int = 9000       # 哭泣/难过时的长停顿上限
    max_chars: int = 320
    style: str = "auto"          # auto | meeting | narrative
    max_blocks: int = 0          # 0=全部


@dataclass
class EmotionCfg:
    enabled: bool = True
    device: str = "cpu"


@dataclass
class RolesCfg:
    enabled: bool = True
    model: str = "qwen3.5:9b-q4_K_M"
    max_input_chars: int = 4000


@dataclass
class SummaryCfg:
    enabled: bool = True
    model: str = "qwen3.5:9b-q4_K_M"
    max_input_chars: int = 6000


@dataclass
class LlmCfg:
    base_url: str = "http://127.0.0.1:11434"
    timeout_sec: int = 300
    keep_alive: str = "10m"


@dataclass
class OpenAIApiCfg:
    enabled: bool = True
    sync_max_audio_sec: int = 1800
    rate_limit_per_min: int = 0   # 0=不限流；>0 时按 IP 限制每分钟请求数
    default_model: str = "tingji-asr"
    max_upload_gb: float = 4.0
    api_key: str = ""              # 留空且只绑回环时不校验


@dataclass
class ObsidianCfg:
    vault_path: str = ""                 # 留空则不归档
    notes_dir: str = "录音笔记"
    notes_by_type: bool = True             # 按内容类型分子目录（会议纪要/日常对话/重要时刻…）
    per_note_folder: bool = True           # 每条录音一个文件夹（md + 音频 + 字幕）
    transcript_dir: str = "转写稿"
    attachment_dir: str = "attachments/audio"
    title_prefix: str = ""
    keep_source: bool = True             # 归档后是否保留 input 中的源文件


@dataclass
class NotifyCfg:
    enabled: bool = False
    webhook_url: str = ""


@dataclass
class ServerCfg:
    bind: str = "0.0.0.0"      # 默认监听所有网卡（局域网可访问）
    port: int = 8000
    token: str = ""            # 访问令牌；留空则启动时自动生成并写入 config/access_token.txt


@dataclass
class Config:
    app: AppCfg = field(default_factory=AppCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    asr: AsrCfg = field(default_factory=AsrCfg)
    diarization: DiarCfg = field(default_factory=DiarCfg)
    text: TextCfg = field(default_factory=TextCfg)
    hotwords: HotwordCfg = field(default_factory=HotwordCfg)
    correction: CorrectionCfg = field(default_factory=CorrectionCfg)
    refine: RefineCfg = field(default_factory=RefineCfg)
    summary: SummaryCfg = field(default_factory=SummaryCfg)
    roles: RolesCfg = field(default_factory=RolesCfg)
    emotion: EmotionCfg = field(default_factory=EmotionCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    obsidian: ObsidianCfg = field(default_factory=ObsidianCfg)
    openai_api: OpenAIApiCfg = field(default_factory=OpenAIApiCfg)
    server: ServerCfg = field(default_factory=ServerCfg)
    notify: NotifyCfg = field(default_factory=NotifyCfg)

    @property
    def data_path(self) -> Path:
        return _abs(self.app.data_dir)

    @property
    def work_path(self) -> Path:
        return _abs(self.app.work_dir)

    @property
    def input_path(self) -> Path:
        return _abs(self.app.input_dir)

    @property
    def output_path(self) -> Path:
        return _abs(self.app.output_dir)

    @property
    def db_path(self) -> Path:
        return self.data_path / "app.db"


def _abs(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _fill(dc: Any, data: dict) -> Any:
    for f in dc.__dataclass_fields__.values():
        if f.name in data:
            setattr(dc, f.name, data[f.name])
    return dc


OVERRIDES_PATH = CODE_ROOT / "config" / "overrides.yaml"


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base or {})
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def save_overrides(values: dict) -> Path:
    """把 Web 上改的值写到 config/overrides.yaml（不动 config.yaml，保留注释）。"""
    cur = {}
    if OVERRIDES_PATH.exists():
        try:
            cur = yaml.safe_load(OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
        except Exception:
            cur = {}
    merged = _deep_merge(cur, values)
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDES_PATH.write_text(
        "# 由 Web 设置页写入，优先级高于 config.yaml；删除本文件即恢复。\n"
        + yaml.safe_dump(merged, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    return OVERRIDES_PATH


def clear_overrides() -> None:
    OVERRIDES_PATH.unlink(missing_ok=True)


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = Config()
    p = Path(path) if path else (CODE_ROOT / "config" / "config.yaml")
    if not p.exists():
        p = CODE_ROOT / "config" / "config.example.yaml"
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    # Web 设置页保存的覆盖项优先（同一份 schema，深度合并）
    if path is None and OVERRIDES_PATH.exists():
        try:
            over = yaml.safe_load(OVERRIDES_PATH.read_text(encoding="utf-8")) or {}
            raw = _deep_merge(raw, over)
        except Exception:
            pass
    for group in ("app", "audio", "asr", "diarization", "text", "hotwords", "correction", "refine", "emotion", "roles", "summary", "llm", "obsidian", "openai_api", "server", "notify"):
        sub = getattr(cfg, group)
        _fill(sub, raw.get(group, {}) or {})
    # 模型路径统一解析成绝对路径：服务的工作目录可能是 code/，相对路径会找不到
    for attr in ("campplus_model", "segmentation_model", "embedding_model"):
        val = getattr(cfg.diarization, attr, "")
        if val:
            setattr(cfg.diarization, attr, str(_abs(val)))
    if getattr(cfg.obsidian, "vault_path", ""):
        cfg.obsidian.vault_path = str(_abs(cfg.obsidian.vault_path))
    for d in (cfg.data_path, cfg.work_path, cfg.input_path, cfg.output_path):
        d.mkdir(parents=True, exist_ok=True)
    return cfg
