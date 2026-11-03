"""SQLite 存储层：建表、任务队列、逐字稿、热词。"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  source_name TEXT NOT NULL,
  source_path TEXT NOT NULL,
  sha256 TEXT UNIQUE,
  size_bytes INTEGER,
  duration_sec REAL,
  status TEXT NOT NULL DEFAULT 'queued',
  stage TEXT,
  progress INTEGER DEFAULT 0,
  message TEXT,
  error_code TEXT,
  degraded INTEGER DEFAULT 0,
  archived INTEGER DEFAULT 0,
  created_at TEXT, started_at TEXT, finished_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs(status);

CREATE TABLE IF NOT EXISTS segments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  start_ms INTEGER NOT NULL,
  end_ms INTEGER NOT NULL,
  raw_text TEXT NOT NULL,
  clean_text TEXT,
  final_text TEXT,
  spk INTEGER,
  quality TEXT DEFAULT 'ok',
  char_rate REAL,
  overlap INTEGER DEFAULT 0,
  UNIQUE(job_id, idx)
);

CREATE TABLE IF NOT EXISTS summaries (
  job_id TEXT PRIMARY KEY,
  json TEXT, model TEXT, created_at TEXT
);

CREATE TABLE IF NOT EXISTS paragraphs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  idx INTEGER NOT NULL,
  start_ms INTEGER, end_ms INTEGER,
  spk INTEGER, text TEXT, polished INTEGER DEFAULT 0,
  UNIQUE(job_id, idx)
);

CREATE TABLE IF NOT EXISTS hotwords (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  term TEXT NOT NULL UNIQUE,
  type TEXT DEFAULT 'term',
  aliases TEXT DEFAULT '[]',
  weight REAL DEFAULT 1.0,
  enabled INTEGER DEFAULT 1,
  note TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS speakers (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS voiceprints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  speaker_id TEXT NOT NULL,
  vector BLOB NOT NULL,
  dim INTEGER NOT NULL,
  source TEXT,
  quality REAL,
  created_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_vp_speaker ON voiceprints(speaker_id);

CREATE TABLE IF NOT EXISTS exports (
  job_id TEXT PRIMARY KEY,
  note TEXT, transcript TEXT, audio TEXT, srt TEXT, tags TEXT, created_at TEXT
);

CREATE TABLE IF NOT EXISTS seg_labels (
  job_id TEXT NOT NULL,
  seg_idx INTEGER NOT NULL,
  label TEXT NOT NULL,
  created_at TEXT,
  PRIMARY KEY(job_id, seg_idx)
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT, ts TEXT, level TEXT, stage TEXT, message TEXT, payload TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_job ON events(job_id, ts);
"""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def new_id() -> str:
    return time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:8]


class DB:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.executescript(SCHEMA)
            cols = {r["name"] for r in c.execute("PRAGMA table_info(segments)")}
            for col in ("emotion", "event", "speaker_kind", "spk_purity", "words", "boundary"):
                if col not in cols:
                    c.execute(f"ALTER TABLE segments ADD COLUMN {col} TEXT")
            # 老库迁移：jobs.archived
            cols = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
            if "archived" not in cols:
                c.execute("ALTER TABLE jobs ADD COLUMN archived INTEGER DEFAULT 0")

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ---------- jobs ----------
    def create_job(self, source_name: str, source_path: str, sha256: str | None,
                   size: int, duration: float) -> str:
        jid = new_id()
        with self.conn() as c:
            c.execute(
                "INSERT INTO jobs(id,source_name,source_path,sha256,size_bytes,duration_sec,status,created_at)"
                " VALUES(?,?,?,?,?,?, 'queued', ?)",
                (jid, source_name, source_path, sha256, size, duration, now()),
            )
        return jid

    def find_by_sha(self, sha: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE sha256=?", (sha,)).fetchone()
            return dict(r) if r else None

    def update_job(self, jid: str, **kw: Any) -> None:
        if not kw:
            return
        cols = ",".join(f"{k}=?" for k in kw)
        with self.conn() as c:
            c.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*kw.values(), jid))

    def get_job(self, jid: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
            return dict(r) if r else None

    def list_jobs(self, limit: int = 100, include_archived: bool = False) -> list[dict]:
        with self.conn() as c:
            if include_archived:
                rows = c.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                                 (limit,)).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM jobs WHERE COALESCE(archived,0)=0 "
                    "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def set_archived(self, job_id: str, archived: bool = True) -> None:
        with self.conn() as c:
            c.execute("UPDATE jobs SET archived=? WHERE id=?", (1 if archived else 0, job_id))

    def next_queued(self) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not r:
                return None
            c.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (now(), r["id"]))
            return dict(r)

    # ---------- segments ----------
    def save_segments(self, job_id: str, segs: list[dict]) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM segments WHERE job_id=?", (job_id,))
            c.executemany(
                "INSERT INTO segments(job_id,idx,start_ms,end_ms,raw_text,clean_text,final_text,spk,quality,char_rate,overlap,emotion,event,speaker_kind,spk_purity,words,boundary)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (job_id, i, s["start_ms"], s["end_ms"], s["raw_text"], s.get("clean_text"),
                     s.get("final_text"), s.get("spk"), s.get("quality", "ok"),
                     s.get("char_rate"), int(s.get("overlap", 0)),
                     s.get("emotion"), s.get("event"), s.get("speaker_kind"),
                     s.get("spk_purity"),
                     json.dumps(s["words"], ensure_ascii=False) if s.get("words") else None,
                     s.get("boundary"))
                    for i, s in enumerate(segs)
                ],
            )

    def _renumber_after(self, c, job_id: str, idx: int) -> None:
        """删除后把 idx>idx 的段整体前移（避开 UNIQUE(job_id, idx) 冲突）。"""
        c.execute("UPDATE segments SET idx = idx + 100000 WHERE job_id=? AND idx>?", (job_id, idx))
        c.execute("UPDATE segments SET idx = idx - 100001 WHERE job_id=? AND idx>?", (job_id, idx))

    def delete_segment(self, job_id: str, idx: int) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM segments WHERE job_id=? AND idx=?", (job_id, idx))
            self._renumber_after(c, job_id, idx)

    def merge_segment_next(self, job_id: str, idx: int) -> bool:
        with self.conn() as c:
            cur = c.execute("SELECT * FROM segments WHERE job_id=? AND idx=?", (job_id, idx)).fetchone()
            nxt = c.execute("SELECT * FROM segments WHERE job_id=? AND idx=?", (job_id, idx + 1)).fetchone()
            if not cur or not nxt:
                return False
            txt = (cur["final_text"] or cur["clean_text"] or cur["raw_text"] or "") + \
                  (nxt["final_text"] or nxt["clean_text"] or nxt["raw_text"] or "")
            raw = (cur["raw_text"] or "") + (nxt["raw_text"] or "")
            c.execute("UPDATE segments SET end_ms=?, raw_text=?, final_text=? WHERE job_id=? AND idx=?",
                      (nxt["end_ms"], raw, txt, job_id, idx))
            c.execute("DELETE FROM segments WHERE job_id=? AND idx=?", (job_id, idx + 1))
            self._renumber_after(c, job_id, idx + 1)
            return True

    def split_segment(self, job_id: str, idx: int, at_ms: int,
                      left_text: str, right_text: str,
                      left_words: list | None = None, right_words: list | None = None) -> int:
        """把第 idx 段在 at_ms 处拆成两段，返回新段号（idx+1）。"""
        import json as _json
        with self.conn() as c:
            cur = c.execute("SELECT * FROM segments WHERE job_id=? AND idx=?",
                            (job_id, idx)).fetchone()
            if not cur:
                raise ValueError("段不存在")
            if not (int(cur["start_ms"]) < at_ms < int(cur["end_ms"])):
                raise ValueError("拆分点必须在该段内部")
            # 给后面的段腾位置（先整体后移，避免 UNIQUE(job_id,idx) 冲突）
            c.execute("UPDATE segments SET idx=idx+100000 WHERE job_id=? AND idx>?", (job_id, idx))
            c.execute("UPDATE segments SET idx=idx-99999 WHERE job_id=? AND idx>?", (job_id, idx + 100000))
            c.execute("UPDATE segments SET end_ms=?, final_text=?, raw_text=?, words=? "
                      "WHERE job_id=? AND idx=?",
                      (at_ms, left_text, left_text,
                       _json.dumps(left_words, ensure_ascii=False) if left_words else cur["words"],
                       job_id, idx))
            c.execute(
                "INSERT INTO segments(job_id,idx,start_ms,end_ms,raw_text,clean_text,final_text,spk,"
                "quality,char_rate,overlap,emotion,event,speaker_kind,spk_purity,words,boundary)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, idx + 1, at_ms, int(cur["end_ms"]), right_text, right_text, right_text,
                 cur["spk"], cur["quality"], None, cur["overlap"], cur["emotion"], cur["event"],
                 cur["speaker_kind"], cur["spk_purity"],
                 _json.dumps(right_words, ensure_ascii=False) if right_words else None,
                 cur["boundary"]))
            return idx + 1

    def delete_job(self, job_id: str) -> None:
        with self.conn() as c:
            for t in ("segments", "paragraphs", "summaries", "events", "exports", "seg_labels"):
                try:
                    c.execute(f"DELETE FROM {t} WHERE job_id=?", (job_id,))
                except Exception:
                    pass
            c.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def update_segment(self, job_id: str, idx: int, **kw: Any) -> None:
        cols = ",".join(f"{k}=?" for k in kw)
        with self.conn() as c:
            c.execute(f"UPDATE segments SET {cols} WHERE job_id=? AND idx=?", (*kw.values(), job_id, idx))

    def get_segments(self, job_id: str) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM segments WHERE job_id=? ORDER BY idx", (job_id,)).fetchall()
            return [dict(r) for r in rows]

    # ---------- hotwords ----------
    # ---------- 人工校对标签 ----------
    def save_seg_labels(self, job_id: str, items: list[dict]) -> None:
        with self.conn() as c:
            c.executemany(
                "INSERT INTO seg_labels(job_id,seg_idx,label,created_at) VALUES(?,?,?,?)"
                " ON CONFLICT(job_id,seg_idx) DO UPDATE SET label=excluded.label, created_at=excluded.created_at",
                [(job_id, int(x["idx"]), str(x["label"]), now()) for x in items],
            )

    def get_seg_labels(self, job_id: str) -> dict[int, str]:
        with self.conn() as c:
            rows = c.execute("SELECT seg_idx,label FROM seg_labels WHERE job_id=?", (job_id,)).fetchall()
            return {int(r["seg_idx"]): r["label"] for r in rows}

    def clear_seg_labels(self, job_id: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM seg_labels WHERE job_id=?", (job_id,))

    def save_summary(self, job_id: str, data: dict) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO summaries(job_id,json,model,created_at) VALUES(?,?,?,?)"
                      " ON CONFLICT(job_id) DO UPDATE SET json=excluded.json, model=excluded.model",
                      (job_id, json.dumps(data, ensure_ascii=False), data.get("model"), now()))

    def get_summary(self, job_id: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM summaries WHERE job_id=?", (job_id,)).fetchone()
            if not r:
                return None
            return json.loads(r["json"])

    def save_paragraphs(self, job_id: str, rows: list[dict]) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM paragraphs WHERE job_id=?", (job_id,))
            c.executemany(
                "INSERT INTO paragraphs(job_id,idx,start_ms,end_ms,spk,text,polished) VALUES(?,?,?,?,?,?,?)",
                [(job_id, r["idx"], r["start_ms"], r["end_ms"], r.get("spk"),
                  r["text"], int(r.get("polished", 0))) for r in rows],
            )

    def get_paragraphs(self, job_id: str) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM paragraphs WHERE job_id=? ORDER BY idx", (job_id,)).fetchall()
            return [dict(r) for r in rows]

    def list_hotwords(self, only_enabled: bool = False) -> list[dict]:
        q = "SELECT * FROM hotwords" + (" WHERE enabled=1" if only_enabled else "") + " ORDER BY weight DESC, id"
        with self.conn() as c:
            rows = c.execute(q).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["aliases"] = json.loads(d["aliases"] or "[]")
            out.append(d)
        return out

    def upsert_hotword(self, term: str, type_: str = "term", aliases: list[str] | None = None,
                       weight: float = 1.0, note: str = "") -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO hotwords(term,type,aliases,weight,note,created_at) VALUES(?,?,?,?,?,?)"
                " ON CONFLICT(term) DO UPDATE SET type=excluded.type, aliases=excluded.aliases,"
                " weight=excluded.weight, note=excluded.note",
                (term, type_, json.dumps(aliases or [], ensure_ascii=False), weight, note, now()),
            )

    def delete_hotword(self, term: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM hotwords WHERE term=?", (term,))

    # ---------- 声纹 ----------
    def list_speakers(self) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM speakers ORDER BY created_at").fetchall()
            out = []
            for r in rows:
                d = dict(r)
                n = c.execute("SELECT COUNT(*) n FROM voiceprints WHERE speaker_id=?", (d["id"],)).fetchone()["n"]
                d["samples"] = n
                out.append(d)
            return out

    def create_speaker(self, name: str) -> str:
        sid = "spk_" + uuid.uuid4().hex[:10]
        with self.conn() as c:
            c.execute("INSERT INTO speakers(id,name,created_at) VALUES(?,?,?)", (sid, name, now()))
        return sid

    def get_speaker_by_name(self, name: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM speakers WHERE name=?", (name,)).fetchone()
            return dict(r) if r else None

    def delete_speaker(self, sid: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM voiceprints WHERE speaker_id=?", (sid,))
            c.execute("DELETE FROM speakers WHERE id=?", (sid,))

    def rename_speaker(self, sid: str, name: str) -> None:
        with self.conn() as c:
            c.execute("UPDATE speakers SET name=? WHERE id=?", (name, sid))

    def get_speaker(self, sid: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM speakers WHERE id=?", (sid,)).fetchone()
            return dict(r) if r else None

    def voiceprint_vectors(self, sid: str) -> list[dict]:
        """返回 (id, vector)，按 id 排序；vector 只在服务端用于计算，不下发前端。"""
        with self.conn() as c:
            rows = c.execute("SELECT id, vector FROM voiceprints WHERE speaker_id=? ORDER BY id",
                             (sid,)).fetchall()
            return [dict(r) for r in rows]

    def list_voiceprints(self, sid: str) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT id,source,quality,created_at,dim FROM voiceprints"
                " WHERE speaker_id=? ORDER BY id", (sid,)).fetchall()
            return [dict(r) for r in rows]

    def delete_voiceprint(self, vid: int) -> bool:
        with self.conn() as c:
            cur = c.execute("DELETE FROM voiceprints WHERE id=?", (vid,))
            return cur.rowcount > 0

    def count_voiceprints(self, sid: str) -> int:
        with self.conn() as c:
            return c.execute("SELECT COUNT(*) n FROM voiceprints WHERE speaker_id=?",
                             (sid,)).fetchone()["n"]

    def speaker_vectors(self, sid: str) -> list[bytes]:
        with self.conn() as c:
            return [r["vector"] for r in c.execute(
                "SELECT vector FROM voiceprints WHERE speaker_id=? ORDER BY id", (sid,))]

    def add_voiceprint(self, speaker_id: str, vector: bytes, dim: int,
                       source: str = "", quality: float = 1.0) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO voiceprints(speaker_id,vector,dim,source,quality,created_at)"
                      " VALUES(?,?,?,?,?,?)", (speaker_id, vector, dim, source, quality, now()))

    def voiceprint_signature(self) -> str:
        """声纹库指纹（条数+最新时间）：用于判断认名缓存是否过期。"""
        with self.conn() as c:
            r = c.execute("SELECT COUNT(*) n, COALESCE(MAX(created_at),'') t FROM voiceprints").fetchone()
            return f"{r['n']}:{r['t']}"

    def all_voiceprints(self) -> list[dict]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT v.vector, v.dim, s.id sid, s.name FROM voiceprints v JOIN speakers s ON s.id=v.speaker_id"
            ).fetchall()
            return [dict(r) for r in rows]

    def save_export(self, job_id: str, info: dict) -> None:
        with self.conn() as c:
            c.execute(
                "INSERT INTO exports(job_id,note,transcript,audio,srt,tags,created_at) VALUES(?,?,?,?,?,?,?)"
                " ON CONFLICT(job_id) DO UPDATE SET note=excluded.note, transcript=excluded.transcript,"
                " audio=excluded.audio, srt=excluded.srt, tags=excluded.tags",
                (job_id, info.get("note"), info.get("transcript"), info.get("audio"),
                 info.get("srt"), json.dumps(info.get("tags") or [], ensure_ascii=False), now()),
            )

    def get_export(self, job_id: str) -> dict | None:
        with self.conn() as c:
            r = c.execute("SELECT * FROM exports WHERE job_id=?", (job_id,)).fetchone()
            return dict(r) if r else None

    # ---------- events ----------
    def log(self, job_id: str | None, level: str, stage: str, message: str, payload: Any = None) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO events(job_id,ts,level,stage,message,payload) VALUES(?,?,?,?,?,?)",
                      (job_id, now(), level, stage, message,
                       json.dumps(payload, ensure_ascii=False) if payload is not None else None))
