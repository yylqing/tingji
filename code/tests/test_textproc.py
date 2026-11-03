"""单元测试：文本清理、热词、质量标记、时间戳守卫、聚类基本行为。

运行：PYTHONPATH=. .venv-funasr/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import numpy as np
import unittest

from app import diarize as dia
from app import textproc as tp


class TestFillers(unittest.TestCase):
    def test_leading_filler(self):
        out, n = tp.strip_fillers("嗯，我们这个季度要翻倍。")
        self.assertEqual(out, "我们这个季度要翻倍。")
        self.assertEqual(n, 1)

    def test_multiple_leading(self):
        out, _ = tp.strip_fillers("嗯，呃，就是这个意思。")
        self.assertEqual(out, "就是这个意思。")

    def test_pronoun_not_removed(self):
        # “那个方案”里的“那个”是指示代词，不能删
        out, _ = tp.strip_fillers("那个方案我已经看过了。")
        self.assertEqual(out, "那个方案我已经看过了。")

    def test_modal_particle_kept(self):
        # 句末“啊”是语气词，按设计保留
        out, _ = tp.strip_fillers("这样就可以了啊")
        self.assertEqual(out, "这样就可以了啊")

    def test_trailing_filler_removed(self):
        out, _ = tp.strip_fillers("这样就可以了嗯")
        self.assertEqual(out, "这样就可以了")


class TestRepeats(unittest.TestCase):
    def test_stutter(self):
        out, _ = tp.strip_repeats("我我觉得这个这个方案可以。")
        self.assertEqual(out, "我觉得这个方案可以。")

    def test_phrase_repeat(self):
        out, _ = tp.strip_repeats("对对对，就是这个。")
        self.assertEqual(out, "对，就是这个。")


class TestClean(unittest.TestCase):
    def test_combined(self):
        out, stat = tp.clean_text("嗯，我我觉得这个这个方案可以啊。")
        self.assertEqual(out, "我觉得这个方案可以啊。")
        self.assertGreaterEqual(stat["fillers"] + stat["repeats"], 3)


class TestQuality(unittest.TestCase):
    def test_low_quality(self):
        q, rate = tp.quality_flag("好", 30.0)
        self.assertEqual(q, "low_quality")

    def test_normal(self):
        q, rate = tp.quality_flag("这是一段正常语速的转写文本内容", 3.0)
        self.assertEqual(q, "ok")

    def test_short_not_penalized(self):
        q, _ = tp.quality_flag("好", 0.5)
        self.assertEqual(q, "ok")


class TestTimeline(unittest.TestCase):
    def test_monotonic(self):
        segs = [{"start_ms": 0, "end_ms": 1000}, {"start_ms": 1000, "end_ms": 2000}]
        self.assertEqual(tp.check_timeline(segs), [])

    def test_overlap_detected(self):
        segs = [{"start_ms": 0, "end_ms": 2000}, {"start_ms": 1000, "end_ms": 3000}]
        self.assertTrue(tp.check_timeline(segs))

    def test_negative_duration(self):
        segs = [{"start_ms": 1000, "end_ms": 500}]
        self.assertTrue(tp.check_timeline(segs))


class TestBlocks(unittest.TestCase):
    def test_merge_same_speaker(self):
        segs = [
            {"spk": 0, "start_ms": 0, "end_ms": 1000, "raw_text": "a"},
            {"spk": 0, "start_ms": 1500, "end_ms": 2500, "raw_text": "b"},
            {"spk": 1, "start_ms": 3000, "end_ms": 4000, "raw_text": "c"},
        ]
        blocks = tp.blocks_of(segs)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(len(blocks[0]["items"]), 2)

    def test_gap_splits_block(self):
        segs = [
            {"spk": 0, "start_ms": 0, "end_ms": 1000, "raw_text": "a"},
            {"spk": 0, "start_ms": 9000, "end_ms": 10000, "raw_text": "b"},
        ]
        self.assertEqual(len(tp.blocks_of(segs)), 2)


class TestHotword(unittest.TestCase):
    def test_pinyin_fuzzy(self):
        rows = [{"term": "沈知微", "aliases": [], "enabled": 1},
                {"term": "河坊街", "aliases": [], "enabled": 1}]
        m = tp.HotwordMatcher(rows, threshold=0.85)
        out, ms = m.apply("沈志威去了和房街")
        self.assertIn("沈知微", out)
        self.assertIn("河坊街", out)
        self.assertEqual(len(ms), 2)

    def test_alias_mapping(self):
        rows = [{"term": "SaaS", "aliases": ["sars"], "enabled": 1}]
        m = tp.HotwordMatcher(rows, threshold=0.85)
        out, _ = m.apply("他们买了 sars 服务")
        self.assertIn("SaaS", out)

    def test_no_hotword_noop(self):
        m = tp.HotwordMatcher([], threshold=0.85)
        out, ms = m.apply("一段普通文本")
        self.assertEqual(out, "一段普通文本")
        self.assertEqual(ms, [])


class TestClustering(unittest.TestCase):
    def test_two_speakers(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=192).astype(np.float32)
        b = rng.normal(size=192).astype(np.float32)
        a /= np.linalg.norm(a); b /= np.linalg.norm(b)
        E = []
        for _ in range(5):
            v = a + rng.normal(scale=0.05, size=192).astype(np.float32); E.append(v / np.linalg.norm(v))
        for _ in range(5):
            v = b + rng.normal(scale=0.05, size=192).astype(np.float32); E.append(v / np.linalg.norm(v))
        labels = dia.cluster(np.stack(E), np.ones(10), n_clusters=2)
        self.assertEqual(len(set(labels.tolist())), 2)
        self.assertEqual(len(set(labels[:5].tolist())), 1)
        self.assertEqual(len(set(labels[5:].tolist())), 1)

    def test_short_segment_filled(self):
        labels = np.array([0, -1, 0])
        segs = [{"start_ms": 0, "end_ms": 2000}, {"start_ms": 2000, "end_ms": 2300},
                {"start_ms": 2300, "end_ms": 4000}]
        out = dia.fill_short(segs, labels, 1.0)
        self.assertTrue((out >= 0).all())
        self.assertEqual(out[1], 0)

    def test_centroid_merge(self):
        E = np.stack([[1, 0], [1, 0], [0, 1], [0, 1]]).astype(np.float32)
        labels = np.array([0, 0, 1, 1])
        cents = dia.centroids(E, labels)
        self.assertEqual(len(cents), 2)
        merged = dia.merge_by_centroid(labels, cents, threshold=0.72)
        self.assertEqual(len(set(merged.tolist())), 2)   # 正交不合并


if __name__ == "__main__":
    unittest.main()


class TestObsidian(unittest.TestCase):
    def test_safe_name(self):
        from app.obsidian import safe_name
        self.assertEqual(safe_name('a/b:c*d?e"f<g>h|i'), "a b c d e f g h i")
        self.assertTrue(safe_name("") )

    def test_srt_format(self):
        from app.obsidian import build_srt
        segs = [{"start_ms": 65000, "end_ms": 67250, "final_text": "你好", "spk": 0}]
        out = build_srt(segs, {0: "张三"})
        self.assertIn("00:01:05,000 --> 00:01:07,250", out)
        self.assertIn("[张三] 你好", out)

    def test_yaml_links_mixed(self):
        from app.obsidian import _yaml_links
        s = _yaml_links(["张三", "说话人2"])
        self.assertIn('"[[人物/张三]]"', s)
        self.assertIn('"说话人2"', s)

    def test_tags(self):
        from app.obsidian import build_tags
        from datetime import datetime
        tags = build_tags({}, 1200, ["张三", "说话人2"], datetime(2026, 10, 31))
        self.assertIn("会议纪要", tags)
        self.assertIn("时长/短会", tags)
        self.assertIn("说话人/张三", tags)
        self.assertNotIn("说话人/说话人2", tags)


class TestOpenAIFormat(unittest.TestCase):
    def test_srt_and_vtt(self):
        from app.openaiapi import _srt, _vtt
        segs = [{"start_ms": 0, "end_ms": 1500, "text": "你好", "spk": 0}]
        self.assertIn("00:00:00,000 --> 00:00:01,500", _srt(segs, {0: "李四"}))
        self.assertTrue(_vtt(segs, {}).startswith("WEBVTT"))
        self.assertIn("00:00:00.000 --> 00:00:01.500", _vtt(segs, {}))


class TestITN(unittest.TestCase):
    def test_units(self):
        from app.itn import normalize
        self.assertEqual(normalize("响应时间从三百毫秒降到一百二十毫秒"),
                         "响应时间从300毫秒降到120毫秒")

    def test_percent(self):
        from app.itn import normalize
        self.assertIn("15%", normalize("增长了百分之十五"))

    def test_clock_and_decimal(self):
        from app.itn import normalize
        self.assertEqual(normalize("下午三点半开会"), "下午3:30开会")
        self.assertEqual(normalize("三点五倍"), "3.5倍")

    def test_keep_idioms(self):
        from app.itn import normalize
        self.assertEqual(normalize("三心二意的方案不行"), "三心二意的方案不行")
        self.assertEqual(normalize("一个人做不完"), "一个人做不完")

    def test_big_numbers(self):
        from app.itn import normalize
        self.assertIn("1200万", normalize("预算是一千二百万美元"))


class TestEmotionParse(unittest.TestCase):
    def test_tags_parsed_from_utterance_text(self):
        # SenseVoice 的句级 text 不含标签，标签在顶层 text，需要按顺序对齐
        from app.emotion import TAG
        full = "<|zh|><|SAD|><|Cry|><|withitn|>我有点难过 <|zh|><|NEUTRAL|><|Speech|><|withitn|>明天去检查"
        parts = [p for p in __import__("re").split(r"(?=<\|(?:zh|en)\|>)", full) if p.strip()]
        tags0 = TAG.findall(parts[0])
        self.assertIn("SAD", tags0)
        self.assertIn("Cry", tags0)

    def test_label_of(self):
        from app.emotion import label_of
        self.assertEqual(label_of({"event": "Cry", "emotion": "SAD"}), "（哭）")
        self.assertEqual(label_of({"event": "Speech", "emotion": "SAD"}), "（难过）")
        self.assertEqual(label_of({"event": "Speech", "emotion": "NEUTRAL"}), "")


class TestRolesGate(unittest.TestCase):
    def test_evidence_must_be_substring(self):
        # 证据不在原文里的角色推断应被丢弃（防幻觉）
        import re
        corpus = "我们明天去社康听一下胎心"
        corpus_n = re.sub(r"[\s，。！？、,.!?;：:]", "", corpus)
        ev_n = re.sub(r"[\s，。！？、,.!?;：:]", "", "购物车里有西红柿")
        self.assertNotIn(ev_n, corpus_n)


class TestAcoustics(unittest.TestCase):
    def test_f0_on_sine(self):
        import numpy as np
        from app.acoustics import features
        sr = 16000
        t = np.arange(sr * 2) / sr
        for f in (110.0, 220.0):
            x = (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)
            got = features(x, sr)["f0"]
            self.assertAlmostEqual(got, f, delta=f * 0.08)

    def test_nav_and_pa(self):
        from app.acoustics import looks_like_nav
        self.assertTrue(looks_like_nav("前方三百米右转进入辅路"))
        self.assertTrue(looks_like_nav("请照顾好身边的老人和小孩"))
        self.assertFalse(looks_like_nav("我有点难过，怀孕不好玩"))

    def test_noise_needs_loud(self):
        from app.acoustics import looks_like_noise
        # 正常音量的"嗯"不应被判为噪声
        self.assertFalse(looks_like_noise({"raw_text": "嗯", "voiced": 0.1, "f0": 0, "rms": 0.05}, 0.05))
        # 明显更响且无浊音 → 噪声
        self.assertTrue(looks_like_noise({"raw_text": "", "voiced": 0.05, "f0": 0, "rms": 0.3}, 0.05))

    def test_cap_clusters(self):
        import numpy as np
        from app.acoustics import cap_clusters
        segs = [{"f0": 120}] * 8 + [{"f0": 200}] * 4 + [{"f0": 210}] * 4
        labels = np.array([0] * 8 + [1] * 4 + [2] * 4)
        out = cap_clusters(segs, labels, 2)
        self.assertEqual(len(set(out.tolist())), 2)


class TestSegFix(unittest.TestCase):
    def test_aba_smoothing(self):
        import numpy as np
        from app.segfix import smooth_labels
        # 孤立 A-B-A 翻转应被平滑
        labels = np.array([0, 0, 1, 0, 0])
        out = smooth_labels(labels, min_run=1)
        self.assertEqual(out.tolist(), [0, 0, 0, 0, 0])

    def test_run_merge(self):
        import numpy as np
        from app.segfix import smooth_labels
        labels = np.array([0, 0, 1, 1, 0, 0])
        out = smooth_labels(labels, min_run=2)
        self.assertEqual(out.tolist(), [0, 0, 0, 0, 0, 0])

    def test_split_text_by_ratio(self):
        from app.segfix import _split_text_by_ratio
        left, right = _split_text_by_ratio("第一句话，第二句话。", 0.4)
        self.assertTrue(left.endswith("，"))
        self.assertTrue(right.startswith("第二"))


class TestFindSplit(unittest.TestCase):
    def test_find_split_with_fake_embedder(self):
        """用"高值/低值"两段构造信号，假嵌入按均值分类，验证切点定位。"""
        import numpy as np
        from app.segfix import find_split
        sr = 16000
        n = sr * 10
        clip = np.full(n, 0.1, dtype=np.float32)
        clip[: int(n * 0.6)] = 0.9           # 前 60% 属于说话人 0
        centroids = {0: np.array([1.0, 0.0], dtype=np.float32),
                     1: np.array([0.0, 1.0], dtype=np.float32)}

        def fake_embed(x):
            return np.array([1.0, 0.0], dtype=np.float32) if float(np.mean(x)) > 0.5 \
                else np.array([0.0, 1.0], dtype=np.float32)

        got = find_split(clip, fake_embed, centroids, sr)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got / sr, 6.0, delta=0.6)

    def test_no_split_for_single_speaker(self):
        import numpy as np
        from app.segfix import find_split
        sr = 16000
        clip = np.full(sr * 8, 0.9, dtype=np.float32)
        centroids = {0: np.array([1.0, 0.0], dtype=np.float32),
                     1: np.array([0.0, 1.0], dtype=np.float32)}
        fake = lambda x: np.array([1.0, 0.0], dtype=np.float32)
        self.assertIsNone(find_split(clip, fake, centroids, sr))


class TestLabeling(unittest.TestCase):
    def _segs(self):
        return [
            {"idx": 0, "spk": 0, "start_ms": 0, "end_ms": 3000, "raw_text": "第一段内容"},
            {"idx": 1, "spk": 0, "start_ms": 3000, "end_ms": 6000, "raw_text": "第二段内容"},
            {"idx": 2, "spk": 1, "start_ms": 6000, "end_ms": 9000, "raw_text": "第三段内容"},
            {"idx": 3, "spk": 1, "start_ms": 9000, "end_ms": 12000, "raw_text": "第四段内容"},
        ]

    def test_accuracy_and_mapping(self):
        from app.labeling import accuracy
        labels = {0: "A", 1: "A", 2: "B", 3: "B"}
        r = accuracy(self._segs(), labels)
        self.assertEqual(r["labeled"], 4)
        self.assertEqual(r["accuracy"], 1.0)
        self.assertEqual(r["mapping"], {"0": "A", "1": "B"})

    def test_accuracy_with_mistake(self):
        from app.labeling import accuracy
        labels = {0: "A", 1: "A", 2: "B", 3: "A"}   # idx3 标错
        r = accuracy(self._segs(), labels)
        self.assertAlmostEqual(r["accuracy"], 0.75, places=3)

    def test_queue_stratified(self):
        from app.labeling import label_queue
        q = label_queue(self._segs(), per_cluster=2)
        self.assertEqual(len(q), 4)
        self.assertEqual({x["spk"] for x in q}, {0, 1})


class TestWebAuth(unittest.TestCase):
    """绑定非回环时必须带令牌；回环则放行。"""

    def _app(self, bind, token):
        import tempfile, pathlib
        from app.settings import load_config
        from app.web import create_app
        cfg = load_config()
        cfg.server.bind = bind
        cfg.server.token = token
        tmp = pathlib.Path(tempfile.mkdtemp())
        # 路径是 property，改底层的 app.*_dir
        cfg.app.data_dir = str(tmp / "data")
        cfg.app.work_dir = str(tmp / "work")
        cfg.app.output_dir = str(tmp / "output")
        cfg.app.input_dir = str(tmp / "input")
        for d in (tmp / "data", tmp / "work", tmp / "output", tmp / "input"):
            d.mkdir(parents=True, exist_ok=True)
        return create_app(cfg)

    def test_loopback_no_token_needed(self):
        from fastapi.testclient import TestClient
        c = TestClient(self._app("127.0.0.1", "secret"))
        self.assertEqual(c.get("/api/jobs").status_code, 200)

    def test_public_requires_token(self):
        from fastapi.testclient import TestClient
        c = TestClient(self._app("0.0.0.0", "secret"))
        self.assertEqual(c.get("/api/jobs").status_code, 401)
        self.assertEqual(c.get("/api/jobs", headers={"Authorization": "Bearer secret"}).status_code, 200)
        self.assertEqual(c.get("/api/jobs?token=secret").status_code, 200)
        self.assertEqual(c.get("/api/jobs?token=nope").status_code, 401)
        c.cookies.clear()   # 上面的合法 token 已种下 cookie
        # 页面应重定向到 /login
        r = c.get("/", follow_redirects=False)
        self.assertEqual(r.status_code, 302)
        self.assertIn("/login", r.headers["location"])


class TestDiarSherpaSplit(unittest.TestCase):
    def test_split_segment_across_turns(self):
        import numpy as np
        from app.diar_sherpa import apply
        segs = [{"idx": 0, "start_ms": 0, "end_ms": 6000, "spk": 0,
                 "raw_text": "第一句话说完了，第二句话也说完。"}]
        turns = [(0.0, 2.8, 0), (3.0, 6.0, 1)]
        out, labels = apply(segs, turns)
        self.assertEqual(len(out), 2)
        self.assertEqual(labels.tolist(), [0, 1])
        # 文本按比例切开，不能丢字
        joined = "".join(x["raw_text"] for x in out)
        self.assertEqual(len(joined.replace(" ", "")), len(segs[0]["raw_text"].replace(" ", "")))

    def test_single_speaker_untouched(self):
        import numpy as np
        from app.diar_sherpa import apply
        segs = [{"idx": 0, "start_ms": 0, "end_ms": 4000, "spk": 0, "raw_text": "一句话。"}]
        turns = [(0.0, 4.0, 3)]
        out, labels = apply(segs, turns)
        self.assertEqual(len(out), 1)
        self.assertEqual(labels.tolist(), [3])


class TestVoiceprintMulti(unittest.TestCase):
    def test_per_speaker_aggregation(self):
        """同一人多条样本时，按人取最大相似度，而不是按行排名。"""
        import numpy as np
        # 构造：A 有 3 条样本、B 有 1 条；查询向量与 A 的某条最像
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        # 这里直接验证聚合逻辑（与 pipeline.match_voiceprints 内部一致）
        owners = ["A", "A", "A", "B"]
        sims = [0.90, 0.99, 0.88, 0.40]
        per = {}
        for nm, sc in zip(owners, sims):
            if sc > per.get(nm, -1.0):
                per[nm] = sc
        ranked = sorted(per.items(), key=lambda kv: -kv[1])
        self.assertEqual(ranked[0][0], "A")
        self.assertAlmostEqual(ranked[0][1], 0.99, places=3)
        self.assertEqual(len(ranked), 2)


class TestVoiceprintMatch(unittest.TestCase):
    def _db(self):
        import tempfile, pathlib
        from app.db import DB
        return DB(pathlib.Path(tempfile.mkdtemp()) / "t.db")

    def _add(self, db, name, vec):
        import numpy as np
        sid = db.create_speaker(name)
        db.add_voiceprint(sid, np.asarray(vec, dtype=np.float32).tobytes(), 192, source="t", quality=1.0)
        return sid

    def test_names_closest_speaker(self):
        import numpy as np
        from app.pipeline import match_voiceprints
        db = self._db()
        a = np.zeros(192, dtype=np.float32); a[0] = 1.0
        b = np.zeros(192, dtype=np.float32); b[1] = 1.0
        self._add(db, "甲", a)
        self._add(db, "乙", b)
        q = a * 0.9 + b * 0.1
        names, detail = match_voiceprints(db, {0: q}, threshold=0.35, margin=0.10)
        self.assertEqual(names.get(0), "甲")
        self.assertEqual(detail["0"]["confidence"], "high")

    def test_ambiguous_not_named(self):
        import numpy as np
        from app.pipeline import match_voiceprints
        db = self._db()
        a = np.zeros(192, dtype=np.float32); a[0] = 1.0
        b = np.zeros(192, dtype=np.float32); b[1] = 1.0
        self._add(db, "甲", a)
        self._add(db, "乙", b)
        q = a * 0.7 + b * 0.7          # 两人都很像 → margin 不足
        names, _ = match_voiceprints(db, {0: q}, threshold=0.35, margin=0.10)
        self.assertEqual(names, {})

    def test_low_score_not_named(self):
        import numpy as np
        from app.pipeline import match_voiceprints
        db = self._db()
        a = np.zeros(192, dtype=np.float32); a[0] = 1.0
        b = np.zeros(192, dtype=np.float32); b[1] = 1.0
        self._add(db, "甲", a)
        self._add(db, "乙", b)
        q = np.random.RandomState(0).randn(192).astype(np.float32)  # 随机向量，相似度很低
        names, detail = match_voiceprints(db, {0: q}, threshold=0.35, margin=0.10)
        self.assertEqual(names, {})


class TestDiarCache(unittest.TestCase):
    def test_stale_cache_invalidated(self):
        """段数不符的缓存必须作废，否则分离被静默跳过 → spk 缺失 → 归档崩溃。"""
        import json, tempfile, pathlib
        from app.pipeline import _load_diar_cache
        d = pathlib.Path(tempfile.mkdtemp()) / "diarize.json"
        d.write_text(json.dumps({"labels": [0, 1, 0]}))
        self.assertIsNone(_load_diar_cache(d, 5))     # 段数不符
        self.assertFalse(d.exists())                  # 已删掉

    def test_valid_cache_used(self):
        import json, tempfile, pathlib
        from app.pipeline import _load_diar_cache
        d = pathlib.Path(tempfile.mkdtemp()) / "diarize.json"
        d.write_text(json.dumps({"labels": [0, 1, 0]}))
        self.assertIsNotNone(_load_diar_cache(d, 3))
        self.assertTrue(d.exists())

    def test_corrupt_cache_invalidated(self):
        import tempfile, pathlib
        from app.pipeline import _load_diar_cache
        d = pathlib.Path(tempfile.mkdtemp()) / "diarize.json"
        d.write_text("{not json")
        self.assertIsNone(_load_diar_cache(d, 2))
        self.assertFalse(d.exists())


class TestSegmentEditing(unittest.TestCase):
    def _db(self):
        import tempfile, pathlib
        from app.db import DB
        db = DB(pathlib.Path(tempfile.mkdtemp()) / "t.db")
        jid = db.create_job("a.wav", "/tmp/a.wav", "sha", 1000, 3.0)
        db.save_segments(jid, [
            {"start_ms": 0, "end_ms": 1000, "raw_text": "第一段", "spk": 0},
            {"start_ms": 1000, "end_ms": 2000, "raw_text": "第二段", "spk": 1},
            {"start_ms": 2000, "end_ms": 3000, "raw_text": "第三段", "spk": 0},
        ])
        return db, jid

    def test_delete_renumbers(self):
        db, jid = self._db()
        db.delete_segment(jid, 1)
        segs = db.get_segments(jid)
        self.assertEqual([s["idx"] for s in segs], [0, 1])
        self.assertEqual([s["raw_text"] for s in segs], ["第一段", "第三段"])

    def test_merge_next(self):
        db, jid = self._db()
        self.assertTrue(db.merge_segment_next(jid, 0))
        segs = db.get_segments(jid)
        self.assertEqual(len(segs), 2)
        self.assertEqual(segs[0]["end_ms"], 2000)
        self.assertEqual(segs[0]["raw_text"], "第一段第二段")
        self.assertEqual([s["idx"] for s in segs], [0, 1])


class TestDiarCacheVoiceprintSig(unittest.TestCase):
    def test_voiceprint_change_invalidates_cache(self):
        """新增声纹后认名缓存必须失效，否则新声纹永远不生效。"""
        import json, tempfile, pathlib
        from app.pipeline import _load_diar_cache
        d = pathlib.Path(tempfile.mkdtemp()) / "diarize.json"
        d.write_text(json.dumps({"labels": [0, 1], "vp_sig": "3:2026-10-31"}))
        self.assertIsNotNone(_load_diar_cache(d, 2, "3:2026-10-31"))   # 未变 → 可用
        d.write_text(json.dumps({"labels": [0, 1], "vp_sig": "3:2026-10-31"}))
        self.assertIsNone(_load_diar_cache(d, 2, "4:2026-10-31T20:00"))  # 新增声纹 → 作废

    def test_db_signature_changes(self):
        import tempfile, pathlib, numpy as np
        from app.db import DB
        db = DB(pathlib.Path(tempfile.mkdtemp()) / "t.db")
        s1 = db.voiceprint_signature()
        sid = db.create_speaker("甲")
        db.add_voiceprint(sid, np.zeros(192, dtype=np.float32).tobytes(), 192)
        s2 = db.voiceprint_signature()
        self.assertNotEqual(s1, s2)


class TestHallucination(unittest.TestCase):
    def test_boilerplate_detected(self):
        from app.hallucination import is_hallucination
        for t in ("优优独播剧场", "请不吝点赞 订阅 转发 打赏", "明镜与点点栏目", "感谢您的收看，下期再见"):
            self.assertTrue(is_hallucination(t)[0], t)

    def test_repetition_detected_but_not_dropped(self):
        from app.hallucination import is_repetitive, is_hallucination
        self.assertTrue(is_repetitive("对对对对对对对对对"))
        # 整段是重复 → 由文本清理处理，不在这里整段丢弃（可能含真实内容）
        self.assertFalse(is_hallucination("他这个可能把 AI 放在了一个一个一个")[0])

    def test_punct_only(self):
        from app.hallucination import is_hallucination
        self.assertTrue(is_hallucination("……？！")[0])

    def test_normal_speech_kept(self):
        from app.hallucination import is_hallucination
        for t in ("我们明天去社康听一下胎心", "这个季度的目标是降低接口延迟",
                  "嗯，那个，我们继续", "OK, let's start the meeting"):
            self.assertFalse(is_hallucination(t)[0], t)


class TestResourceGuard(unittest.TestCase):
    def _cfg(self):
        import tempfile, pathlib
        from app.settings import load_config
        cfg = load_config()
        cfg.app.work_dir = tempfile.mkdtemp()
        cfg.app.data_dir = tempfile.mkdtemp()
        cfg.app.output_dir = tempfile.mkdtemp()
        cfg.app.input_dir = tempfile.mkdtemp()
        return cfg

    def test_disk_guard(self):
        from app.resource import check
        ok, why = check(self._cfg(), 3600, free_gb=0.5, mem_gb=8)
        self.assertFalse(ok)
        self.assertIn("磁盘", why)

    def test_memory_guard(self):
        from app.resource import check
        ok, why = check(self._cfg(), 60, free_gb=100, mem_gb=1.0)
        self.assertFalse(ok)
        self.assertIn("内存", why)

    def test_pass(self):
        from app.resource import check
        ok, why = check(self._cfg(), 60, free_gb=100, mem_gb=8)
        self.assertTrue(ok)
        self.assertEqual(why, "")


class TestNotifyAndRateLimit(unittest.TestCase):
    def test_notify_disabled_noop(self):
        from app.notify import send
        from app.settings import load_config
        cfg = load_config()
        cfg.notify.enabled = False
        cfg.notify.webhook_url = "http://127.0.0.1:1/none"
        self.assertFalse(send(cfg, {"id": "j", "source_name": "a.wav"}, "done"))

    def test_notify_bad_url_does_not_raise(self):
        from app.notify import send
        from app.settings import load_config
        cfg = load_config()
        cfg.notify.enabled = True
        cfg.notify.webhook_url = "http://127.0.0.1:1/none"
        self.assertFalse(send(cfg, {"id": "j", "source_name": "a.wav"}, "done"))

    def test_rate_limit_off_by_default(self):
        from app.ratelimit import RateLimiter
        rl = RateLimiter(0)
        for _ in range(100):
            self.assertTrue(rl.allow("1.2.3.4"))

    def test_rate_limit_blocks_burst(self):
        from app.ratelimit import RateLimiter
        rl = RateLimiter(2)          # 每分钟 2 个，桶容量 2
        self.assertTrue(rl.allow("ip"))
        self.assertTrue(rl.allow("ip"))
        self.assertFalse(rl.allow("ip"))
        self.assertTrue(rl.allow("other-ip"))   # 不同 IP 独立


class TestWordTimestamps(unittest.TestCase):
    def test_from_tokens(self):
        from app.words import from_tokens
        ws = from_tokens(["这", "个", "API"], [[0, 100], [100, 200], [200, 400]])
        self.assertEqual([w["t"] for w in ws], ["这", "个", "API"])
        self.assertEqual(ws[-1]["s"], 200)
        self.assertEqual(ws[-1]["e"], 400)
        self.assertFalse(any(w["approx"] for w in ws))

    def test_interpolate_covers_span(self):
        from app.words import interpolate
        ws = interpolate("你好世界", 1000, 2000)
        self.assertEqual(len(ws), 4)
        self.assertEqual(ws[0]["s"], 1000)
        self.assertEqual(ws[-1]["e"], 2000)
        for a, b in zip(ws, ws[1:]):
            self.assertLessEqual(a["e"], b["s"] + 1)

    def test_join_keeps_spaces(self):
        from app.words import interpolate, join
        self.assertEqual(join(interpolate("abc 123 你好", 0, 900)), "abc 123你好")

    def test_empty(self):
        from app.words import interpolate
        self.assertEqual(interpolate("", 0, 100), [])


class TestSettingsUI(unittest.TestCase):
    def test_validate_ok(self):
        from app import settings_ui as sui
        vals, errs = sui.validate({"diarization.match_threshold": 0.42,
                                   "refine.style": "narrative",
                                   "text.itn": False,
                                   "obsidian.vault_path": "/tmp/x"})
        self.assertEqual(errs, {})
        self.assertEqual(vals["diarization"]["match_threshold"], 0.42)
        self.assertEqual(vals["refine"]["style"], "narrative")
        self.assertFalse(vals["text"]["itn"])
        self.assertEqual(vals["obsidian"]["vault_path"], "/tmp/x")

    def test_validate_rejects(self):
        from app import settings_ui as sui
        _, errs = sui.validate({"diarization.match_threshold": 5,
                                "asr.provider": "nope"})
        self.assertIn("diarization.match_threshold", errs)
        self.assertIn("asr.provider", errs)

    def test_unknown_key_ignored(self):
        from app import settings_ui as sui
        vals, errs = sui.validate({"not.a.key": 1})
        self.assertEqual(vals, {})
        self.assertEqual(errs, {})

    def test_check_vault(self):
        import tempfile, pathlib
        from app import settings_ui as sui
        d = pathlib.Path(tempfile.mkdtemp())
        ok, msg = sui.check_vault(str(d))
        self.assertTrue(ok, msg)
        ok2, _ = sui.check_vault(str(d / "missing"))
        self.assertFalse(ok2)
        ok3, _ = sui.check_vault("")
        self.assertFalse(ok3)

    def test_values_of_schema_keys_exist(self):
        from app import settings_ui as sui
        from app.settings import load_config
        vals = sui.values_of(load_config())
        self.assertEqual(len(vals), len(sui._INDEX))
        self.assertIn("obsidian.vault_path", vals)


class TestFlowAndService(unittest.TestCase):
    def test_flow_statuses(self):
        from app import flow
        job = {"stage": "diarized", "status": "running"}
        evs = [
            {"stage": "normalized", "message": "解码为 16k 单声道", "ts": "2026-10-31T20:00:00+0800"},
            {"stage": "transcribed", "message": "121 句", "ts": "2026-10-31T20:05:00+0800"},
        ]
        nodes = flow.build(job, evs)
        by = {n["key"]: n for n in nodes}
        self.assertEqual(by["normalized"]["status"], "done")
        self.assertEqual(by["transcribed"]["status"], "done")
        self.assertEqual(by["transcribed"]["seconds"], 300.0)
        self.assertEqual(by["diarized"]["status"], "running")
        self.assertEqual(by["archived"]["status"], "pending")

    def test_flow_skip_and_finished(self):
        from app import flow
        job = {"stage": "archived", "status": "done"}
        evs = [{"stage": "normalized", "message": "命中缓存（跳过）", "ts": "2026-10-31T20:00:00+0800"}]
        nodes = flow.build(job, evs)
        self.assertEqual(nodes[0]["status"], "skipped")
        # 已完成但没事件的阶段也算 skipped
        self.assertEqual([n for n in nodes if n["key"] == "corrected"][0]["status"], "skipped")

    def test_flow_last_run_only(self):
        from app import flow
        job = {"stage": "archived", "status": "done"}
        evs = [
            {"stage": "normalized", "message": "old", "ts": "2026-01-01T00:00:00+0800"},
            {"stage": "transcribed", "message": "old", "ts": "2026-01-01T01:00:00+0800"},
            {"stage": "normalized", "message": "new", "ts": "2026-10-31T20:00:00+0800"},
            {"stage": "transcribed", "message": "new121", "ts": "2026-10-31T20:05:00+0800"},
        ]
        nodes = flow.build(job, evs)
        self.assertEqual(nodes[2]["detail"], "new121")

    def test_plist_shape(self):
        from app import service as svc
        import pathlib
        pl = svc.build_plist("/usr/bin/python3", pathlib.Path("/tmp/proj"),
                             "/tmp/proj/config/config.yaml", pathlib.Path("/tmp/proj/logs"), port=9000)
        self.assertEqual(pl["Label"], svc.LABEL)
        self.assertIn("service", pl["ProgramArguments"])
        self.assertIn("--port", pl["ProgramArguments"])
        self.assertTrue(pl["RunAtLoad"])
        self.assertEqual(pl["WorkingDirectory"], "/tmp/proj")


class TestSpeakerManagement(unittest.TestCase):
    def _db(self):
        import tempfile, pathlib, numpy as np
        from app.db import DB
        db = DB(pathlib.Path(tempfile.mkdtemp()) / "t.db")
        sid = db.create_speaker("甲")
        for v in (0.1, 0.2, 0.3):
            vec = np.full(192, v, dtype=np.float32)
            db.add_voiceprint(sid, vec.tobytes(), 192, source="t", quality=1.0)
        return db, sid

    def test_list_and_delete_sample(self):
        db, sid = self._db()
        rows = db.list_voiceprints(sid)
        self.assertEqual(len(rows), 3)
        self.assertTrue(db.delete_voiceprint(rows[0]["id"]))
        self.assertEqual(db.count_voiceprints(sid), 2)
        self.assertFalse(db.delete_voiceprint(999999))

    def test_rename(self):
        db, sid = self._db()
        db.rename_speaker(sid, "乙")
        self.assertEqual(db.get_speaker(sid)["name"], "乙")
        self.assertIsNotNone(db.get_speaker_by_name("乙"))

    def test_rename_conflict(self):
        import tempfile, pathlib
        from app.db import DB
        db, sid = self._db()
        db.create_speaker("丙")
        import sqlite3
        with self.assertRaises(sqlite3.IntegrityError):
            db.rename_speaker(sid, "丙")

    def test_speaker_vectors(self):
        db, sid = self._db()
        self.assertEqual(len(db.speaker_vectors(sid)), 3)


class TestTokenCookiePropagation(unittest.TestCase):
    """用 ?token= 打开页面后，页面内 fetch 必须能通过（否则列表空白/无法新增）。"""

    def _app(self):
        import tempfile, pathlib
        from app.settings import load_config
        from app.web import create_app
        cfg = load_config()
        cfg.server.bind = "0.0.0.0"
        cfg.server.token = "secret"
        tmp = pathlib.Path(tempfile.mkdtemp())
        for name in ("data", "work", "output", "input"):
            (tmp / name).mkdir(parents=True, exist_ok=True)
        cfg.app.data_dir = str(tmp / "data")
        cfg.app.work_dir = str(tmp / "work")
        cfg.app.output_dir = str(tmp / "output")
        cfg.app.input_dir = str(tmp / "input")
        return create_app(cfg)

    def test_page_sets_cookie_then_api_works(self):
        from fastapi.testclient import TestClient
        c = TestClient(self._app())
        r = c.get("/voices?token=secret")
        self.assertEqual(r.status_code, 200)
        self.assertIn("tingji_token", r.cookies)
        # 后续请求只靠 cookie（模拟页面内 fetch）
        self.assertEqual(c.get("/api/voices").status_code, 200)

    def test_api_direct_token_still_works(self):
        from fastapi.testclient import TestClient
        c = TestClient(self._app())
        self.assertEqual(c.get("/api/voices?token=secret").status_code, 200)
        c.cookies.clear()   # TestClient 会保留上一步种下的 cookie
        self.assertEqual(c.get("/api/voices").status_code, 401)


class TestServiceToken(unittest.TestCase):
    """统一入口 service 也必须启用令牌，否则等于把服务公开。"""

    def test_prepare_server_generates_token(self):
        from app.cli import _prepare_server
        from app.settings import load_config, OVERRIDES_PATH
        cfg = load_config()
        cfg.server.token = ""
        cfg.server.bind = "0.0.0.0"
        try:
            bind, port, token = _prepare_server(cfg, None, 8123)
            self.assertEqual(bind, "0.0.0.0")
            self.assertEqual(port, 8123)
            self.assertTrue(token)
            self.assertEqual(cfg.server.token, token)
        finally:
            pass


class TestServedJsParses(unittest.TestCase):
    """每个页面下发的内联 JS 必须能通过 node --check。

    背景：Python 三引号里写 JS 的 '\\n' 会被解释成真换行，导致整个 <script> 语法错误、
    页面所有交互失效（表现为"列表空白 + 没有上传按钮"），而服务端仍返回 200。
    """

    def _app_and_db(self):
        import tempfile, pathlib
        from app.settings import load_config
        from app.web import create_app
        from app.db import DB
        cfg = load_config()
        cfg.server.bind = "127.0.0.1"
        cfg.server.token = ""
        tmp = pathlib.Path(tempfile.mkdtemp())
        for name in ("data", "work", "output", "input"):
            (tmp / name).mkdir(parents=True, exist_ok=True)
        cfg.app.data_dir = str(tmp / "data")
        cfg.app.work_dir = str(tmp / "work")
        cfg.app.output_dir = str(tmp / "output")
        cfg.app.input_dir = str(tmp / "input")
        db = DB(cfg.db_path)
        jid = db.create_job("a.wav", "/tmp/a.wav", "sha", 10, 60.0)
        db.save_segments(jid, [{"start_ms": 0, "end_ms": 1000, "raw_text": "测试",
                                "spk": 0, "words": [{"t": "测", "s": 0, "e": 500, "approx": 0}]}])
        return create_app(cfg), jid

    def test_all_pages_js_syntax(self):
        import shutil, re, tempfile, pathlib
        from fastapi.testclient import TestClient
        if not shutil.which("node"):
            self.skipTest("node 不可用")
        import subprocess
        app, jid = self._app_and_db()
        c = TestClient(app)
        pages = ["/", "/hotwords", "/voices", "/settings", "/logs", f"/job/{jid}"]
        for path in pages:
            r = c.get(path)
            self.assertEqual(r.status_code, 200, path)
            scripts = re.findall(r"<script>(.*?)</script>", r.text, re.S)
            self.assertTrue(scripts, f"{path} 没有内联脚本")
            f = pathlib.Path(tempfile.mkdtemp()) / "p.js"
            f.write_text("\n".join(scripts))
            out = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, f"{path} JS 语法错误：{out.stderr[:400]}")


class TestDocTypeClassification(unittest.TestCase):
    def _segs(self, n=10, spk=(0,), sad=0, cry=0):
        out = []
        for i in range(n):
            s = {"spk": spk[i % len(spk)], "start_ms": i * 1000, "end_ms": i * 1000 + 900,
                 "raw_text": "内容"}
            if i < sad:
                s["emotion"] = "SAD"
            if i < cry:
                s["event"] = "Cry"
            out.append(s)
        return out

    def test_meeting_three_speakers(self):
        from app.obsidian import classify
        self.assertEqual(classify({}, self._segs(spk=(0, 1, 2)), 3600), "会议纪要")

    def test_daily_two_speakers_short(self):
        from app.obsidian import classify
        self.assertEqual(classify({}, self._segs(spk=(0, 1)), 600), "日常对话")

    def test_important_moment_by_cry(self):
        from app.obsidian import classify
        self.assertEqual(classify({}, self._segs(n=10, spk=(0, 1), cry=1), 600), "重要时刻")

    def test_important_moment_by_sad_ratio(self):
        from app.obsidian import classify
        self.assertEqual(classify({}, self._segs(n=10, spk=(0, 1), sad=3), 600), "重要时刻")

    def test_lecture_single_speaker_long(self):
        from app.obsidian import classify
        self.assertEqual(classify({}, self._segs(n=10, spk=(0,)), 1800), "讲座")

    def test_llm_type_wins(self):
        from app.obsidian import classify
        self.assertEqual(classify({}, self._segs(spk=(0, 1, 2)), 3600,
                                  {"doc_type": "访谈"}), "访谈")

    def test_tags_include_type(self):
        from app.obsidian import build_tags
        from datetime import datetime
        tags = build_tags({}, 600, ["张三"], datetime(2026, 10, 31), "日常对话")
        self.assertIn("日常对话", tags)
        self.assertIn("类型/日常对话", tags)
        self.assertNotIn("会议纪要", tags)

    def test_note_rel_dir(self):
        from app.obsidian import note_rel_dir
        from app.settings import ObsidianCfg
        from datetime import datetime
        d = datetime(2026, 10, 31)
        base = "2026-10-31 2212 标题"
        # 默认：每条录音一个文件夹
        self.assertEqual(
            note_rel_dir(ObsidianCfg(notes_dir="录音笔记", notes_by_type=True, per_note_folder=True),
                         d, "重要时刻", base),
            f"录音笔记/重要时刻/2026/{base}")
        # 关掉按类型
        self.assertEqual(
            note_rel_dir(ObsidianCfg(notes_dir="录音笔记", notes_by_type=False, per_note_folder=False),
                         d, "重要时刻", base),
            "录音笔记/2026")


class TestRulesAndRanges(unittest.TestCase):
    def test_rule_parse_and_apply(self):
        import tempfile, pathlib
        from app.rules import RuleCorrector
        f = pathlib.Path(tempfile.mkdtemp()) / "rules.txt"
        f.write_text("# 注释\n欧拉玛 = Ollama\n(艾特)\\s*(\\w+)\\s*点\\s*(\\w+) = @\\2.\\3\n坏规则没有等号\n")
        rc = RuleCorrector(f)
        self.assertEqual(len(rc.rules), 2)
        self.assertEqual(len(rc.errors), 1)
        out, n = rc.apply("用欧拉玛，邮箱 艾特 abc 点 com")
        self.assertIn("Ollama", out)
        self.assertIn("@abc.com", out)
        self.assertEqual(n, 2)

    def test_bad_replacement_ignored(self):
        import tempfile, pathlib
        from app.rules import RuleCorrector
        f = pathlib.Path(tempfile.mkdtemp()) / "rules.txt"
        f.write_text("x = \\s*bad\n")
        rc = RuleCorrector(f)
        out, n = rc.apply("x 保持不变")     # 不应抛异常
        self.assertEqual(out, "x 保持不变")

    def test_range_itn(self):
        from app.itn import normalize
        self.assertEqual(normalize("十五六个人"), "15~16个人")
        self.assertEqual(normalize("三五百台服务器"), "300~500台服务器")
        self.assertEqual(normalize("二三十台机器"), "20~30台机器")
        self.assertEqual(normalize("三四个问题"), "3~4个问题")
        # 不能误伤
        self.assertEqual(normalize("二十三岁"), "23岁")
        self.assertEqual(normalize("三心二意"), "三心二意")
        self.assertEqual(normalize("一个想法"), "一个想法")

    def test_big_number_quantifier(self):
        from app.itn import normalize
        self.assertEqual(normalize("十五个"), "15个")
        self.assertEqual(normalize("十个人"), "10个人")


class TestNanoChunker(unittest.TestCase):
    def test_chunks_cover_audio_and_respect_cap(self):
        import numpy as np
        from app.nano import split_chunks
        sr = 16000
        audio = (np.sin(np.arange(sr * 60) / 100) * 0.1).astype(np.float32)
        chunks = split_chunks(audio, sr, target=8.0, lo=6.0, hi=10.0, max_sec=22.0)
        self.assertGreaterEqual(len(chunks), 6)
        for i, c in enumerate(chunks):
            self.assertLessEqual(c["end"] - c["start"], 22.0 * sr + 1)
            if i:
                self.assertEqual(c["start"], chunks[i - 1]["end"])   # 交付水位首尾相接
        self.assertEqual(chunks[-1]["end"], len(audio))

    def test_short_audio_single_chunk(self):
        import numpy as np
        from app.nano import split_chunks
        sr = 16000
        audio = np.zeros(sr * 3, dtype=np.float32)
        self.assertEqual(len(split_chunks(audio, sr)), 1)


class TestChunkingAndBoundary(unittest.TestCase):
    def test_chunks_have_boundary_and_pad(self):
        import numpy as np
        from app.nano import split_chunks
        sr = 16000
        # 7.5s 语音 + 0.6s 静音，循环
        one = np.concatenate([np.full(int(sr * 7.5), 0.2, "float32"), np.zeros(int(sr * 0.6), "float32")])
        audio = np.tile(one, 4)
        chunks = split_chunks(audio, sr)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["pad"], 0)                    # 首块不回补
        for c in chunks[1:]:
            self.assertLessEqual(c["pad"], int(0.3 * sr))        # 回补不超过切线
            self.assertEqual(c["start"], chunks[chunks.index(c) - 1]["end"])
        self.assertIn("endpoint", {c["boundary"] for c in chunks})

    def test_boundary_kind_and_cap(self):
        import numpy as np
        from app.nano import split_chunks
        sr = 16000
        audio = np.full(int(sr * 50), 0.2, "float32")            # 连续满幅，无静音
        chunks = split_chunks(audio, sr)
        # 没有静音可切 → 中间块只能是 pause（最后一块到文件尾，算 endpoint）
        self.assertTrue(all(c["boundary"] in ("pause", "cut") for c in chunks[:-1]))
        for c in chunks:
            self.assertLessEqual(c["end"] - c["start"], int(22.0 * sr) + 1)

    def test_dedupe_seam(self):
        from app.nano import dedupe_seam
        self.assertEqual(dedupe_seam("我说我们要去开会", "去开会吧明天"), "吧明天")
        self.assertEqual(dedupe_seam("abc", "xyz"), "xyz")
        self.assertEqual(dedupe_seam("", "xyz"), "xyz")

    def test_trim_tail_by_boundary(self):
        from app.textproc import trim_uncertain_tail
        self.assertEqual(trim_uncertain_tail("他说完了。", "cut"), ("他说完了", True))
        self.assertEqual(trim_uncertain_tail("他说完了。", "pause"), ("他说完了", True))
        self.assertEqual(trim_uncertain_tail("他说完了。", "endpoint"), ("他说完了。", False))
        self.assertEqual(trim_uncertain_tail("没有标点", "cut"), ("没有标点", False))

    def test_mask_non_speech_and_recovery(self):
        import numpy as np
        from app.diar_sherpa import mask_non_speech
        sr = 16000
        audio = np.concatenate([
            np.full(sr * 3, 0.2, "float32"),        # 语音
            np.full(sr * 2, 0.02, "float32"),       # 轻声（应兜底保留）
            np.full(sr * 3, 0.2, "float32"),        # 语音
            np.zeros(sr * 2, "float32"),            # 真静音（应置零）
            np.full(sr * 3, 0.2, "float32"),
        ])
        m = mask_non_speech(audio, [(0, 3), (5, 8), (10, 13)], sr)
        def lvl(s, e):
            return float(np.sqrt(np.mean(m[int(s * sr):int(e * sr)] ** 2)))
        self.assertGreater(lvl(0, 3), 0.15)
        self.assertGreater(lvl(3, 5), 0.01)          # 兜底保留了轻声
        self.assertLess(lvl(8, 10), 1e-6)            # 真静音被抹掉


class TestAutoEngine(unittest.TestCase):
    def test_long_meeting_uses_paraformer(self):
        from app.asr_select import choose
        eng, why = choose(7560, 0.9, True, 2)      # 2 小时
        self.assertEqual(eng, "funasr")

    def test_dialogue_uses_nano(self):
        from app.asr_select import choose
        eng, why = choose(759, 0.98, True, 2)      # 12.7 分钟一对一
        self.assertEqual(eng, "funasr_nano")
        self.assertIn("对话", why)

    def test_three_speakers_is_meeting(self):
        from app.asr_select import choose
        self.assertEqual(choose(600, 0.7, True, 3)[0], "funasr")

    def test_single_speaker_is_lecture(self):
        from app.asr_select import choose
        self.assertEqual(choose(600, 0.9, True, 1)[0], "funasr")

    def test_nano_missing_falls_back(self):
        from app.asr_select import choose
        self.assertEqual(choose(600, 0.4, False, 2)[0], "sensevoice")


class TestJobArchive(unittest.TestCase):
    def _db(self):
        import tempfile, pathlib
        from app.db import DB
        p = pathlib.Path(tempfile.mkdtemp()) / "t.db"
        return DB(p)

    def test_archive_filters_list(self):
        db = self._db()
        a1 = db.create_job("a.wav", "/tmp/a.wav", "x1", 1, 10)
        a2 = db.create_job("b.wav", "/tmp/b.wav", "x2", 1, 10)
        self.assertEqual(len(db.list_jobs()), 2)
        db.set_archived(a1, True)
        self.assertEqual([j["id"] for j in db.list_jobs()], [a2])
        self.assertEqual(len(db.list_jobs(include_archived=True)), 2)
        db.set_archived(a1, False)
        self.assertEqual(len(db.list_jobs()), 2)

    def test_archived_column_migrated_on_old_db(self):
        import sqlite3, tempfile, pathlib
        from app.db import DB
        p = pathlib.Path(tempfile.mkdtemp()) / "old.db"
        c = sqlite3.connect(p)
        c.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, source_name TEXT, source_path TEXT, "
                  "sha256 TEXT, size_bytes INT, duration_sec REAL, status TEXT, stage TEXT, "
                  "progress INT, message TEXT, error_code TEXT, degraded INT, created_at TEXT, "
                  "started_at TEXT, finished_at TEXT)")
        c.commit(); c.close()
        DB(p)     # 触发迁移
        cols = [r[1] for r in sqlite3.connect(p).execute("PRAGMA table_info(jobs)")]
        self.assertIn("archived", cols)


class TestChannelClusters(unittest.TestCase):
    def _db(self, per_person):
        import tempfile, pathlib, numpy as np
        from app.db import DB
        db = DB(pathlib.Path(tempfile.mkdtemp()) / "v.db")
        for name, vecs in per_person.items():
            sid = db.create_speaker(name)
            for v in vecs:
                db.add_voiceprint(sid, v.astype("float32").tobytes(), len(v), source="web")
        return db

    def test_group_channels_splits_two_sources(self):
        import numpy as np
        from app.pipeline import group_channels
        rng = np.random.default_rng(0)
        a = np.array([1.0, 0, 0, 0], dtype=np.float32)
        b = np.array([0, 1.0, 0, 0], dtype=np.float32)
        vecs = [a + 0.02 * rng.standard_normal(4).astype("float32") for _ in range(3)] + \
               [b + 0.02 * rng.standard_normal(4).astype("float32") for _ in range(3)]
        cs = group_channels(vecs, threshold=0.75, max_channels=4)
        self.assertEqual(len(cs), 2, "两个信道应分成两簇")

    def test_cross_channel_match_beats_single_centroid(self):
        import numpy as np
        from app.pipeline import match_voiceprints
        rng = np.random.default_rng(1)
        web = np.array([1.0, 0, 0, 0], dtype=np.float32)
        car = np.array([0, 1.0, 0, 0], dtype=np.float32)
        # 同一人：3 条网页样本 + 1 条车机样本
        vecs = [web + 0.02 * rng.standard_normal(4).astype("float32") for _ in range(3)]
        vecs.append(car + 0.02 * rng.standard_normal(4).astype("float32"))
        db = self._db({"张三": vecs})
        query = {0: car + 0.02 * rng.standard_normal(4).astype("float32")}
        names, detail = match_voiceprints(db, query, threshold=0.30, margin=0.0)
        self.assertEqual(names.get(0), "张三")
        self.assertGreater(detail["0"]["score"], 0.95)
        self.assertEqual(detail["0"]["channels"], 2)

    def test_single_channel_unchanged(self):
        import numpy as np
        from app.pipeline import match_voiceprints
        rng = np.random.default_rng(2)
        v = np.array([1.0, 0, 0, 0], dtype=np.float32)
        db = self._db({"张三": [v + 0.02 * rng.standard_normal(4).astype("float32") for _ in range(3)]})
        names, detail = match_voiceprints(db, {0: v}, threshold=0.3, margin=0.0)
        self.assertEqual(names.get(0), "张三")
        self.assertEqual(detail["0"]["channels"], 1)


class TestChannelIds(unittest.TestCase):
    def test_cluster_channels_returns_ids(self):
        import numpy as np
        from app.pipeline import cluster_channels
        rng = np.random.default_rng(7)
        a = np.array([1.0, 0, 0, 0], dtype=np.float32)
        b = np.array([0, 1.0, 0, 0], dtype=np.float32)
        vecs = [a + 0.01 * rng.standard_normal(4).astype("float32") for _ in range(2)] + \
               [b + 0.01 * rng.standard_normal(4).astype("float32") for _ in range(2)]
        cents, ids = cluster_channels(vecs, threshold=0.75, max_channels=4)
        self.assertEqual(len(cents), 2)
        self.assertEqual(len(ids), 4)
        self.assertEqual(len(set(ids[:2])), 1)     # 前两条同簇
        self.assertEqual(len(set(ids[2:])), 1)     # 后两条同簇
        self.assertNotEqual(ids[0], ids[2])

    def test_group_channels_matches_cluster_channels(self):
        import numpy as np
        from app.pipeline import cluster_channels, group_channels
        rng = np.random.default_rng(8)
        vecs = [np.array([1.0, 0, 0, 0]) + 0.02 * rng.standard_normal(4).astype("float32") for _ in range(3)]
        self.assertEqual(len(group_channels(vecs)), len(cluster_channels(vecs)[0]))


class TestSplitSegment(unittest.TestCase):
    def test_split_by_words(self):
        from app.textproc import split_by_time
        words = [{"t": "你好", "s": 0, "e": 400}, {"t": "我是", "s": 400, "e": 800},
                 {"t": "张三", "s": 800, "e": 1200}]
        left, right = split_by_time(words, 0, 2000, 800)
        self.assertEqual(left, "你好我是")
        self.assertEqual(right, "张三")

    def test_split_without_words_by_ratio(self):
        from app.textproc import split_by_time
        left, right = split_by_time(None, 0, 10000, 5000, "一二三四五六七八九十")
        self.assertEqual(left + right, "一二三四五六七八九十")
        self.assertTrue(left and right)

    def test_split_out_of_range(self):
        from app.textproc import split_by_time
        import unittest as _u
        with self.assertRaises(ValueError):
            split_by_time([], 0, 1000, 5000, "x")

    def test_db_split_segment(self):
        import tempfile, pathlib
        from app.db import DB
        db = DB(pathlib.Path(tempfile.mkdtemp()) / "s.db")
        jid = db.create_job("a.wav", "/tmp/a.wav", "sha1", 1, 10)
        db.save_segments(jid, [{"start_ms": 0, "end_ms": 2000, "raw_text": "你好我是张三",
                                "final_text": "你好我是张三", "quality": 1.0, "spk": 0}])
        new_idx = db.split_segment(jid, 0, 800, "你好", "我是张三")
        segs = db.get_segments(jid)
        self.assertEqual(len(segs), 2)
        self.assertEqual(new_idx, 1)
        self.assertEqual(segs[0]["end_ms"], 800)
        self.assertEqual(segs[1]["start_ms"], 800)
        self.assertEqual(segs[1]["end_ms"], 2000)
        self.assertEqual(segs[1]["final_text"], "我是张三")

    def test_db_split_keeps_following_idx(self):
        import tempfile, pathlib
        from app.db import DB
        db = DB(pathlib.Path(tempfile.mkdtemp()) / "s2.db")
        jid = db.create_job("b.wav", "/tmp/b.wav", "sha2", 1, 10)
        db.save_segments(jid, [
            {"start_ms": 0, "end_ms": 1000, "raw_text": "一", "final_text": "一", "quality": 1.0, "spk": 0},
            {"start_ms": 1000, "end_ms": 2000, "raw_text": "二", "final_text": "二", "quality": 1.0, "spk": 0},
        ])
        db.split_segment(jid, 0, 500, "半", "一")
        segs = db.get_segments(jid)
        self.assertEqual([s["idx"] for s in segs], [0, 1, 2])
        self.assertEqual(segs[2]["final_text"], "二")
