"""评估声纹登记文本的音素覆盖率（声母/韵母/声调）。

用法: python coverage.py <文本文件，每行一句>
"""
import re, sys
from pypinyin import pinyin, Style

INITIALS = set("b p m f d t n l g k h j q x zh ch sh r z c s".split())
FINALS = set("""a o e i u ü ai ei ao ou an en ang eng ong er
ia ie iao iou ian in iang ing iong
ua uo uai uei uan uen uang ueng üe üan ün""".split())
# pypinyin 会把 iou/uei/uen 缩写成 iu/ui/un，这里还原
EXPAND = {"iu": "iou", "ui": "uei", "un": "uen"}

def analyze(texts):
    ini, fin, tone, detail = set(), set(), set(), []
    for t in texts:
        for ch in t:
            if not re.match(r"[\u4e00-\u9fff]", ch):
                continue
            i = pinyin(ch, style=Style.INITIALS, strict=True)[0][0]
            f = pinyin(ch, style=Style.FINALS, strict=True)[0][0].replace("v", "ü")
            f = EXPAND.get(f, f)
            tn = pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)[0][0]
            m = re.search(r"\d", tn)
            if i in INITIALS: ini.add(i)
            if f in FINALS: fin.add(f)
            elif f: detail.append((ch, f))
            if m: tone.add(m.group())
    return ini, fin, tone, detail

if __name__ == "__main__":
    texts = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    ini, fin, tone, detail = analyze(texts)
    chars = sum(len(re.findall(r"[\u4e00-\u9fff]", t)) for t in texts)
    print(f"句数={len(texts)}  汉字数={chars}  预估朗读时长≈{chars/4.2:.1f}s")
    print(f"声母 {len(ini)}/21   缺: {' '.join(sorted(INITIALS-ini)) or '无'}")
    print(f"韵母 {len(fin)}/{len(FINALS)}  缺: {' '.join(sorted(FINALS-fin)) or '无'}")
    print(f"声调 {len(tone)}/5    缺: {' '.join(sorted(set('12345')-tone)) or '无'}")
    if detail: print("未匹配韵母明细:", detail[:12])
    tot = (len(ini)/21 + len(fin)/len(FINALS) + len(tone)/5) / 3
    print(f"综合覆盖率 = {tot*100:.1f}%")
