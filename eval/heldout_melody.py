"""留出集上的旋律提取测试。

allocator 的权重是在三首曲子上调的, 这里拿没见过的曲子测。被测的只读四声部 MIDI,
标准答案从乐谱 PDF 上识别出来, 两条路互不相干。
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from cantabile.ingest.midi import load
from cantabile.ingest.sheet import read_melody
from cantabile.arrange.allocate import allocate
from cantabile.analyze.melody import melody_line, evaluate_against
from cantabile.analyze.streams import skyline
from cantabile.model import Note, Voice

HERE = Path(__file__).parent / "heldout"


def pairs():
    for mid in sorted(HERE.glob("*.mid")):
        pdf = mid.with_suffix(".pdf")
        if pdf.exists() and pdf.stat().st_size > 10000:
            yield mid.stem, mid, pdf


def truth_from_score(pdf):
    """从乐谱 PDF 读旋律, 返回 (序号, 音高, 时长)。"""
    try:
        m = read_melody(str(pdf))
    except Exception as e:
        return None, f"sheet reader failed: {e}"
    if not m:
        return None, "no noteheads found"
    xuanlv = []
    for i, e in enumerate(m):
        p = e[1] if not hasattr(e, "pitch") else e.pitch
        xuanlv.append((float(i), int(p), 1.0))
    return xuanlv, None


def pitch_sequence_f1(truth, got):
    """两条旋律当音高序列对齐 (Needleman-Wunsch) 打分。识谱只拿得到音高顺序, 不比起始时间。"""
    a = [p for _, p, _ in truth]
    b = [p for _, p, _ in got]
    n, m = len(a), len(b)
    if not n or not m:
        return dict(f1=0.0, n_truth=n, n_pred=m, matched=0)
    prev = list(range(0, -(m + 1), -1))
    for i in range(1, n + 1):
        cur = [-i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = max(prev[j - 1] + (1 if a[i - 1] == b[j - 1] else -1),
                         prev[j] - 1, cur[j - 1] - 1)
        prev = cur
    matched = max(0, (prev[m] + n + m) // 2) if prev[m] is not None else 0
    # 准确的匹配数用 LCS 算
    matched = _traceback_matches(a, b)
    p = matched / m
    r = matched / n
    return dict(f1=(2 * p * r / (p + r) if p + r else 0.0), precision=p,
                recall=r, n_truth=n, n_pred=m, matched=matched)


def _traceback_matches(a, b):
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = max(D[i - 1][j - 1] + (1 if a[i - 1] == b[j - 1] else 0),
                          D[i - 1][j], D[i][j - 1])
    return D[n][m]


def main():
    jieguo = []
    for ming, mid, pdf in pairs():
        daan, err = truth_from_score(pdf)
        if daan is None:
            print(f"  {ming}: skipped ({err})")
            continue
        try:
            ev, tpb, tm = load(str(mid))
        except Exception as e:
            print(f"  {ming}: skipped (midi: {e})")
            continue
        yinfu = [Note(o, d, p, v, Voice.INNER, "src") for o, d, p, v in ev]
        got_alloc = [(x.onset, x.pitch, x.dur) for x in allocate(ev).melody]
        got_sky = [(x.onset, x.pitch, x.dur) for x in skyline(yinfu).notes]
        try:
            got_contig = melody_line(yinfu, parts=4)
        except Exception:
            got_contig = []
        a = pitch_sequence_f1(daan, got_alloc)
        s = pitch_sequence_f1(daan, got_sky)
        c = pitch_sequence_f1(daan, got_contig) if got_contig else dict(f1=0.0)
        jieguo.append((ming, len(daan), s["f1"], c["f1"], a["f1"]))
        print(f"  {ming:8s} score {len(daan):3d} notes | "
              f"skyline {s['f1']:.3f}  contig {c['f1']:.3f}  allocate {a['f1']:.3f}")
    if jieguo:
        n = len(jieguo)
        print(f"\n  {'MEAN':8s} {'':10s}| "
              f"skyline {sum(r[2] for r in jieguo)/n:.3f}  "
              f"contig {sum(r[3] for r in jieguo)/n:.3f}  "
              f"allocate {sum(r[4] for r in jieguo)/n:.3f}   (n={n} held-out pieces)")
    return jieguo


if __name__ == "__main__":
    main()
