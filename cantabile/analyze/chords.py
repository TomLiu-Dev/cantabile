"""找调和推和弦。

调: Krumhansl-Schmuckler (KK 或 Temperley CBMS 音级分布)。
和弦: Temperley 的和声偏好规则, 简化成一遍 Viterbi; 平滑比每格都准更要紧。
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..theory import QUALITIES, Chord, Harmony
from .structure import as_events, metric_weight

__all__ = [
    "key_of", "infer", "chord_scores", "evaluate", "pc_profile",
    "scale_of", "fifths_distance", "KEY_PROFILES", "DEFAULT_QUALITIES",
]


KK_MAJOR = (6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88)
KK_MINOR = (6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17)

CBMS_MAJOR = (5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0)
CBMS_MINOR = (5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0)

KEY_PROFILES = {
    "KS": (KK_MAJOR, KK_MINOR),
    "KK": (KK_MAJOR, KK_MINOR),
    "CBMS": (CBMS_MAJOR, CBMS_MINOR),
    "temperley": (CBMS_MAJOR, CBMS_MINOR),
}

MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10, 11)     # 自然小调加升七


# 根音以上各音程的分量。五音排在三音前面, {G B D E} 低音 G 才读成 G6 不是 Em7
_ROLE = {
    0: 1.00,
    7: 0.90,
    4: 0.86,
    3: 0.80,
    10: 0.62, 11: 0.62,
    6: 0.55, 8: 0.55,
    1: 0.40,
    5: 0.90,                      # sus4 顶替三音
    9: 0.50,
    2: 0.15,
}

# 在曲库和弦上坐标搜索调出来的
_OUTSIDE = 0.40
_MISSING = 0.055
_SIZE = 0.00
_BASS_ROOT = 0.085
_BASS_TONE = 0.030
_BASS_ALIEN = -0.120
_KEY_TONES = 0.100
_KEY_ROOT = 0.030

DEFAULT_QUALITIES = tuple(QUALITIES)


def scale_of(key) -> frozenset:
    """key 是 (tonic_pc, mode)。"""
    if key is None:
        return frozenset(range(12))
    tonic, mode = int(key[0]) % 12, str(key[1]).lower()
    jieji = MINOR_SCALE if mode.startswith("min") else MAJOR_SCALE
    return frozenset((tonic + d) % 12 for d in jieji)


def fifths_distance(a: int, b: int) -> int:
    """五度线上的距离 0..6。"""
    d = ((int(a) - int(b)) * 7) % 12
    return int(min(d, 12 - d))


def pc_profile(notes, *, t0: float = None, t1: float = None,
               beats_per_bar: int = 4, metric: bool = True) -> np.ndarray:
    """[t0, t1) 里按时值加权的音级分布; metric 时再乘节拍权重。"""
    jieguo = np.zeros(12, dtype=float)
    for onset, dur, pitch, _vel in as_events(notes):
        lo = onset if t0 is None else max(onset, t0)
        hi = (onset + dur) if t1 is None else min(onset + dur, t1)
        span = hi - lo
        if span <= 1e-9:
            continue
        w = span
        if metric:
            w *= 0.25 + 0.75 * metric_weight(max(onset, lo), beats_per_bar)
        jieguo[pitch % 12] += w
    return jieguo


def key_of(notes, *, profile: str = "KS") -> tuple:
    """K-S 找调, 返回 (tonic_pc, mode, 相关系数)。"""
    try:
        maj, minr = KEY_PROFILES[profile]
    except KeyError:
        raise ValueError(f"unknown key profile {profile!r}; "
                         f"have {sorted(KEY_PROFILES)}")

    x = pc_profile(notes, metric=False)
    if x.sum() <= 0:
        raise ValueError("no sounding notes: cannot find a key")

    best = None
    for tonic in range(12):
        xuanzhuan = np.roll(x, -tonic)
        for mode, prof in (("major", np.array(maj)), ("minor", np.array(minr))):
            r = _pearson(xuanzhuan, prof)
            if best is None or r > best[2]:
                best = (tonic, mode, r)
    return (best[0], best[1], float(best[2]))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    da, db = a - a.mean(), b - b.mean()
    den = float(np.sqrt((da * da).sum() * (db * db).sum()))
    return float((da * db).sum() / den) if den > 0 else 0.0


def _as_vector(pcs_weighted) -> np.ndarray:
    if isinstance(pcs_weighted, Mapping):
        v = np.zeros(12, dtype=float)
        for pc, w in pcs_weighted.items():
            v[int(pc) % 12] += float(w)
        return v
    v = np.asarray(list(pcs_weighted), dtype=float)
    if v.shape != (12,):
        raise ValueError(f"expected 12 pitch-class weights, got {v.shape}")
    return v


def chord_scores(pcs_weighted, key=None, *, bass_pc: int = None,
                 qualities: Sequence = None) -> list:
    """给所有候选和弦打分, 返回 [(Chord, score)], 好的在前。"""
    w = _as_vector(pcs_weighted)
    total = float(w.sum())
    if total <= 0:
        return []
    w = w / total
    in_key = scale_of(key) if key is not None else None
    quals = tuple(qualities) if qualities else DEFAULT_QUALITIES

    jieguo = []
    for root in range(12):
        for q in quals:
            ivs = QUALITIES[q]
            hexianyin = [(root + i) % 12 for i in ivs]
            fugai = 0.0
            queshao = 0
            for i, pc in zip(ivs, hexianyin):
                quanzhong = w[pc]
                fugai += quanzhong * _ROLE.get(i % 12, 0.4)
                if quanzhong <= 0.0:
                    queshao += 1
            waimian = float(w.sum() - sum(w[pc] for pc in set(hexianyin)))
            defen = fugai - _OUTSIDE * waimian
            defen -= _MISSING * queshao
            defen -= _SIZE * len(set(hexianyin))
            if bass_pc is not None:
                bpc = int(bass_pc) % 12
                if bpc == root:
                    defen += _BASS_ROOT
                elif bpc in hexianyin:
                    defen += _BASS_TONE
                else:
                    defen += _BASS_ALIEN
            if in_key is not None:
                if all(pc in in_key for pc in hexianyin):
                    defen += _KEY_TONES
                if root in in_key:
                    defen += _KEY_ROOT
            bass = bass_pc % 12 if (bass_pc is not None and bass_pc % 12 in hexianyin
                                    and bass_pc % 12 != root) else None
            jieguo.append((Chord(root, q, bass if bass is not None else root), defen))
    jieguo.sort(key=lambda cs: -cs[1])
    return jieguo


# Viterbi 转移代价
_SHARPNESS = 7.0
_CHANGE = 0.80
_VARIANCE = 0.11       # 根音在五度线上每走一步
_QUALITY_ONLY = 0.30   # 同根换性质
_WEAK_BEAT = 0.45      # 弱拍换和弦另加


def _slot_evidence(events, t0, t1, beats_per_bar):
    w = np.zeros(12, dtype=float)
    bass_pitch, bass_pc, bass_w = 128, None, 0.0
    for onset, dur, pitch, vel in events:
        lo, hi = max(onset, t0), min(onset + dur, t1)
        span = hi - lo
        if span <= 1e-9:
            continue
        m = 0.25 + 0.75 * metric_weight(max(onset, t0), beats_per_bar)
        w[pitch % 12] += span * m
        if pitch < bass_pitch or (pitch == bass_pitch and span > bass_w):
            bass_pitch, bass_pc, bass_w = pitch, pitch % 12, span
    return w, bass_pc


def infer(notes, *, resolution: float = 2.0, key=None, smooth: bool = True,
          qualities: Sequence = None, beats_per_bar: int = 4,
          span: float = None) -> Harmony:
    """按 resolution 拍一格推和弦, smooth 时走 Viterbi。key=None 自己找调, False 不用调。"""
    events = as_events(notes)
    if not events:
        raise ValueError("no notes to analyse")
    if resolution <= 0:
        raise ValueError("resolution must be positive")

    end = span if span is not None else max(o + d for o, d, _p, _v in events)
    n_slots = max(1, int(np.ceil(end / resolution - 1e-9)))

    if key is None:
        tonic, mode, _r = key_of(notes)
        key = (tonic, mode)
    elif key is False:
        key = None

    quals = tuple(qualities) if qualities else DEFAULT_QUALITIES
    states = [(r, q) for r in range(12) for q in quals]
    index = {s: i for i, s in enumerate(states)}

    emit = np.full((n_slots, len(states)), -10.0, dtype=float)
    basses = [None] * n_slots
    live = set()
    for i in range(n_slots):
        t0 = i * resolution
        w, bass_pc = _slot_evidence(events, t0, t0 + resolution, beats_per_bar)
        basses[i] = bass_pc
        if w.sum() <= 0:
            continue
        live.add(i)
        for chord, sc in chord_scores(w, key, bass_pc=bass_pc, qualities=quals):
            emit[i, index[(chord.root, chord.quality)]] = sc
    if not live:
        raise ValueError("no sounding notes in any slot")

    if smooth:
        path = _viterbi(emit, states, resolution, beats_per_bar, n_slots)
    else:
        path = [int(np.argmax(emit[i])) for i in range(n_slots)]

    wangge, prev = {}, None
    yanxu = None
    for i in range(n_slots):
        root, qual = states[path[i]]
        if i not in live:                     # 没声音的格子沿用上一个和弦
            if yanxu is None:
                continue
            hexian = yanxu
        else:
            hexianyin = {(root + iv) % 12 for iv in QUALITIES[qual]}
            b = basses[i]
            bass = b if (b is not None and b in hexianyin) else root
            hexian = Chord(root, qual, bass)
            yanxu = hexian
        if prev is None or hexian != prev:
            wangge[float(i * resolution)] = hexian
            prev = hexian
    if not wangge:
        raise ValueError("no chord could be inferred")
    return Harmony(wangge)


def _viterbi(emit, states, resolution, beats_per_bar, n_slots) -> list:
    # 保持同一个和弦不花钱, 免得在等价写法之间来回跳
    n = len(states)
    roots = np.array([s[0] for s in states])
    fifths = np.zeros((12, 12), dtype=float)
    for a in range(12):
        for b in range(12):
            fifths[a, b] = fifths_distance(a, b)

    base = _CHANGE + _VARIANCE * fifths[np.ix_(roots, roots)]
    same_root = roots[:, None] == roots[None, :]
    base = np.where(same_root, _QUALITY_ONLY, base)
    np.fill_diagonal(base, 0.0)

    delta = _SHARPNESS * emit[0]
    back = np.zeros((n_slots, n), dtype=np.int32)
    for i in range(1, n_slots):
        weak = 1.0 - metric_weight(i * resolution, beats_per_bar)
        trans = -(base * (1.0 + _WEAK_BEAT * weak))
        np.fill_diagonal(trans, 0.0)
        cand = delta[:, None] + trans
        best = np.argmax(cand, axis=0)
        back[i] = best
        delta = cand[best, np.arange(n)] + _SHARPNESS * emit[i]

    path = [0] * n_slots
    path[-1] = int(np.argmax(delta))
    for i in range(n_slots - 1, 0, -1):
        path[i - 1] = int(back[i, path[i]])
    return path


def evaluate(inferred: Harmony, truth: Harmony, span: float,
             *, resolution: float = 2.0) -> dict:
    """逐格跟参考和弦比: root / exact / full / pcset / subset 的比例, 加 n 和 errors。"""
    n = max(1, int(round(span / resolution)))
    root = exact = full = pcset = subset = 0
    errors = []
    for i in range(n):
        beat = i * resolution + 1e-9
        a, b = inferred.at(beat), truth.at(beat)
        if a.root == b.root:
            root += 1
            if a.quality == b.quality:
                exact += 1
                if a.bass == b.bass:
                    full += 1
        if a.tones == b.tones:
            pcset += 1
        if a.tones <= b.tones or b.tones <= a.tones:
            subset += 1
        if not (a.root == b.root and a.quality == b.quality):
            errors.append((i * resolution, a.label(), b.label()))
    return {"n": n,
            "root": root / n, "exact": exact / n, "full": full / n,
            "pcset": pcset / n, "subset": subset / n,
            "errors": errors}


if __name__ == "__main__":                                   # pragma: no cover
    from ..corpus import TUNES
    from ..theory import NAMES_FLAT

    # 曲库在仓库外面, 取前三首
    NAMES = tuple(sorted(TUNES))[:3]

    print("key finding on the corpus melodies:")
    for nm in NAMES:
        t = TUNES[nm]
        tonic, mode, r = key_of(t.melody)
        ok = "ok" if (NAMES_FLAT[tonic] == t.key and mode == "major") else "MISS"
        print(f"    {nm:14s} -> {NAMES_FLAT[tonic]:2s} {mode:5s} r={r:.3f}   "
              f"expected {t.key} major   [{ok}]")

    print("\nranked chords for {G, B, D, E} over a G bass, in G major:")
    for chord, sc in chord_scores({7: 2.0, 11: 1.0, 2: 1.0, 4: 1.0},
                                  key=(7, "major"), bass_pc=7)[:5]:
        print(f"    {chord.label():10s} {sc:+.3f}")

    # 四声部 MIDI 不放仓库里; CANTABILE_MIDI_DIR 指到放 a.mid b.mid c.mid 的目录
    # (按顺序对应上面三首)
    import os as _os
    _mdir = _os.environ.get("CANTABILE_MIDI_DIR", "")
    SRC = dict(zip(NAMES, (_os.path.join(_mdir, f) for f in ("a.mid", "b.mid", "c.mid"))))
    try:
        if not SRC:
            raise LookupError("corpus is empty")
        from ..ingest.midi import load
        sources = {}
        for nm, path in SRC.items():
            sources[nm] = load(path)[0]
    except Exception as exc:                       # missing mido, or no files
        print(f"\n(skipping the four-part MIDI validation: {exc})")
        raise SystemExit(0)

    print("\nkey finding on the four-part MIDI settings:")
    for nm in NAMES:
        for prof in ("KS", "CBMS"):
            tonic, mode, r = key_of(sources[nm], profile=prof)
            ok = "ok" if (NAMES_FLAT[tonic] == TUNES[nm].key
                          and mode == "major") else "MISS"
            print(f"    {nm:14s} {prof:5s} -> {NAMES_FLAT[tonic]:2s} {mode:5s} "
                  f"r={r:.3f}   [{ok}]")

    print("\nchord inference against the hand-authored half-bar grids:")
    header = f"    {'':14s} {'root':>6s} {'exact':>6s} {'full':>6s} {'pcset':>6s} {'subset':>7s}"
    for label, kw in (("per-slot best", dict(smooth=False)),
                      ("smoothed (Viterbi)", dict(smooth=True))):
        print(f"  {label}:")
        print(header)
        acc = {k: 0.0 for k in ("root", "exact", "full", "pcset", "subset")}
        total = 0
        for nm in NAMES:
            t = TUNES[nm]
            truth = Harmony({k: Chord.parse(*v) for k, v in t.harmony.items()})
            h = infer(sources[nm], resolution=2.0, span=t.length, **kw)
            r = evaluate(h, truth, t.length)
            for k in acc:
                acc[k] += r[k] * r["n"]
            total += r["n"]
            print(f"    {nm:14s} {r['root']:6.3f} {r['exact']:6.3f} {r['full']:6.3f} "
                  f"{r['pcset']:6.3f} {r['subset']:7.3f}")
        print(f"    {'OVERALL':14s} " + " ".join(
            f"{acc[k] / total:6.3f}" for k in ("root", "exact", "full", "pcset"))
            + f" {acc['subset'] / total:7.3f}   ({total} slots)")

    print("\nwhere the exact test fails (smoothed):")
    tally = {}
    for nm in NAMES:
        t = TUNES[nm]
        truth = Harmony({k: Chord.parse(*v) for k, v in t.harmony.items()})
        h = infer(sources[nm], resolution=2.0, span=t.length)
        for beat, _a, _b in evaluate(h, truth, t.length)["errors"]:
            got, want = h.at(beat + 1e-9), truth.at(beat + 1e-9)
            if got.tones == want.tones:
                k = "same notes, different name (Em7/G vs G6)"
            elif got.tones > want.tones:
                k = "thicker reading (Gmaj9 for Gmaj7)"
            elif got.tones < want.tones:
                k = "thinner reading (Dm for Dm7)"
            elif got.root == want.root:
                k = "same root, different quality"
            else:
                k = "different harmony"
            tally[k] = tally.get(k, 0) + 1
    n_err = sum(tally.values())
    for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"    {v:3d}  {v / n_err:4.0%}  {k}")
    same = sum(v for k, v in tally.items() if k.startswith(("thicker", "thinner",
                                                            "same notes")))
    print(f"    {n_err} failures out of {total} slots, of which {same} "
          f"({same / n_err:.0%}) are the same\n    harmony at a different "
          "thickness or spelling.")
