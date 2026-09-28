"""时值人性化: 起音偏移和速度变化.

参考录音: 和弦散开中位 11 ms, p95 46 ms, 16% 超过 25 ms.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from ..model import Performance, TempoMap, Voice

__all__ = [
    "TimingProfile", "TIGHT", "NATURAL", "LOOSE",
    "humanize", "phrase_rubato", "ritardando", "measure_spread",
]

_SIMULTANEOUS = 0.02     # 谱面上算同一个和弦
_CLUSTER = 0.12          # 听起来算同一个和弦
_ONBEAT = 0.02
ROLL_FINGERS = 3


@dataclass
class TimingProfile:
    """单位都是拍."""
    jitter_sd: float = 0.0055
    melody_lead: tuple = (0.0, 0.004)      # 大部分领先来自 key_travel
    key_travel: bool = True                # 弱音出声晚
    key_travel_scale: float = 0.65
    offbeat_lean: tuple = (0.002, 0.010)
    chord_spread: tuple = (0.002, 0.005)
    roll_prob: float = 0.03
    roll_spread: tuple = (0.018, 0.032)
    wide_roll_bonus: float = 0.155         # 跨度超过八度
    dense_roll_bonus: float = 0.08         # 4 个音以上
    full_roll_share: float = 0.50
    jitter_clip: float = 3.0


TIGHT = TimingProfile(
    jitter_sd=0.0028, melody_lead=(0.003, 0.008), offbeat_lean=(0.001, 0.005),
    chord_spread=(0.001, 0.003), roll_prob=0.012, roll_spread=(0.012, 0.022),
    wide_roll_bonus=0.05, dense_roll_bonus=0.02, full_roll_share=0.15)

NATURAL = TimingProfile()

LOOSE = TimingProfile(
    jitter_sd=0.0090, melody_lead=(0.010, 0.024), offbeat_lean=(0.004, 0.016),
    chord_spread=(0.003, 0.008), roll_prob=0.07, roll_spread=(0.022, 0.040),
    wide_roll_bonus=0.18, dense_roll_bonus=0.09, full_roll_share=0.60)


def _clusters(notes, gap):
    zu = []
    for n in sorted(notes, key=lambda x: (x.onset, x.pitch)):
        if zu and n.onset - zu[-1][-1].onset <= gap:
            zu[-1].append(n)
        else:
            zu.append([n])
    return zu


def _is_offbeat(beat: float) -> bool:
    return abs(beat - round(beat)) > _ONBEAT


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    i = q * (len(s) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def key_travel_ms(vel: float) -> float:
    """击弦延迟 ms, 拟合 (30, 140) 和 (100, 30). 参考: Goebl 2001"""
    v = max(8.0, float(vel))
    return 10890.0 * v ** -1.28


def _travel_beats(vel, ref, bpm):
    return (key_travel_ms(vel) - key_travel_ms(ref)) / 1000.0 * bpm / 60.0


def humanize(perf: Performance, profile: TimingProfile = NATURAL,
             seed: int = 0) -> Performance:
    rng = random.Random(seed)
    out = perf.copy()
    lidu_lb = sorted(n.vel for n in out.notes) or [64]
    zhongwei_lidu = lidu_lb[len(lidu_lb) // 2]

    for hexian in _clusters(out.notes, _SIMULTANEOUS):
        mingyi = min(n.onset for n in hexian)
        houyi = (rng.uniform(*profile.offbeat_lean)
                 if _is_offbeat(mingyi) else 0.0)

        hexian.sort(key=lambda n: n.pitch)
        pianyi = [0.0] * len(hexian)
        if len(hexian) > 1:
            kuadu = hexian[-1].pitch - hexian[0].pitch
            p = profile.roll_prob
            if kuadu > 12:
                p += profile.wide_roll_bonus
            if len(hexian) >= 4:
                p += profile.dense_roll_bonus
            xuanlv_i = [i for i, n in enumerate(hexian) if n.voice is Voice.MELODY]
            banzou_i = [i for i, n in enumerate(hexian)
                        if n.voice is not Voice.MELODY]
            if rng.random() < p:
                # 只琶音伴奏, 旋律留在拍上
                idx = banzou_i if len(banzou_i) >= 2 else list(range(len(hexian)))
                shouzhi = (len(idx)
                           if rng.random() < profile.full_roll_share
                           else ROLL_FINGERS)
                arp = idx[:shouzhi]
                leiji_lb, leiji = [0.0], 0.0
                for _ in arp[1:]:
                    leiji += rng.uniform(*profile.roll_spread)
                    leiji_lb.append(leiji)
                for k, i in enumerate(arp):
                    pianyi[i] = leiji_lb[k] - leiji  # 最后一个音落在拍上
                for i in idx[shouzhi:]:
                    pianyi[i] = 0.0
            else:
                for zu in (banzou_i, xuanlv_i):
                    if len(zu) < 2:
                        continue
                    leiji_lb, leiji = [0.0], 0.0
                    for _ in zu[1:]:
                        leiji += rng.uniform(*profile.chord_spread)
                        leiji_lb.append(leiji)
                    for k, i in enumerate(zu):
                        pianyi[i] = leiji_lb[k] - leiji / 2.0

        # 同一只手的和弦用同一个键速, 只有旋律单独算
        banzou_lidu = [n.vel for n in hexian if n.voice is not Voice.MELODY]
        shou_lidu = sum(banzou_lidu) / len(banzou_lidu) if banzou_lidu else None
        for n, off in zip(hexian, pianyi):
            dt = houyi + off
            if n.voice is Voice.MELODY:
                dt -= rng.uniform(*profile.melody_lead)
            if profile.key_travel and out.tempo is not None:
                v = n.vel if n.voice is Voice.MELODY or shou_lidu is None else shou_lidu
                dt += profile.key_travel_scale * _travel_beats(
                    v, zhongwei_lidu, out.tempo.bpm(n.onset))
            if profile.jitter_sd > 0:
                j = rng.gauss(0.0, profile.jitter_sd)
                lim = profile.jitter_clip * profile.jitter_sd
                dt += max(-lim, min(lim, j))
            n.onset = max(0.0, n.onset + dt)

    return out.sorted()


def _compose(tempo, fn):
    yuanlai = tempo.shape

    def shape(beat, bpm):
        if yuanlai is not None:
            bpm = yuanlai(beat, bpm)
        return fn(beat, bpm)

    return TempoMap(list(tempo.anchors), shape)


def phrase_rubato(tempo: TempoMap, phrase: float = 16.0, ease: float = 0.025,
                  tail: float = 0.16) -> TempoMap:
    """乐句中间稍推, 结尾 tail 部分放慢."""
    if phrase <= 0:
        return _compose(tempo, lambda b, v: v)
    zhuti = max(1e-6, 1.0 - tail)

    def fn(beat, bpm):
        weizhi = (beat % phrase) / phrase
        if weizhi < zhuti:
            f = 1.0 + ease * math.sin(math.pi * weizhi / zhuti)
        else:
            u = (weizhi - zhuti) / max(1e-6, tail)
            f = 1.0 - (ease * 2.2) * (u ** 1.4)
        return bpm * f

    return _compose(tempo, fn)


def ritardando(tempo: TempoMap, start: float, span: float,
               amount: float) -> TempoMap:
    """从 start 开始 span 拍内慢下 amount (0.25 = 25%)."""
    span = max(1e-6, span)

    def fn(beat, bpm):
        if beat <= start:
            return bpm
        u = min(1.0, (beat - start) / span)
        return bpm * (1.0 - amount * (u * u * (3.0 - 2.0 * u)))

    return _compose(tempo, fn)


def measure_spread(perf: Performance, tempo: Optional[TempoMap] = None,
                   rolled_ms: float = 25.0) -> dict:
    """和弦散开统计 (ms)."""
    tempo = tempo or perf.tempo
    fensan = []
    for g in _clusters(perf.notes, _CLUSTER):
        if len(g) < 2:
            continue
        lo = min(n.onset for n in g)
        hi = max(n.onset for n in g)
        bpm = tempo.bpm((lo + hi) / 2.0)
        fensan.append((hi - lo) * 60000.0 / bpm)
    n = len(fensan)
    gunda = sum(1 for s in fensan if s > rolled_ms)
    return {
        "chords": n,
        "median_ms": round(_pct(fensan, 0.50), 1),
        "p95_ms": round(_pct(fensan, 0.95), 1),
        "rolled_frac": round(gunda / n, 3) if n else 0.0,
    }


if __name__ == "__main__":
    from . import demo_performance

    base, _ = demo_performance()
    piece = Performance(tempo=base.tempo)
    for bian in range(4):
        for n in base.notes:
            piece.add(n.shifted(32.0 * bian))
    piece.sorted()

    print("quantised input:", measure_spread(piece))
    SEEDS = 16
    for name, prof in (("TIGHT", TIGHT), ("NATURAL", NATURAL), ("LOOSE", LOOSE)):
        heji = {"median_ms": 0.0, "p95_ms": 0.0, "rolled_frac": 0.0}
        for seed in range(SEEDS):
            h = humanize(piece, prof, seed=seed)
            assert len(h.notes) == len(piece.notes)
            assert all(n.onset >= 0.0 for n in h.notes)
            m = measure_spread(h)
            for k in heji:
                heji[k] += m[k] / SEEDS
        print("%-8s median %5.1f ms   p95 %5.1f ms   rolled %4.1f %%"
              % (name, heji["median_ms"], heji["p95_ms"], 100 * heji["rolled_frac"]))
        if prof is NATURAL:
            natural = heji

    print("reference       median  11.0 ms   p95  46.0 ms   rolled 16.0 %")
    assert abs(natural["median_ms"] - 11.0) <= 3.0, natural
    assert abs(natural["p95_ms"] - 46.0) <= 8.0, natural
    assert abs(100 * natural["rolled_frac"] - 16.0) <= 5.0, natural

    a = humanize(piece, NATURAL, seed=3)
    b = humanize(piece, NATURAL, seed=3)
    assert [n.onset for n in a.notes] == [n.onset for n in b.notes], "not deterministic"
    assert [n.onset for n in humanize(piece, NATURAL, seed=4).notes] != \
           [n.onset for n in a.notes], "seed ignored"
    assert piece.notes[0].onset == 0.0, "input was mutated"

    lingxian = []
    for beat in (4.0, 8.0, 20.0):
        near = [n for n in a.notes if abs(n.onset - beat) < 0.12]
        xuanlv = [n for n in near if n.voice is Voice.MELODY]
        banzou = [n for n in near if n.voice is not Voice.MELODY]
        if xuanlv and banzou:
            lingxian.append(sum(n.onset for n in banzou) / len(banzou)
                            - sum(n.onset for n in xuanlv) / len(xuanlv))
    print("melody lead over its own chord: %+.1f ms"
          % (1000 * sum(lingxian) / len(lingxian) * 60.0 / 84.0))
    assert sum(lingxian) > 0, "melody is not leading"

    t = TempoMap([(0.0, 84.0)])
    r = phrase_rubato(t)
    print("rubato bpm @0,4,8,14,15.5:",
          [round(r.bpm(b), 2) for b in (0, 4, 8, 14, 15.5)])
    assert r.bpm(8.0) > r.bpm(0.0) > r.bpm(15.5), "phrase does not breathe"
    assert r.seconds(16.0) > 0
    rit = ritardando(r, start=24.0, span=8.0, amount=0.22)
    print("rit bpm @24,28,32,40:", [round(rit.bpm(b), 2) for b in (24, 28, 32, 40)])
    assert abs(rit.bpm(32.0) - r.bpm(32.0) * 0.78) < 0.01, "rit amount wrong"
    assert rit.bpm(20.0) == r.bpm(20.0), "rit leaked backwards"
    print("timing ok")
