"""力度处理. 参考录音分布 p10 42 / 中位 63 / p90 88, p90 超过 92 就像砸琴."""
from __future__ import annotations

import math
from typing import Sequence, Tuple

from ..model import Performance, Voice

__all__ = ["voice_balance", "phrase_arc", "accents", "section_dynamics",
           "clamp", "quantiles"]

_TOL = 1.0 / 16.0
_LONG = 2.0
VEL_LO, VEL_HI = 1, 127


def _bounded(v: float, lo: int = VEL_LO, hi: int = VEL_HI) -> int:
    return int(max(lo, min(hi, round(v))))


def voice_balance(perf: Performance, melody_gain: int = 23,
                  inner_cut: int = 20, colour_cut: int = 26) -> Performance:
    # 低音不动, 压低音会显得空
    out = perf.copy()
    for n in out.notes:
        if n.voice is Voice.MELODY:
            n.vel = _bounded(n.vel + melody_gain)
        elif n.voice is Voice.INNER:
            n.vel = _bounded(n.vel - inner_cut)
        elif n.voice is Voice.COLOUR:
            n.vel = _bounded(n.vel - colour_cut)
    return out


def phrase_arc(perf: Performance, phrase: float = 16.0,
               depth: float = 5.0) -> Performance:
    if phrase <= 0:
        return perf.copy()
    out = perf.copy()
    for n in out.notes:
        weizhi = (n.onset % phrase) / phrase
        d = depth * (math.sin(math.pi * weizhi) - 0.5)
        if n.voice is not Voice.MELODY:
            d *= 0.6
        n.vel = _bounded(n.vel + d)
    return out


def accents(perf: Performance, meter: float = 4, strong: int = 3,
            medium: int = 2, long_note_bonus: int = 2,
            compound: bool = False) -> Performance:
    out = perf.copy()
    banxiao = meter / 2.0
    for n in out.notes:
        weizhi = n.onset % meter
        d = 0
        if weizhi < _TOL or meter - weizhi < _TOL:
            d += strong
        elif (meter % 2 == 0 or compound) and abs(weizhi - banxiao) < _TOL:
            d += medium
        if n.dur >= _LONG:
            d += long_note_bonus
        n.vel = _bounded(n.vel + d)
    return out


def section_dynamics(perf: Performance,
                     sections: Sequence[Tuple[float, float, float]]) -> Performance:
    """sections: [(start, end, base)], 每段平均力度调到 base."""
    out = perf.copy()
    for start, end, base in sections:
        xuanzhong = [n for n in out.notes if start <= n.onset < end]
        if not xuanzhong:
            continue
        pianyi = base - (sum(n.vel for n in xuanzhong) / len(xuanzhong))
        for n in xuanzhong:
            n.vel = _bounded(n.vel + pianyi)
    return out


def clamp(perf: Performance, lo: int = 20, hi: int = 108) -> Performance:
    out = perf.copy()
    for n in out.notes:
        n.vel = _bounded(n.vel, lo, hi)
    return out


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    i = q * (len(s) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def quantiles(perf: Performance) -> dict:
    v = [n.vel for n in perf.notes]
    xuanlv = [n.vel for n in perf.notes if n.voice is Voice.MELODY]
    banzou = [n.vel for n in perf.notes if n.voice is not Voice.MELODY]
    return {
        "n": len(v),
        "p10": round(_pct(v, 0.10), 1),
        "median": round(_pct(v, 0.50), 1),
        "p90": round(_pct(v, 0.90), 1),
        "min": min(v) if v else 0,
        "max": max(v) if v else 0,
        "melody_median": round(_pct(xuanlv, 0.50), 1),
        "accomp_median": round(_pct(banzou, 0.50), 1),
    }


if __name__ == "__main__":
    from . import demo_performance

    perf, _ = demo_performance()
    print("flat input:      ", quantiles(perf))

    p = voice_balance(perf)
    print("+voice_balance:  ", quantiles(p))
    p = phrase_arc(p)
    p = accents(p)
    p = section_dynamics(p, [(0.0, 16.0, 62.0), (16.0, 32.0, 66.0)])
    p = clamp(p)
    q = quantiles(p)
    print("+arc+accents+sec:", q)

    ref = {"p10": 42, "median": 63, "p90": 88}
    for k, mubiao in ref.items():
        got = q[k]
        print("  %-7s %5.1f   (reference %d, delta %+.1f)"
              % (k, got, mubiao, got - mubiao))
        assert abs(got - mubiao) <= 3.0, \
            "%s %.1f is off reference %d" % (k, got, mubiao)
    assert q["p90"] <= 92, "p90 above 92 reads as hammering"
    assert q["melody_median"] > q["accomp_median"] + 15, "melody is not in front"
    assert perf.notes[0].vel == 63, "input was mutated"
    print("dynamics ok")
