"""人性化处理: 时值, 力度, 衰减补偿, 踏板. 顺序见 house_style."""
from __future__ import annotations

from ..model import Note, Performance, TempoMap, Voice
from ..theory import Chord, Harmony, chord_near, pc_near, stack

from .timing import (TimingProfile, TIGHT, NATURAL, LOOSE,
                     humanize, phrase_rubato, ritardando, measure_spread)
from .dynamics import (voice_balance, phrase_arc, accents, section_dynamics,
                       clamp, quantiles)
from .decay import (decay_db, level_db, register_db, refresh_sustained_melody,
                    duck_under_sustain, sustain_report)
from .pedal import harmonic_pedal, pedal_events, clear_at

__all__ = [
    "TimingProfile", "TIGHT", "NATURAL", "LOOSE",
    "humanize", "phrase_rubato", "ritardando", "measure_spread",
    "voice_balance", "phrase_arc", "accents", "section_dynamics",
    "clamp", "quantiles",
    "decay_db", "level_db", "register_db", "refresh_sustained_melody",
    "duck_under_sustain", "sustain_report",
    "harmonic_pedal", "pedal_events", "clear_at",
    "demo_performance", "house_style", "selftest",
]


# 测试用的小曲子 (onset, dur, pitch)
_MELODY = [
    (0.0, 2.0, 72), (2.0, 1.0, 71), (3.0, 1.0, 69),
    (4.0, 2.0, 69), (6.0, 2.0, 72),
    (8.0, 1.0, 74), (9.0, 1.0, 72), (10.0, 2.0, 71),
    (12.0, 4.0, 72),
    (16.0, 2.0, 76), (18.0, 1.0, 74), (19.0, 1.0, 72),
    (20.0, 2.0, 74), (22.0, 2.0, 72),
    (24.0, 1.0, 71), (25.0, 1.0, 72), (26.0, 2.0, 74),
    (28.0, 4.0, 72),
]
_CHORDS = [("C", ""), ("F", ""), ("G", ""), ("C", ""),
           ("A", "m"), ("F", ""), ("G", "7"), ("C", "")]


def demo_performance(vel: int = 63, bpm: float = 84.0):
    """八小节 4/4 伴奏织体, 返回 (performance, harmony)."""
    perf = Performance(tempo=TempoMap([(0.0, bpm)]))
    harmony = Harmony({i * 4.0: Chord.parse(*c) for i, c in enumerate(_CHORDS)})

    for onset, dur, pitch in _MELODY:
        perf.add(Note(onset, dur, pitch, vel, Voice.MELODY, "melody"),
                 Note(onset, dur, pitch - 12, vel, Voice.MELODY, "melody.8vb"))

    for bar, (root, qual) in enumerate(_CHORDS):
        b0 = bar * 4.0
        ch = Chord.parse(root, qual)
        low = pc_near(ch.bass, 40, lo=36, hi=52)
        perf.add(Note(b0, 2.0, low, vel, Voice.BASS, "bass"),
                 Note(b0, 2.0, low + 12, vel, Voice.BASS, "bass.8va"),
                 Note(b0 + 2.0, 2.0, low + 12, vel, Voice.BASS, "bass"))
        tian = bar in (3, 7)
        step = 1.0 if tian else 2.0
        off = 0.0
        while off < 4.0:
            for p in stack(ch, 55, 67, 2, min_gap=3):
                perf.add(Note(b0 + off, step, p, vel, Voice.INNER, "inner"))
            off += step
        if bar % 2 == 1:
            perf.add(Note(b0 + 3.5, 0.5, chord_near(ch, 84), vel,
                          Voice.COLOUR, "colour"))
    return perf.sorted(), harmony


def house_style(perf: Performance, harmony: Harmony = None,
                seed: int = 1, sections=None) -> Performance:
    p = voice_balance(perf)
    p = phrase_arc(p)
    p = accents(p)
    if sections:
        p = section_dynamics(p, sections)
    p = clamp(p)
    p = refresh_sustained_melody(p)
    p = duck_under_sustain(p)
    if harmony is not None:
        p = harmonic_pedal(p, harmony)
    p = humanize(p, NATURAL, seed=seed)
    p.tempo = phrase_rubato(p.tempo)
    return p


def selftest() -> None:
    import runpy

    for mod in ("timing", "dynamics", "decay", "pedal"):
        print("=" * 64)
        print("cantabile.perform." + mod)
        print("=" * 64)
        runpy.run_module("cantabile.perform." + mod, run_name="__main__")

    print("=" * 64)
    print("the four passes together")
    print("=" * 64)
    yiduan, grid = demo_performance()
    BIANSHU = 4
    piece = Performance(tempo=yiduan.tempo)
    for v in range(BIANSHU):
        for n in yiduan.notes:
            piece.add(n.shifted(32.0 * v))
    piece.sorted()
    harmony = Harmony({k + 32.0 * v: c
                       for v in range(BIANSHU) for k, c in grid.grid.items()})
    duanluo = [(32.0 * v, 32.0 * (v + 1), b)
                for v, b in enumerate((60.0, 64.0, 58.0, 66.0))]

    perf = house_style(piece, harmony, seed=1, sections=duanluo)
    q = quantiles(perf)
    print("velocity   p10 %.0f  median %.0f  p90 %.0f   (reference 42 / 63 / 88)"
          % (q["p10"], q["median"], q["p90"]))
    huizong = {"median_ms": 0.0, "p95_ms": 0.0, "rolled_frac": 0.0}
    ZHONGZI = 8
    for seed in range(ZHONGZI):
        m = measure_spread(house_style(piece, harmony, seed=seed, sections=duanluo))
        for k in huizong:
            huizong[k] += m[k] / ZHONGZI
    print("spread     median %.1f ms  p95 %.1f ms  rolled %.1f %%"
          "   (reference 11 / 46 / 16 %%)"
          % (huizong["median_ms"], huizong["p95_ms"], 100 * huizong["rolled_frac"]))
    rep = sustain_report(perf)
    print("sustain   ", rep.stats)
    print(rep.summary())
    print("pedal      %d events, %d harmony changes"
          % (len(pedal_events(perf)), len(harmony)))

    assert abs(q["p10"] - 42) <= 4 and abs(q["median"] - 63) <= 4 \
        and abs(q["p90"] - 88) <= 4, q
    assert q["p90"] <= 92, "hammering"
    assert rep.ok() and rep.stats["worst_margin_db"] >= 0.0, rep.stats
    assert rep.stats["spans_with_hole"] == 0
    assert len(piece.notes) < len(perf.notes)        # refreshes were added
    assert len(perf.accompaniment()) > 0.85 * len(piece.accompaniment())
    assert yiduan.notes[0].vel == 63, "input was mutated"
    print("\nperform layer ok")


if __name__ == "__main__":  # pragma: no cover
    selftest()
