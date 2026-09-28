"""KTH 规则组合, 按固定顺序执行并记录每条规则的效果.
参考: Friberg, Bresin & Sundberg 2006
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from ..model import Finding, Note, Performance, Report, Voice
from . import kth

__all__ = [
    "RuleSetting", "RuleSet", "ORDER", "available", "explain",
    "DEADPAN", "NOMINAL", "ROMANTIC", "ACCOMPANIMENT", "PALETTES",
]


ORDER = [
    "high_loud",
    "melodic_charge",
    "harmonic_charge",
    "double_duration",
    "inegales",
    "swing",
    "duration_contrast",
    "faster_uphill",
    "leap_tone_duration",
    "leap_articulation",
    "punctuation",
    "phrase_arch",
    "final_ritardando",
]


@dataclass
class RuleSetting:
    rule: str
    k: float = 1.0
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.rule not in kth.RULES:
            raise ValueError("unknown KTH rule %r (see rules.available())"
                             % (self.rule,))

    def __str__(self) -> str:
        fujia = ""
        if self.params:
            fujia = "  " + ", ".join("%s=%r" % kv
                                     for kv in sorted(self.params.items()))
        return "%-19s k=%-5g%s" % (self.rule, self.k, fujia)


def _setting(x):
    if isinstance(x, RuleSetting):
        return x
    if isinstance(x, str):
        return RuleSetting(x)
    if isinstance(x, (tuple, list)):
        if len(x) == 2:
            return RuleSetting(x[0], x[1])
        if len(x) == 3:
            return RuleSetting(x[0], x[1], dict(x[2]))
    raise TypeError("cannot read %r as a RuleSetting" % (x,))


@dataclass
class RuleSet:
    """settings 可以是 RuleSetting, 规则名, (name, k) 或 (name, k, params)."""
    name: str
    settings: list = field(default_factory=list)

    def __post_init__(self):
        self.settings = [_setting(s) for s in self.settings]

    def ordered(self) -> List[RuleSetting]:
        def key(item):
            i, s = item
            try:
                return (ORDER.index(s.rule), i)
            except ValueError:                           # pragma: no cover
                return (len(ORDER), i)
        return [s for _, s in sorted(enumerate(self.settings), key=key)]

    def apply(self, perf: Performance, *, trace: bool = True,
              preserve_length: bool = True, preserve_level: bool = True,
              **common) -> Performance:
        """common 里的参数 (harmony, key 等) 传给接受它的规则.
        preserve_length / preserve_level 把总长度和平均力度拉回原来的.
        """
        out = perf.copy()
        jilu = []
        for s in self.ordered():
            fn, kind = kth.RULES[s.rule]
            canshu = dict(s.params)
            for key, val in common.items():
                if key not in canshu and _accepts(fn, key):
                    canshu[key] = val
            zhiqian = out
            out = fn(zhiqian, s.k, **canshu)
            if trace:
                hang = _diff(zhiqian, out)
                hang["rule"] = s.rule
                hang["k"] = s.k
                hang["kind"] = kind
                hang["params"] = dict(s.params)
                jilu.append(hang)

        if preserve_length and perf.length > 1e-9 and out.length > 1e-9:
            zhiqian = out
            beishu = perf.length / out.length
            if abs(beishu - 1.0) > 1e-12:
                out = out.copy()
                for n in out.notes:
                    n.onset *= beishu
                    n.dur *= beishu
            if trace:
                jilu.append(_normrow(zhiqian, out, "normalise.length",
                                     "x%.5f on the beat axis" % beishu))

        if preserve_level:
            zhiqian = out
            db = _mean_gain_db(perf, out)
            if abs(db) > 1e-9:
                out = out.copy()
                f = 10.0 ** (-db / kth.VEL_SLOPE_DB)
                for n in out.notes:
                    n.vel = int(max(1, min(127, round(n.vel * f))))
            if trace:
                jilu.append(_normrow(zhiqian, out, "normalise.level",
                                     "%+.2f dB removed" % db))

        if trace:
            out.meta["kth_trace"] = jilu
            out.meta["kth_palette"] = self.name
        return out

    def with_k(self, factor: float) -> "RuleSet":
        return RuleSet("%s x%g" % (self.name, factor),
                       [RuleSetting(s.rule, s.k * factor, dict(s.params))
                        for s in self.settings])

    def __str__(self) -> str:
        biaoti = "RuleSet %r (%d rules, applied in this order)" % (
            self.name, len(self.settings))
        return "\n".join([biaoti] + ["  " + str(s) for s in self.ordered()])


def _accepts(fn, name):
    code = getattr(fn, "__code__", None)
    if code is None:                                     # pragma: no cover
        return False
    return name in code.co_varnames[:code.co_argcount + code.co_kwonlyargcount]


def available() -> List[str]:
    return [r for r in ORDER if r in kth.RULES] + \
           sorted(r for r in kth.RULES if r not in ORDER)


DEADPAN = RuleSet("deadpan", [])

# 全部 k=1, melodic_charge 原文是 2, 这里旋律有八度重叠所以用 1
NOMINAL = RuleSet("nominal", [
    ("high_loud", 1.0),
    ("melodic_charge", 1.0),
    ("harmonic_charge", 1.0),
    ("double_duration", 1.0),
    ("duration_contrast", 1.0),
    ("faster_uphill", 1.0),
    ("leap_tone_duration", 1.0),
    ("leap_articulation", 1.0),
    ("punctuation", 1.0),
    ("phrase_arch", 1.0),
    ("final_ritardando", 1.0, {"q": 3.0, "w": 0.65}),
])

ROMANTIC = RuleSet("romantic", [
    ("high_loud", 1.0),
    ("melodic_charge", 2.0),
    ("harmonic_charge", 1.5),
    ("double_duration", 1.0),
    ("duration_contrast", -0.5),
    ("leap_tone_duration", 1.0),
    ("leap_articulation", 0.5),
    ("punctuation", 1.0),
    ("phrase_arch", 2.0, {"turn": 0.4, "last": 1.4, "level_k": (1.0, 0.7)}),
    ("final_ritardando", 1.6, {"q": 3.0, "w": 0.55, "span": 6.0}),
])

# 伴奏用, 时值规则大约是 nominal 的三分之一, 主要靠力度
ACCOMPANIMENT = RuleSet("accompaniment", [
    ("high_loud", 0.25),
    ("melodic_charge", 0.6),
    ("harmonic_charge", 0.35),
    ("double_duration", 0.5),
    ("duration_contrast", 0.35),
    ("leap_articulation", 0.4),
    ("punctuation", 0.35),
    ("phrase_arch", 0.35, {"turn": 0.5, "level_k": (1.0, 0.4)}),
    ("final_ritardando", 0.7, {"q": 2.0, "w": 0.85, "span": 4.0}),
])

PALETTES = {p.name: p
            for p in (DEADPAN, NOMINAL, ROMANTIC, ACCOMPANIMENT)}


def _pct(xs, q):
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    i = q * (len(s) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (i - lo))


def _pair(before, after):
    if len(before.notes) == len(after.notes) and \
            all(a.pitch == b.pitch for a, b in zip(before.notes, after.notes)):
        return list(zip(before.notes, after.notes))
    jieguo, shengxia = [], list(after.notes)
    for a in before.notes:
        houxuan = [b for b in shengxia if b.pitch == a.pitch]
        if not houxuan:
            continue
        b = min(houxuan, key=lambda n: abs(n.onset - a.onset))
        shengxia.remove(b)
        jieguo.append((a, b))
    return jieguo


def _diff(before, after):
    """一条规则改了什么 (ms 和力度)."""
    peidui = _pair(before, after)
    bpm = before.tempo.bpm(0.0) or 90.0

    def ms(beats, at):
        return beats * 60000.0 / (before.tempo.bpm(at) or bpm)

    qishi, shichang, lidu = [], [], []
    for a, b in peidui:
        qishi.append(ms(b.onset - a.onset, a.onset))
        shichang.append(ms(b.dur - a.dur, a.onset))
        lidu.append(b.vel - a.vel)

    changdu = max(before.length, 1e-6)
    sudu = []
    for i in range(9):
        beat = changdu * i / 8.0
        t0 = before.tempo.bpm(beat)
        t1 = after.tempo.bpm(beat)
        sudu.append(0.0 if t0 == 0 else 100.0 * (t1 - t0) / t0)

    yidong = sum(1 for x in qishi if abs(x) > 0.5)
    gaichang = sum(1 for x in shichang if abs(x) > 0.5)
    gailidu = sum(1 for x in lidu if x != 0)
    return {
        "notes": len(peidui),
        "onsets_moved": yidong,
        "durations_changed": gaichang,
        "velocities_changed": gailidu,
        "onset_ms": _span(qishi),
        "dur_ms": _span(shichang),
        "vel": _span(lidu),
        "tempo_pct": [round(t, 2) for t in sudu],
        "touched": bool(yidong or gaichang or gailidu
                        or any(abs(t) > 0.01 for t in sudu)),
    }


def _mean_gain_db(before, after):
    g = [kth.VEL_SLOPE_DB * math.log10(max(1, b.vel) / float(max(1, a.vel)))
         for a, b in _pair(before, after)]
    return sum(g) / len(g) if g else 0.0


def _normrow(before, after, name, note):
    hang = _diff(before, after)
    hang["rule"] = name
    hang["k"] = 1.0
    hang["kind"] = "normalise"
    hang["params"] = {"note": note}
    return hang


def _span(xs):
    nz = [x for x in xs if abs(x) > 1e-9]
    return {
        "n": len(nz),
        "min": round(min(xs), 2) if xs else 0.0,
        "max": round(max(xs), 2) if xs else 0.0,
        "mean_abs": (round(sum(abs(x) for x in nz) / len(nz), 2)
                     if nz else 0.0),
        "p90_abs": (round(_pct([abs(x) for x in nz], 0.90), 2)
                    if nz else 0.0),
    }


def velocity_quantiles(perf: Performance) -> dict:
    v = [n.vel for n in perf.notes]
    return {"p10": round(_pct(v, 0.10), 1),
            "median": round(_pct(v, 0.50), 1),
            "p90": round(_pct(v, 0.90), 1),
            "min": min(v) if v else 0, "max": max(v) if v else 0}


def chord_spread_ms(perf: Performance, gap: float = 0.12) -> dict:
    """KTH 规则不散开和弦, 要 humanize 之后才不是 0."""
    yinfu = sorted(perf.notes, key=lambda n: n.onset)
    zu = []
    for n in yinfu:
        if zu and n.onset - zu[-1][-1].onset <= gap:
            zu[-1].append(n)
        else:
            zu.append([n])
    fensan = []
    for g in zu:
        if len(g) < 2:
            continue
        lo = min(n.onset for n in g)
        hi = max(n.onset for n in g)
        fensan.append((hi - lo) * 60000.0 / perf.tempo.bpm((lo + hi) / 2.0))
    return {"chords": len(fensan),
            "median_ms": round(_pct(fensan, 0.50), 1),
            "p95_ms": round(_pct(fensan, 0.95), 1)}


def explain(perf_before: Performance, perf_after: Performance) -> Report:
    """前后对比, 有 kth_trace 就按规则分开列."""
    rep = Report()
    zongti = _diff(perf_before, perf_after)
    rep.stats["notes"] = zongti["notes"]
    rep.stats["onsets_moved"] = zongti["onsets_moved"]
    rep.stats["durations_changed"] = zongti["durations_changed"]
    rep.stats["velocities_changed"] = zongti["velocities_changed"]
    rep.stats["onset_shift_ms"] = "mean %.1f  p90 %.1f  range %.1f..%.1f" % (
        zongti["onset_ms"]["mean_abs"], zongti["onset_ms"]["p90_abs"],
        zongti["onset_ms"]["min"], zongti["onset_ms"]["max"])
    rep.stats["duration_change_ms"] = (
        "mean %.1f  p90 %.1f  range %.1f..%.1f" % (
        zongti["dur_ms"]["mean_abs"], zongti["dur_ms"]["p90_abs"],
        zongti["dur_ms"]["min"], zongti["dur_ms"]["max"]))
    rep.stats["velocity_change"] = "mean %.1f  p90 %.1f  range %.0f..%.0f" % (
        zongti["vel"]["mean_abs"], zongti["vel"]["p90_abs"],
        zongti["vel"]["min"], zongti["vel"]["max"])
    rep.stats["tempo_pct_at_eighths"] = zongti["tempo_pct"]
    rep.stats["velocity_quantiles"] = velocity_quantiles(perf_after)
    rep.stats["chord_spread"] = chord_spread_ms(perf_after)

    genzong = perf_after.meta.get("kth_trace")
    if genzong:
        rep.stats["palette"] = perf_after.meta.get("kth_palette", "(unnamed)")
        for row in genzong:
            xiangqing = ("k=%-4g %-19s onsets %3d [%7.1f..%7.1f ms]"
                      "  durs %3d [%7.1f..%7.1f ms]  vels %3d [%+.0f..%+.0f]"
                      % (row["k"], row["kind"],
                         row["onsets_moved"],
                         row["onset_ms"]["min"], row["onset_ms"]["max"],
                         row["durations_changed"],
                         row["dur_ms"]["min"], row["dur_ms"]["max"],
                         row["velocities_changed"],
                         row["vel"]["min"], row["vel"]["max"]))
            if row["kind"] == "tempo":
                xiangqing += "  tempo %s %%" % (row["tempo_pct"],)
            if row["kind"] == "normalise":
                xiangqing += "  (%s)" % row["params"].get("note", "")
            rep.add(Finding("kth." + row["rule"], 0.0, xiangqing, "info"))
            if row["k"] != 0.0 and row["kind"] != "normalise" \
                    and not row["touched"]:
                rep.add(Finding(
                    "kth.inert", 0.0,
                    "%s ran at k=%g and changed nothing - missing harmony, or "
                    "its context condition never fired"
                    % (row["rule"], row["k"]), "warn"))
    else:
        rep.stats["palette"] = ("(no trace: use RuleSet.apply for a "
                                "per-rule breakdown)")

    dingdi = sum(1 for n in perf_after.notes if n.vel <= 1 or n.vel >= 127)
    if dingdi:
        rep.add(Finding("kth.clipped", 0.0,
                        "%d note(s) at the velocity rail; further level rules "
                        "have nowhere left to go" % dingdi, "warn"))

    an_shengbu = {}
    for n in perf_after.notes:
        an_shengbu.setdefault((n.voice, n.pitch), []).append(n)
    for key, notes in an_shengbu.items():
        notes.sort(key=lambda n: n.onset)
        for a, b in zip(notes, notes[1:]):
            if a.onset > b.onset + 1e-9:                 # pragma: no cover
                rep.add(Finding("kth.overlap", a.onset,
                                "timing rules reordered two notes of the same "
                                "voice", "error", (a, b)))
    return rep


if __name__ == "__main__":
    from ..model import TempoMap
    from ..theory import Chord, Harmony, pc_near, stack

    BPM = 88.0

    def build(vel=(64, 64, 64)):
        xuanlv = [(0.0, 1.0, 67), (1.0, 1.0, 69), (2.0, 2.0, 71),
               (4.0, 1.0, 72), (5.0, 1.0, 71), (6.0, 2.0, 69),
               (8.0, 1.0, 67), (9.0, 0.5, 69), (9.5, 0.5, 71), (10.0, 2.0, 72),
               (12.0, 4.0, 71),
               (16.0, 1.0, 72), (17.0, 1.0, 74), (18.0, 2.0, 76),
               (20.0, 1.0, 64), (21.0, 1.0, 65), (22.0, 2.0, 67),
               (24.0, 2.0, 69), (26.0, 1.0, 71), (27.0, 1.0, 72),
               (28.0, 4.0, 74),
               (32.0, 1.0, 74), (33.0, 1.0, 72), (34.0, 2.0, 71),
               (36.0, 1.0, 69), (37.0, 1.0, 71), (38.0, 2.0, 72),
               (40.0, 1.0, 71), (41.0, 1.0, 69), (42.0, 2.0, 67),
               (44.0, 4.0, 69),
               (48.0, 1.0, 67), (49.0, 1.0, 69), (50.0, 2.0, 71),
               (52.0, 1.0, 72), (53.0, 1.0, 74), (54.0, 2.0, 76),
               (56.0, 1.0, 74), (57.0, 1.0, 72), (58.0, 2.0, 71),
               (60.0, 4.0, 67)]
        chords = [("G", ""), ("C", ""), ("D", "7"), ("G", ""),
                  ("E", "m"), ("A", "m"), ("D", "7"), ("G", ""),
                  ("C", ""), ("G", ""), ("A", "m"), ("D", "7"),
                  ("G", ""), ("C", ""), ("D", "7"), ("G", "")]
        harmony = Harmony({i * 4.0: Chord.parse(*c)
                           for i, c in enumerate(chords)})
        p = Performance(tempo=TempoMap([(0.0, BPM)]))
        p.meta["harmony"] = harmony
        p.meta["key"] = "G"
        vm, vb, vi = vel
        for onset, dur, pitch in xuanlv:
            p.add(Note(onset, dur, pitch, vm, Voice.MELODY, "melody"))
        for bar, (root, qual) in enumerate(chords):
            b0 = bar * 4.0
            ch = Chord.parse(root, qual)
            low = pc_near(ch.bass, 43, lo=36, hi=52)
            p.add(Note(b0, 2.0, low, vb, Voice.BASS, "bass"),
                  Note(b0 + 2.0, 2.0, low + 12, vb, Voice.BASS, "bass"))
            for off in (0.0, 2.0):
                for pit in stack(ch, 55, 67, 2, min_gap=3):
                    p.add(Note(b0 + off, 2.0, pit, vi, Voice.INNER, "inner"))
        return p.sorted(), harmony

    def sig(p):
        return [(round(n.onset, 9), round(n.dur, 9), n.pitch, n.vel)
                for n in p.notes]

    def section(title):
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)

    perf, harmony = build()
    print("piece: %d notes, %.0f beats at %.0f bpm"
          % (len(perf.notes), perf.length, BPM))
    print("available rules:", ", ".join(available()))

    section("1. deadpan")
    d = DEADPAN.apply(perf)
    assert sig(d) == sig(perf), "empty palette changed the performance"
    assert d.tempo.bpm(30.0) == perf.tempo.bpm(30.0)
    zero = RuleSet("all at k=0", [RuleSetting(r, 0.0) for r in available()])
    d0 = zero.apply(perf, harmony=harmony, key="G")
    assert sig(d0) == sig(perf), "k=0 palette changed the performance"
    assert [round(d0.tempo.bpm(b), 9) for b in range(0, 65, 8)] == \
           [round(perf.tempo.bpm(b), 9) for b in range(0, 65, 8)]
    r0 = explain(perf, d0)
    assert (r0.stats["onsets_moved"] == 0
            and r0.stats["velocities_changed"] == 0)
    assert not [f for f in r0.findings if f.severity == "warn"
                and f.check == "kth.inert"], "k=0 must not be reported inert"
    print("DEADPAN and all-rules-at-k=0 leave every note unchanged")

    section("2. each rule alone at k=1, without normalisation")
    print("%-19s %6s %6s %6s   %-22s %-22s %s"
          % ("rule", "onset", "dur", "vel", "onset ms", "dur ms", "vel"))
    RAW = dict(preserve_length=False, preserve_level=False)
    yuqi = {
        "high_loud":          ("level",),
        "melodic_charge":     ("level", "timing"),
        "harmonic_charge":    ("level", "timing"),
        "double_duration":    ("timing",),
        "duration_contrast":  ("level", "timing"),
        "faster_uphill":      ("timing",),
        "leap_tone_duration": ("timing",),
        "leap_articulation":  ("articulation",),
        "punctuation":        ("timing", "articulation"),
        "phrase_arch":        ("level", "timing"),
        "inegales":           ("timing",),
        "swing":              ("timing",),
        "final_ritardando":   ("tempo",),
    }
    for name in available():
        one = RuleSet(name, [RuleSetting(name, 1.0)])
        out = one.apply(perf, harmony=harmony, key="G", **RAW)
        row = out.meta["kth_trace"][0]
        print("%-19s %6d %6d %6d   %8.1f..%-10.1f %8.1f..%-10.1f %+.0f..%+.0f"
              % (name, row["onsets_moved"], row["durations_changed"],
                 row["velocities_changed"],
                 row["onset_ms"]["min"], row["onset_ms"]["max"],
                 row["dur_ms"]["min"], row["dur_ms"]["max"],
                 row["vel"]["min"], row["vel"]["max"]))
        want = yuqi[name]
        assert row["touched"], "%s did nothing" % name
        if "level" not in want and "tempo" not in want:
            assert row["velocities_changed"] == 0, \
                "%s touched velocity" % name
        if "level" in want:
            assert row["velocities_changed"] > 0, \
                "%s touched no velocity" % name
        if "timing" in want:
            assert row["onsets_moved"] > 0, "%s moved no onset" % name
        if "timing" not in want and "tempo" not in want:
            assert row["onsets_moved"] == 0, "%s moved an onset" % name
        if want == ("articulation",):
            assert row["dur_ms"]["max"] <= 0.0, \
                "%s must only shorten notes" % name
        if want == ("tempo",):
            assert row["onsets_moved"] == 0 and row["velocities_changed"] == 0
            assert min(row["tempo_pct"]) < -1.0, \
                "%s did not slow anything" % name
        else:
            assert max(abs(t) for t in row["tempo_pct"]) < 1e-9, \
                "%s touched the tempo map" % name
        assert sig(perf) == sig(build()[0]), "%s mutated its input" % name

    hl = RuleSet("hl", [("high_loud", 1.0)]).apply(perf, **RAW)
    pairs = sorted((n.pitch, n.vel) for n in hl.notes)
    assert all(b >= a for (_, a), (_, b) in zip(pairs, pairs[1:])), \
        "high_loud must be monotone in pitch"

    mc = RuleSet("mc", [("melodic_charge", 1.0)]).apply(
        perf, harmony=harmony, **RAW)
    assert kth.melodic_charge_of(67, 7) == 0.0
    assert kth.melodic_charge_of(72, 7) == 2.5
    assert kth.melodic_charge_of(66, 7) == 5.0
    lifted_zero, lifted_any = 0, 0
    for a, b in zip(perf.notes, mc.notes):
        if a.voice is not Voice.MELODY:
            continue
        c = kth.melodic_charge_of(a.pitch, harmony.at(a.onset).root)
        if c == 0.0:
            lifted_zero += (b.vel != a.vel)
        else:
            lifted_any += (b.vel > a.vel)
    assert lifted_zero == 0, "melodic_charge lifted a chord root"
    assert lifted_any > 0

    no_h = perf.copy()
    no_h.meta.pop("harmony")
    quiet = RuleSet("mc", [("melodic_charge", 1.0)]).apply(no_h, **RAW)
    rq = explain(no_h, quiet)
    assert any(f.check == "kth.inert" for f in rq.findings), \
        "melodic_charge without harmony was not reported as inert"

    section("3. final ritardando: v(x) = [1 + (w^q - 1)x]^(1/q)")
    print("   x      " + "".join("%7.2f" % (i / 10.0) for i in range(11)))
    for q, w in ((2.0, 0.30), (3.0, 0.30), (2.0, 0.65), (3.0, 0.65)):
        cur = kth.ritardando_curve(k=1.0, q=q, w=w, n=11)
        print("q=%.0f w=%.2f" % (q, w) + "".join("%7.3f" % v for _, v in cur))
    c2 = kth.ritardando_curve(k=1.0, q=2.0, w=0.30, n=11)
    for x, v in c2:
        want = math.sqrt(1.0 + (0.30 ** 2 - 1.0) * x)
        assert abs(v - want) < 1e-12, "q=2 is not the square-root model"
    c3 = kth.ritardando_curve(k=1.0, q=3.0, w=0.30, n=11)
    assert all(b >= a - 1e-9 for (_, a), (_, b) in zip(c2, c3))
    assert abs(c2[0][1] - 1.0) < 1e-12 and abs(c2[-1][1] - 0.30) < 1e-9

    fr_params = {"q": 3.0, "w": 0.65, "span": 8.0}
    fr = RuleSet("fr", [("final_ritardando", 1.0, fr_params)]).apply(
        perf, **RAW)
    print("  " + "  ".join("beat %d: %.1f bpm" % (b, fr.tempo.bpm(b))
                           for b in (54, 56, 58, 60, 62, 64)))
    assert abs(fr.tempo.bpm(55.9) - BPM) < 1e-6
    assert abs(fr.tempo.bpm(64.0) - BPM * 0.65) < 1e-6

    section("4. palettes")
    for ps in (NOMINAL, ROMANTIC, ACCOMPANIMENT):
        out = ps.apply(perf, harmony=harmony, key="G")
        rep = explain(perf, out)
        q = rep.stats["velocity_quantiles"]
        tempo = rep.stats["tempo_pct_at_eighths"]
        print("\n--- %s" % ps.name)
        print(rep.summary(limit=20))
        print("  velocity p10/median/p90 = %.0f / %.0f / %.0f"
              % (q["p10"], q["median"], q["p90"]))
        print("  tempo deviation: %.1f%% .. %.1f%%" % (min(tempo), max(tempo)))
        print("  length %.2f beats (nominal %.2f)" % (out.length, perf.length))
        assert len(out.notes) == len(perf.notes)
        assert all(n.dur > 0.0 and n.onset >= 0.0 for n in out.notes)
        assert not rep.errors, rep.summary()
        assert not [f for f in rep.findings if f.check == "kth.inert"], \
            "%s has a rule that did nothing" % ps.name
        assert abs(out.length - perf.length) < 0.06 * perf.length, \
            "%s drifts the piece length" % ps.name

    nominal = NOMINAL.apply(perf, harmony=harmony, key="G")
    banzou = ACCOMPANIMENT.apply(perf, harmony=harmony, key="G")

    def max_shift(p):
        return max(abs(b.onset - a.onset) * 60000.0 / BPM
                   for a, b in zip(perf.notes, p.notes))

    print("\nmax onset shift: nominal %.1f ms, accompaniment %.1f ms"
          % (max_shift(nominal), max_shift(banzou)))
    assert max_shift(banzou) < max_shift(nominal), \
        "accompaniment palette is not the calmer one"
    assert banzou.tempo.bpm(64.0) > nominal.tempo.bpm(64.0), \
        "accompaniment palette slows more than nominal"

    # KTH 没有声部平衡, 先做一遍再和参考比
    section("4b. velocity profile against the reference (42 / 63 / 88)")

    def balance(p):
        out = p.copy()
        for n in out.notes:
            if n.voice is Voice.MELODY:
                n.vel = min(127, n.vel + 23)
            elif n.voice is Voice.INNER:
                n.vel = max(1, n.vel - 20)
            elif n.voice is Voice.COLOUR:
                n.vel = max(1, n.vel - 26)
        return out

    def level_to(p, target=63.0):
        out = p.copy()
        pianyi = target - sum(n.vel for n in out.notes) / float(len(out.notes))
        for n in out.notes:
            n.vel = int(max(20, min(108, round(n.vel + pianyi))))
        return out

    def show(label, p):
        q = velocity_quantiles(p)
        print("  %-40s %5.0f %8.0f %5.0f %5d"
              % (label, q["p10"], q["median"], q["p90"], q["max"]))
        return q

    print("  %-40s %5s %8s %5s %5s" % ("", "p10", "median", "p90", "max"))
    show("flat deadpan score", perf)
    for ps in (NOMINAL, ROMANTIC, ACCOMPANIMENT):
        show("KTH " + ps.name + " alone",
             ps.apply(perf, harmony=harmony, key="G"))
    show("voice_balance alone", balance(perf))
    zuizhong = {}
    for ps in (NOMINAL, ACCOMPANIMENT):
        out = level_to(balance(ps.apply(perf, harmony=harmony, key="G")))
        zuizhong[ps.name] = show("KTH " + ps.name + " + balance + level", out)

    q = zuizhong["accompaniment"]
    assert abs(q["p10"] - 42) <= 6 and abs(q["median"] - 63) <= 6 \
        and abs(q["p90"] - 88) <= 6, q
    assert q["p90"] <= 92, "accompaniment palette too loud at the top"

    section("5. chord spread")
    for label, p in (("deadpan", perf), ("nominal", nominal), ("accompaniment", banzou)):
        s = chord_spread_ms(p)
        print("  %-8s %3d chords, median %.1f ms, p95 %.1f ms"
              % (label, s["chords"], s["median_ms"], s["p95_ms"]))
    assert chord_spread_ms(nominal)["median_ms"] == 0.0

    section("6. corpus cross-check")
    try:
        from ..corpus import get, names
        mingdan = names()
        if not mingdan:
            raise ImportError("语料库是空的")
        tune = get(mingdan[-1])
        hp = Performance(tempo=TempoMap([(0.0, 76.0)]))
        grid = Harmony({float(b): Chord.parse(*c)
                        for b, c in tune.harmony.items()})
        hp.meta["harmony"] = grid
        hp.meta["key"] = tune.key
        for beat, pitch, dur in tune.melody:
            hp.add(Note(float(beat), float(dur), int(pitch), 64,
                        Voice.MELODY, "melody"))
        for beat, _hi, ch in grid.spans():
            end = min(beat + 2.0, tune.length)
            if beat >= tune.length:
                break
            low = pc_near(ch.bass, 45, lo=36, hi=55)
            hp.add(Note(beat, end - beat, low, 64, Voice.BASS, "bass"))
            for pit in stack(ch, 55, 70, 3, min_gap=3):
                hp.add(Note(beat, end - beat, pit, 64, Voice.INNER, "inner"))
        hp.sorted()
        out = ACCOMPANIMENT.apply(hp, harmony=grid, key=tune.key)
        rep = explain(hp, out)
        print("  %s (%s): %d notes, %.0f beats"
              % (tune.name, tune.key, len(hp.notes), hp.length))
        print("  " + rep.stats["onset_shift_ms"])
        print("  " + rep.stats["velocity_change"])
        assert not rep.errors and len(out.notes) == len(hp.notes)
        assert abs(out.length - hp.length) < 0.06 * hp.length
    except ImportError as exc:                           # pragma: no cover
        print("  skipped:", exc)

    print("\nrules ok")
