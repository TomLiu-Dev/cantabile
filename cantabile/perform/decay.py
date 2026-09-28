"""长音衰减补偿: 旋律长音会越来越弱, 伴奏一直在弹, 所以要补."""
from __future__ import annotations

import bisect
import math
from typing import Optional

from ..model import Finding, Note, Performance, Report, TempoMap, Voice

__all__ = ["decay_db", "register_db", "level_db",
           "refresh_sustained_melody", "duck_under_sustain", "sustain_report"]

TAU_AT_C4 = 3.00        # 秒, 中央 C
PITCH_HALVING = 19.5
TAU_MIN, TAU_MAX = 0.40, 16.0
PROMPT_DB = 4.0
PROMPT_TAU = 0.22
_NEPER_DB = 8.685889638065035          # 20 / ln(10)

VEL_SLOPE = 40.0
SALIENCE = 0.55         # A 计权用 55%
REG_LO, REG_HI = -9.0, 4.0

REFRESH_FLOOR = 20
RING = 10.0             # 拍
_JOIN = 0.05
INTEGRATION = 0.15      # 秒, 响度积分时间
ROLL_S = 0.12


def _tau(pitch):
    t = TAU_AT_C4 * (2.0 ** (-(pitch - 60) / PITCH_HALVING))
    return max(TAU_MIN, min(TAU_MAX, t))


def decay_db(pitch: int, seconds: float) -> float:
    t = float(seconds)
    if t <= 0.0:
        return 0.0
    kuaijiang = PROMPT_DB * (1.0 - math.exp(-t / PROMPT_TAU))
    return -(kuaijiang + _NEPER_DB * t / _tau(pitch))


def _a_weight(f):
    f2 = f * f
    fenzi = (12194.0 ** 2) * f2 * f2
    fenmu = ((f2 + 20.6 ** 2)
             * math.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
             * (f2 + 12194.0 ** 2))
    return 20.0 * math.log10(fenzi / fenmu) + 2.0


_A_REF = _a_weight(440.0)


def register_db(pitch: int) -> float:
    f = 440.0 * (2.0 ** ((pitch - 69) / 12.0))
    return max(REG_LO, min(REG_HI, SALIENCE * (_a_weight(f) - _A_REF)))


def level_db(pitch: int, vel: int, seconds: float = 0.0) -> float:
    return (VEL_SLOPE * math.log10(max(1, int(vel)) / 127.0)
            + register_db(pitch) + decay_db(pitch, seconds))


class _Span:
    __slots__ = ("pitch", "start", "end", "strikes")

    def __init__(self, note):
        self.pitch = note.pitch
        self.start = note.onset
        self.end = note.end
        self.strikes = [(note.onset, note.vel)]

    def extend(self, note):
        self.end = max(self.end, note.end)
        self.strikes.append((note.onset, note.vel))

    @property
    def dur(self):
        return self.end - self.start


def _spans(perf, min_dur=0.0, top_only=True):
    """把拆开的旋律音合并回长音."""
    changyin = []
    an_yingao = {}
    for n in sorted(perf.of(Voice.MELODY), key=lambda x: (x.onset, x.pitch)):
        cur = an_yingao.get(n.pitch)
        if cur is not None and n.onset <= cur.end + _JOIN:
            cur.extend(n)
        else:
            s = _Span(n)
            an_yingao[n.pitch] = s
            changyin.append(s)
    baoliu = []
    for s in changyin:
        if s.dur < min_dur - 1e-9:
            continue
        if top_only and any(
                o is not s and o.pitch > s.pitch
                and min(o.end, s.end) - max(o.start, s.start) > 0.5 * s.dur
                for o in changyin):
            continue            # 下方八度
        baoliu.append(s)
    return baoliu


def refresh_sustained_melody(perf: Performance, min_dur: float = 2.0,
                             step: float = 1.0, first_drop: int = 17,
                             per_step: int = 3,
                             tail_refresh: bool = True) -> Performance:
    """旋律长音每 step 拍轻轻重弹一次."""
    out = perf.copy()
    if step <= 0:
        return out
    xin_yinfu = []
    tihuan = set()
    for n in out.notes:
        if n.voice is not Voice.MELODY or n.dur < min_dur - 1e-9:
            continue
        fenjie = []
        k = 1
        while n.onset + k * step < n.end - 0.2:
            fenjie.append(n.onset + k * step)
            k += 1
        if tail_refresh:
            weiba = n.end - 0.5
            if (weiba > n.onset + 0.25
                    and (not fenjie or weiba > fenjie[-1] + 0.25)):
                fenjie.append(weiba)
        if not fenjie:
            continue
        bianjie = [n.onset] + fenjie + [n.end]
        tihuan.add(id(n))
        for i in range(len(bianjie) - 1):
            if i == 0:
                lidu = n.vel
            else:
                lidu = max(REFRESH_FLOOR,
                           n.vel - first_drop - per_step * (i - 1))
            tag = n.tag if i == 0 else (n.tag or "melody") + "+refresh"
            xin_yinfu.append(Note(bianjie[i], bianjie[i + 1] - bianjie[i], n.pitch,
                                  int(lidu), Voice.MELODY, tag))
    out.notes = [n for n in out.notes if id(n) not in tihuan] + xin_yinfu
    return out.sorted()


def duck_under_sustain(perf: Performance, grace: float = 0.7,
                       max_cut: int = 11, rate: float = 6.0,
                       thin_after: float = 2.2,
                       thin_above: int = 56,
                       max_hole: float = 1.25) -> Performance:
    """旋律长音下面的伴奏随时间压低, 高声部变稀, 但起音间隔不超过 max_hole."""
    out = perf.copy()
    changyin = _spans(out, min_dur=grace, top_only=True)
    if not changyin:
        return out
    kaishi = [s.start for s in changyin]

    def age_at(beat):
        i = bisect.bisect_right(kaishi, beat + 1e-9) - 1
        if i < 0:
            return -1.0
        s = changyin[i]
        return (beat - s.start) if beat < s.end - 1e-9 else -1.0

    shandiao = []
    for n in out.notes:
        if n.voice is Voice.MELODY:
            continue
        nianling = age_at(n.onset)
        if nianling <= grace:
            continue
        jian = min(float(max_cut), rate * (nianling - grace))
        n.vel = int(max(1, round(n.vel - jian)))
        if nianling >= thin_after and n.voice.droppable and n.pitch > thin_above:
            shandiao.append(n)

    # 按起音算空隙, 从最响的开始往回加
    ids = set(id(n) for n in shandiao)

    def widest(keep_ids):
        ons = sorted({round(n.onset, 3) for n in out.notes
                      if id(n) not in keep_ids})
        return max((b - a for a, b in zip(ons, ons[1:])), default=0.0)

    shunxu = sorted(shandiao, key=lambda n: (-n.vel, n.onset))
    while ids and widest(ids) > max_hole:
        for n in shunxu:
            if id(n) in ids:
                ids.discard(id(n))
                break
        else:
            break

    out.notes = [n for n in out.notes if id(n) not in ids]
    return out.sorted()


def _look_beats(tempo, beat):
    return INTEGRATION * tempo.bpm(beat) / 60.0


def sustain_report(perf: Performance, min_dur: float = 2.0,
                   step: float = 0.125, tempo: Optional[TempoMap] = None,
                   over_margin: float = 14.0,
                   tolerance_db: float = 3.0) -> Report:
    """旋律长音比伴奏高多少 dB. 低于 0 报 decay.sustain, 太高报 decay.overcut."""
    tempo = tempo or perf.tempo
    rep = Report()
    changyin = _spans(perf, min_dur=min_dur, top_only=True)
    banzou = sorted(perf.accompaniment(), key=lambda n: n.onset)
    banzou_qishi = [n.onset for n in banzou]

    miao_huancun = {}

    def sec(beat):
        k = round(beat, 6)
        if k not in miao_huancun:
            miao_huancun[k] = tempo.seconds(k)
        return miao_huancun[k]

    xuanlv_yinfu = sorted(perf.of(Voice.MELODY), key=lambda n: n.onset)
    xuanlv_qishi = [n.onset for n in xuanlv_yinfu]

    def loudest(notes, onsets, beat, look):
        lo = bisect.bisect_left(onsets, beat - RING)
        hi = bisect.bisect_right(onsets, beat + look + 1e-9)
        zuida = None
        for n in notes[lo:hi]:
            if beat < n.end:
                nianling = max(0.0, sec(beat) - sec(n.onset))
                lv = level_db(n.pitch, n.vel, nianling)
                if zuida is None or lv > zuida:
                    zuida = lv
        return zuida

    yuliang, zuicha, zuicha_pai = [], None, 0.0
    kongdong = 0
    for s in changyin:
        s_zuicha, s_zuicha_pai, s_kong = None, s.start, 0
        b = s.start
        while b < s.end - 1e-9:
            look = _look_beats(tempo, b)
            # 琶音时旋律在最后, 一个琶音长度内到的旋律也算
            xl = loudest(xuanlv_yinfu, xuanlv_qishi, b, look + _look_beats(tempo, b) * ROLL_S / INTEGRATION)
            hi = loudest(banzou, banzou_qishi, b, look)
            if xl is None:
                b += step
                continue
            if hi is None:
                s_kong += 1
            else:
                m = xl - hi
                yuliang.append(m)
                if s_zuicha is None or m < s_zuicha:
                    s_zuicha, s_zuicha_pai = m, b
            b += step
        if s_kong * step > 1.0:
            kongdong += 1
            rep.add(Finding("decay.texture", s.start,
                            "%.1f beats with no accompaniment under the melody"
                            % (s_kong * step), "warn"))
        if s_zuicha is None:
            continue
        if zuicha is None or s_zuicha < zuicha:
            zuicha, zuicha_pai = s_zuicha, s_zuicha_pai
        if s_zuicha < 0.0:
            rep.add(Finding("decay.sustain", s_zuicha_pai,
                            "melody %.1f dB under the accompaniment"
                            % (-s_zuicha),
                            "error" if -s_zuicha > tolerance_db else "warn"))
        elif s_zuicha > over_margin:
            rep.add(Finding("decay.overcut", s_zuicha_pai,
                            "melody never less than %.1f dB over the "
                            "accompaniment - hammering" % s_zuicha, "warn"))

    yuliang.sort()
    zhongwei = yuliang[len(yuliang) // 2] if yuliang else 0.0
    rep.stats.update({
        "held_notes": len(changyin),
        "samples": len(yuliang),
        "worst_margin_db": round(zuicha, 2) if zuicha is not None else None,
        "worst_at_beat": round(zuicha_pai, 3),
        "median_margin_db": round(zhongwei, 2),
        "spans_with_hole": kongdong,
    })
    return rep


if __name__ == "__main__":
    from . import demo_performance
    from .dynamics import accents, clamp, phrase_arc, quantiles, voice_balance

    print("decay_db, 1 s after the strike:")
    for p in (28, 40, 52, 60, 72, 84, 96):
        print("   pitch %3d  tau %5.2f s   1 s %6.2f dB   3 s %6.2f dB"
              % (p, _tau(p), decay_db(p, 1.0), decay_db(p, 3.0)))
    assert decay_db(40, 1.0) > decay_db(84, 1.0), "low notes must ring longer"
    assert decay_db(60, 0.0) == 0.0
    assert decay_db(60, 2.0) < decay_db(60, 1.0) < 0.0

    raw, _ = demo_performance()
    flat = sustain_report(raw)
    print("\nflat velocities (the failure case):", flat.stats)
    print(flat.summary(3))
    assert flat.stats["worst_margin_db"] < -15.0, flat.stats

    perf = clamp(accents(phrase_arc(voice_balance(raw))))
    before = sustain_report(perf)
    print("\nafter voice balance:", before.stats)
    assert before.stats["worst_margin_db"] < 2.0, before.stats

    mid = refresh_sustained_melody(perf)
    after_one = sustain_report(mid)
    print("after refresh:", after_one.stats)

    post = duck_under_sustain(mid)
    rep = sustain_report(post)
    print("after duck:   ", rep.stats)
    print(rep.summary())

    assert rep.stats["worst_margin_db"] >= 0.0, \
        "melody still under the accompaniment"
    assert rep.stats["worst_margin_db"] > after_one.stats["worst_margin_db"], \
        "ducking should add margin"
    assert rep.stats["worst_margin_db"] < 14.0, \
        "melody hammered over a gutted texture"
    assert rep.ok(), "sustain calibration failed"
    assert rep.stats["spans_with_hole"] == 0, "texture gutted"
    q = quantiles(post)
    print("velocities after decay passes:", q)
    assert q["max"] <= 108 and q["p90"] <= 92, "hammering"
    baoliu = len(post.accompaniment()) / float(len(perf.accompaniment()))
    print("accompaniment notes kept: %.0f%%" % (100 * baoliu))
    assert baoliu > 0.80, "accompaniment gutted"
    print("decay ok")
