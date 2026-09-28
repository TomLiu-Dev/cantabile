"""KTH 演奏规则. k=0 完全不变, 1 是标准值, 负数反向.
参考: Friberg 1991, Friberg/Bresin/Sundberg 2006
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from ..model import Note, Performance, TempoMap, Voice

__all__ = [
    "phrase_arch", "final_ritardando", "high_loud",
    "duration_contrast", "faster_uphill",
    "double_duration", "inegales", "swing",
    "punctuation", "leap_articulation", "leap_tone_duration",
    "melodic_charge", "harmonic_charge",
    "MELODIC_CHARGE", "DURATION_CONTRAST_BP", "VEL_SLOPE_DB",
    "melodic_charge_of", "harmonic_charge_of", "ritardando_curve",
    "RULES",
]

VEL_SLOPE_DB = 40.0      # 和 decay.VEL_SLOPE 一样
VEL_LO, VEL_HI = 1, 127

_TOL = 0.02
_MIN_SOUND = 0.02
_EPS = 1e-9


# 根音以上半音数 -> C_mel
MELODIC_CHARGE = {
    0: 0.0,
    7: 1.0,
    2: 2.0,
    9: 3.0,
    4: 4.0,
    11: 5.0,
    6: 6.0,
    1: 6.5,
    8: 5.5,
    3: 4.5,
    10: 3.5,
    5: 2.5,
}

# (IOI ms, dDR ms), 后两行是自己补的
DURATION_CONTRAST_BP = [
    (30.0, 0.0),
    (200.0, -16.5),
    (400.0, -10.5),
    (600.0, 0.0),
    (1200.0, 16.5),
    (2400.0, 16.5),
]

_DC_DB_PER_MS = 0.05


def _vel(v):
    return int(max(VEL_LO, min(VEL_HI, round(v))))


def _gain(vel, db):
    if db == 0.0:
        return int(vel)
    return _vel(vel * (10.0 ** (db / VEL_SLOPE_DB)))


def _bpm(perf, beat):
    return perf.tempo.bpm(beat)


def _ms_to_beats(perf, ms, beat):
    return ms / 1000.0 * _bpm(perf, beat) / 60.0


def _beats_to_ms(perf, beats, at):
    return beats * 60000.0 / _bpm(perf, at)


def _interp(table, x):
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table, table[1:]):
        if x0 <= x <= x1:
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return y0 + (y1 - y0) * t
    return table[-1][1]


def _compose(tempo, fn):
    yuanlai = tempo.shape

    def shape(beat, bpm):
        if yuanlai is not None:
            bpm = yuanlai(beat, bpm)
        return fn(beat, bpm)

    return TempoMap(list(tempo.anchors), shape)


class _Event:

    __slots__ = ("beat", "idx", "pitch", "melodic", "ioi", "dur")

    def __init__(self, beat, idx, pitch, melodic, ioi, dur):
        self.beat = beat
        self.idx = idx            # perf.notes 的下标
        self.pitch = pitch
        self.melodic = melodic    # 旋律音的下标
        self.ioi = ioi
        self.dur = dur

    def __repr__(self):                                  # pragma: no cover
        return "<ev %.3f p%d ioi %.3f>" % (self.beat, self.pitch, self.ioi)


def _events(perf):
    """按起音分组, 每组取旋律音高 (Melodic sync)."""
    shunxu = sorted(range(len(perf.notes)),
                   key=lambda i: (perf.notes[i].onset, -perf.notes[i].pitch))
    zu = []
    for i in shunxu:
        n = perf.notes[i]
        if zu and n.onset - perf.notes[zu[-1][0]].onset <= _TOL:
            zu[-1].append(i)
        else:
            zu.append([i])

    evs = []
    for g in zu:
        beat = min(perf.notes[i].onset for i in g)
        xuanlv = [i for i in g if perf.notes[i].voice is Voice.MELODY]
        if not xuanlv:
            zuigao = max(perf.notes[i].pitch for i in g)
            xuanlv = [i for i in g if perf.notes[i].pitch == zuigao]
        pitch = max(perf.notes[i].pitch for i in xuanlv)
        dur = max(perf.notes[i].dur for i in g)
        evs.append(_Event(beat, g, pitch, xuanlv, 0.0, dur))

    for a, b in zip(evs, evs[1:]):
        a.ioi = b.beat - a.beat
    if evs:
        evs[-1].ioi = evs[-1].dur
    return evs


def _warp(perf, evs, pianyi):
    """evs[i] 的 IOI 加长 pianyi[i] 拍, 整体做单调时间变形."""
    if not evs or not any(abs(d) > 1e-12 for d in pianyi):
        return perf.copy()

    weiyi = [0.0] * len(evs)
    lashen = [1.0] * len(evs)
    leiji = 0.0
    for i, ev in enumerate(evs):
        weiyi[i] = leiji
        kuadu = max(ev.ioi, _EPS)
        d = max(pianyi[i], -0.9 * kuadu)
        lashen[i] = (kuadu + d) / kuadu
        leiji += d

    pai = [ev.beat for ev in evs]

    def at(t):
        if t <= pai[0]:
            return t + weiyi[0]
        lo, hi = 0, len(pai) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if pai[mid] <= t:
                lo = mid
            else:
                hi = mid - 1
        return pai[lo] + weiyi[lo] + (t - pai[lo]) * lashen[lo]

    out = perf.copy()
    for n in out.notes:
        a = at(n.onset)
        b = at(n.onset + n.dur)
        n.onset = max(0.0, a)
        n.dur = max(_MIN_SOUND, b - a)
    return out


def _shorten(perf, dro):
    if not dro:
        return perf.copy()
    out = perf.copy()
    for i, gap in dro.items():
        if gap <= 0.0:
            continue
        n = out.notes[i]
        n.dur = max(_MIN_SOUND, n.dur - gap)
    return out


def _harmony(perf, harmony=None):
    if harmony is not None:
        return harmony
    return perf.meta.get("harmony")


def _chord_root_at(harmony, beat):
    if harmony is None:
        return None
    try:
        return harmony.at(beat).root % 12
    except Exception:                                    # pragma: no cover
        return None


def _key_root(perf, harmony, key=None):
    from ..theory import pc_of
    for houxuan in (key, perf.meta.get("key")):
        if houxuan is not None:
            try:
                return pc_of(houxuan)
            except Exception:
                pass
    if harmony is not None:
        try:
            return harmony.at(0.0).root % 12
        except Exception:                                # pragma: no cover
            return None
    return None


def melodic_charge_of(pitch: int, root: int) -> float:
    return MELODIC_CHARGE[(int(pitch) - int(root)) % 12]


def harmonic_charge_of(chord, key_root: int) -> float:
    """C_harm = 根三五音 C_mel 的均方根 (原文公式看不清, 这是重建的)."""
    yin = sorted(chord.tones, key=lambda pc: (pc - chord.root) % 12)
    xuanzhong = yin[:3] if len(yin) >= 3 else yin
    cs = [MELODIC_CHARGE[(pc - int(key_root)) % 12] for pc in xuanzhong]
    if not cs:
        return 0.0
    return math.sqrt(sum(c * c for c in cs) / len(cs))


def _fixed_phrases(length, span):
    if span <= 0 or length <= 0:
        return []
    yuju, b = [], 0.0
    while b < length - _EPS:
        yuju.append((b, min(length, b + span)))
        b += span
    return yuju


def _detected_phrases(perf):
    try:
        from ..analyze import structure
    except Exception:
        return None
    try:
        xuanlv = perf.melody() or sorted(perf.notes, key=lambda n: n.onset)
        ph = structure.phrases(xuanlv)
    except Exception:
        return None
    if not ph or len(ph) < 2:
        return None
    return [(float(a), float(b)) for a, b in ph]


def _phrase_levels(perf, phrases=None, levels=(16.0, 4.0)):
    if phrases is not None:
        return [[(float(a), float(b)) for a, b in phrases]]
    changdu = perf.length
    jiance = _detected_phrases(perf)
    if jiance:
        # 两两合并成上一层
        shangceng = []
        for i in range(0, len(jiance), 2):
            kuai = jiance[i:i + 2]
            shangceng.append((kuai[0][0], kuai[-1][1]))
        return [shangceng, jiance] if len(shangceng) >= 2 else [jiance]
    return [_fixed_phrases(changdu, s) for s in levels if s > 0]


def phrase_arch(perf: Performance, k: float = 1.0, *,
                phrases=None, levels: Sequence[float] = (16.0, 4.0),
                level_k: Sequence[float] = (1.0, 0.6),
                acc: float = 1.0, turn: float = 0.33, power: float = 2.0,
                last: float = 1.0, amp: float = 1.0,
                sl_acc_coeff: float = 0.5,
                sl_rit_coeff: float = 1.0,
                normalize: bool = True) -> Performance:
    """Phrase arch, Friberg 1995. 乐句中间快而响, 结尾慢而轻."""
    if k == 0.0:
        return perf.copy()
    evs = _events(perf)
    if not evs:
        return perf.copy()

    ddr = [0.0] * len(evs)
    dl = [0.0] * len(evs)

    for li, ph in enumerate(_phrase_levels(perf, phrases, levels)):
        kk = k * (level_k[li] if li < len(level_k) else level_k[-1])
        if kk == 0.0 or not ph:
            continue
        for (a, b) in ph:
            kuadu = b - a
            if kuadu <= 0:
                continue
            neibu = [i for i, ev in enumerate(evs)
                      if a - _EPS <= ev.beat < b - _EPS]
            if not neibu:
                continue
            zhuanzhe = a + max(0.0, min(1.0, turn)) * kuadu
            ddr_ceng = dict((i, 0.0) for i in neibu)
            for i in neibu:
                t = evs[i].beat
                if t < zhuanzhe:
                    x = (t - a) / max(_EPS, zhuanzhe - a)
                    d = 0.1 * kk * acc * (1.0 - x) ** power
                    ddr[i] += d
                    ddr_ceng[i] += d
                    dl[i] += (-sl_acc_coeff * kk * acc * amp
                              * (1.0 - x) ** power)
                else:
                    x = (t - zhuanzhe) / max(_EPS, b - zhuanzhe)
                    d = 0.2 * kk * x ** power
                    ddr[i] += d
                    ddr_ceng[i] += d
                    dl[i] += -sl_rit_coeff * kk * amp * x ** power
            if last != 1.0:
                ddr[neibu[-1]] -= ddr_ceng[neibu[-1]] * (1.0 - last)
                ddr_ceng[neibu[-1]] *= last
            if normalize:
                quanzhong = sum(evs[i].ioi for i in neibu)
                if quanzhong > _EPS:
                    pingjun = sum(ddr_ceng[i] * evs[i].ioi
                               for i in neibu) / quanzhong
                    for i in neibu:
                        ddr[i] -= pingjun

    out = _warp(perf, evs, [ddr[i] * evs[i].ioi for i in range(len(evs))])
    for i, ev in enumerate(evs):
        if dl[i] == 0.0:
            continue
        for j in ev.idx:
            out.notes[j].vel = _gain(out.notes[j].vel, dl[i])
    return out


def ritardando_curve(k: float = 1.0, q: float = 3.0, w: float = 0.65,
                     n: int = 11) -> List[Tuple[float, float]]:
    """返回 [(x, v)]."""
    w_eff = 1.0 - k * (1.0 - w)
    w_eff = max(0.02, min(1.0, w_eff))
    quxian = []
    for i in range(n):
        x = i / float(n - 1) if n > 1 else 1.0
        quxian.append((x, (1.0 + (w_eff ** q - 1.0) * x) ** (1.0 / q)))
    return quxian


def final_ritardando(perf: Performance, k: float = 1.0, *,
                     q: float = 3.0, w: float = 0.65,
                     span: Optional[float] = None,
                     start: Optional[float] = None) -> Performance:
    """v(x) = [1 + (w**q - 1) x] ** (1/q), 改的是 tempo 不是音符."""
    if k == 0.0:
        return perf.copy()
    changdu = perf.length
    if changdu <= 0:
        return perf.copy()
    if span is None:
        span = min(4.0, changdu * 0.25) if changdu > 0 else 1.0
    span = max(_EPS, float(span))
    if start is None:
        start = max(0.0, changdu - span)

    w_eff = max(0.02, min(1.0, 1.0 - k * (1.0 - w)))
    qq = max(0.25, float(q))
    a = w_eff ** qq - 1.0

    def fn(beat: float, bpm: float) -> float:
        if beat <= start:
            return bpm
        x = min(1.0, (beat - start) / span)
        return bpm * (1.0 + a * x) ** (1.0 / qq)

    out = perf.copy()
    out.tempo = _compose(out.tempo, fn)
    return out


def high_loud(perf: Performance, k: float = 1.0, *,
              ref: int = 60) -> Performance:
    """dL = k * (N - ref) / 4 dB"""
    if k == 0.0:
        return perf.copy()
    out = perf.copy()
    for n in out.notes:
        n.vel = _gain(n.vel, k * (n.pitch - ref) / 4.0)
    return out


def _double_duration_pairs(perf, evs, tol=0.08):
    """IOI 比例 2:1 的相邻事件."""
    peidui = []
    for i in range(len(evs) - 1):
        a, b = evs[i].ioi, evs[i + 1].ioi
        if a <= _EPS or b <= _EPS:
            continue
        r = a / b
        if abs(r - 2.0) < tol * 2.0 or abs(r - 0.5) < tol:
            duan = min(a, b)
            if _beats_to_ms(perf, duan, evs[i].beat) < 1000.0:
                peidui.append((i, i + 1))
    return peidui


def duration_contrast(perf: Performance, k: float = 1.0, *,
                      respect_double_duration: bool = True) -> Performance:
    """短的更短, 长的更长. DDC 1"""
    if k == 0.0:
        return perf.copy()
    evs = _events(perf)
    if not evs:
        return perf.copy()

    tiaoguo = set()
    if respect_double_duration:
        for i, j in _double_duration_pairs(perf, evs):
            tiaoguo.add(i)
            tiaoguo.add(j)

    pianyi = [0.0] * len(evs)
    dl = [0.0] * len(evs)
    for i, ev in enumerate(evs):
        if i in tiaoguo:
            continue
        ms = _beats_to_ms(perf, ev.ioi, ev.beat)
        d_ms = k * _interp(DURATION_CONTRAST_BP, ms)
        if d_ms == 0.0:
            continue
        pianyi[i] = _ms_to_beats(perf, d_ms, ev.beat)
        dl[i] = _DC_DB_PER_MS * d_ms

    out = _warp(perf, evs, pianyi)
    for i, ev in enumerate(evs):
        if dl[i] == 0.0:
            continue
        for j in ev.idx:
            out.notes[j].vel = _gain(out.notes[j].vel, dl[i])
    return out


def faster_uphill(perf: Performance, k: float = 1.0) -> Performance:
    """上行音缩短 2k ms. GMI 1C"""
    if k == 0.0:
        return perf.copy()
    evs = _events(perf)
    if len(evs) < 3:
        return perf.copy()
    pianyi = [0.0] * len(evs)
    for i in range(1, len(evs) - 1):
        if evs[i - 1].pitch < evs[i].pitch < evs[i + 1].pitch:
            pianyi[i] = _ms_to_beats(perf, -2.0 * k, evs[i].beat)
    return _warp(perf, evs, pianyi)


def leap_tone_duration(perf: Performance, k: float = 1.0, *,
                       min_leap: int = 3) -> Performance:
    """孤立跳进. 上行的符号按 Director Musices. GMI 1B"""
    if k == 0.0:
        return perf.copy()
    evs = _events(perf)
    if len(evs) < 2:
        return perf.copy()
    pianyi = [0.0] * len(evs)
    for i in range(len(evs) - 1):
        dn = evs[i + 1].pitch - evs[i].pitch
        if abs(dn) < min_leap:
            continue
        qian = evs[i - 1].pitch if i > 0 else evs[i].pitch
        hou = evs[i + 2].pitch if i + 2 < len(evs) else evs[i + 1].pitch
        if abs(evs[i].pitch - qian) >= min_leap:
            continue
        if abs(hou - evs[i + 1].pitch) >= min_leap:
            continue
        fudu = 4.2 * abs(dn) * k
        if dn > 0:
            pianyi[i] += _ms_to_beats(perf, -fudu, evs[i].beat)
            pianyi[i + 1] += _ms_to_beats(perf, fudu, evs[i + 1].beat)
        else:
            pianyi[i] += _ms_to_beats(perf, fudu, evs[i].beat)
            pianyi[i + 1] += _ms_to_beats(perf, -2.4 * abs(dn) * k,
                                         evs[i + 1].beat)
    return _warp(perf, evs, pianyi)


def double_duration(perf: Performance, k: float = 1.0) -> Performance:
    """2:1 的一对音缩小对比, 总长不变. DDC 2B"""
    if k == 0.0:
        return perf.copy()
    evs = _events(perf)
    if len(evs) < 2:
        return perf.copy()
    pianyi = [0.0] * len(evs)
    for i, j in _double_duration_pairs(perf, evs):
        si, li = (i, j) if evs[i].ioi < evs[j].ioi else (j, i)
        duan_ms = _beats_to_ms(perf, evs[si].ioi, evs[si].beat)
        d = _ms_to_beats(perf, 0.12 * duan_ms * k, evs[si].beat)
        pianyi[si] += d
        pianyi[li] -= d
    return _warp(perf, evs, pianyi)


def inegales(perf: Performance, k: float = 1.0, *,
             unit: float = 0.5, amount: float = 0.22) -> Performance:
    """等长的两个音弹成长短. GMI 3"""
    if k == 0.0 or amount == 0.0:
        return perf.copy()
    evs = _events(perf)
    if len(evs) < 2:
        return perf.copy()
    pianyi = [0.0] * len(evs)
    yidui = 2.0 * unit
    for i in range(len(evs) - 1):
        a, b = evs[i], evs[i + 1]
        if abs(a.ioi - unit) > 1e-6 or abs(b.ioi - unit) > 1e-6:
            continue
        # 要从强拍开始
        if abs((a.beat / yidui) - round(a.beat / yidui)) > 1e-6:
            continue
        d = amount * k * a.ioi
        pianyi[i] += d
        pianyi[i + 1] -= d
    return _warp(perf, evs, pianyi)


def swing(perf: Performance, k: float = 1.0, *,
          unit: float = 0.5, ratio: float = 1.56) -> Performance:
    r = max(1.0, float(ratio))
    return inegales(perf, k, unit=unit, amount=(r - 1.0) / (r + 1.0))


def _boundaries(perf, evs, yuzhi):
    """小乐句的结尾. 优先用 LBDM, 否则看跳进/长音/休止."""
    try:
        from ..analyze import structure
        xuanlv = perf.melody() or sorted(perf.notes, key=lambda n: n.onset)
        lunkuo = structure.lbdm(xuanlv)
        an_pai = {}
        for beat, s in lunkuo:
            an_pai[round(float(beat), 6)] = float(s)
        bianjie = []
        for i, ev in enumerate(evs[:-1]):
            s = an_pai.get(round(ev.beat, 6))
            if s is not None and s >= yuzhi:
                bianjie.append(i)
        if bianjie:
            return bianjie
    except Exception:
        pass

    bianjie = []
    for i in range(len(evs) - 1):
        a, b = evs[i], evs[i + 1]
        chuangkou = evs[max(0, i - 2):i + 3]
        iois = [e.ioi for e in chuangkou if e.ioi > 0]
        zhongwei = sorted(iois)[len(iois) // 2] if iois else a.ioi
        tiaojin = abs(b.pitch - a.pitch) >= 5
        gengchang = a.ioi > 1.5 * zhongwei + _EPS
        xiuzhi = a.dur < 0.75 * a.ioi - _EPS
        if tiaojin or gengchang or xiuzhi:
            bianjie.append(i)
    return bianjie


def punctuation(perf: Performance, k: float = 1.0, *,
                threshold: float = 0.45,
                lengthen_ms: float = 40.0,
                micropause_ms: float = 80.0) -> Performance:
    """小乐句最后一个音加长并留一点空. Friberg 1998"""
    if k == 0.0:
        return perf.copy()
    evs = _events(perf)
    if len(evs) < 3:
        return perf.copy()

    biaoji = _boundaries(perf, evs, threshold)
    if not biaoji:
        return perf.copy()

    pianyi = [0.0] * len(evs)
    for i in biaoji:
        pianyi[i] += _ms_to_beats(perf, lengthen_ms * k, evs[i].beat)
    out = _warp(perf, evs, pianyi)

    dro = {}
    for i in biaoji:
        gap = _ms_to_beats(perf, micropause_ms * k, evs[i].beat)
        for j in evs[i].melodic:
            dro[j] = max(dro.get(j, 0.0), gap)
    return _shorten(out, dro)


def leap_articulation(perf: Performance, k: float = 1.0, *,
                      min_leap: int = 3, cap_ms: float = 100.0) -> Performance:
    """跳进之间加微小停顿, 公式是重建的. GMI 1A"""
    if k == 0.0:
        return perf.copy()
    evs = _events(perf)
    if len(evs) < 2:
        return perf.copy()
    dro = {}
    for i in range(len(evs) - 1):
        dn = abs(evs[i + 1].pitch - evs[i].pitch)
        if dn < min_leap:
            continue
        dn = min(dn, 9)
        dr_ms = _beats_to_ms(perf, evs[i].ioi, evs[i].beat)
        ms = 0.3 * dr_ms * (8.0 * dn / 100.0 + 0.3) * k
        ms = max(-cap_ms, min(cap_ms, ms))
        gap = _ms_to_beats(perf, ms, evs[i].beat)
        for j in evs[i].melodic:
            dro[j] = max(dro.get(j, 0.0), gap)
    return _shorten(perf, dro)


def melodic_charge(perf: Performance, k: float = 1.0, *,
                   harmony=None, all_voices: bool = False) -> Performance:
    """离和弦根音远的旋律音加重. DPC 2A"""
    if k == 0.0:
        return perf.copy()
    h = _harmony(perf, harmony)
    if h is None:
        out = perf.copy()
        out.meta.setdefault("kth_skipped", []).append(
            "melodic_charge: no harmony (pass harmony= or set meta['harmony'])")
        return out

    evs = _events(perf)
    if not evs:
        return perf.copy()

    pianyi = [0.0] * len(evs)
    dl = [0.0] * len(evs)
    for i, ev in enumerate(evs):
        root = _chord_root_at(h, ev.beat)
        if root is None:
            continue
        c = melodic_charge_of(ev.pitch, root)
        if c == 0.0:
            continue
        dl[i] = 0.2 * c * k
        pianyi[i] = (2.0 / 3.0) * c * k / 100.0 * ev.ioi

    out = _warp(perf, evs, pianyi)
    for i, ev in enumerate(evs):
        if dl[i] == 0.0:
            continue
        mubiao = ev.idx if all_voices else ev.melodic
        for j in mubiao:
            out.notes[j].vel = _gain(out.notes[j].vel, dl[i])
    return out


def harmonic_charge(perf: Performance, k: float = 1.0, *,
                    harmony=None, key=None,
                    ramp_s: float = 1.9) -> Performance:
    """离调远的和弦加重, 换和弦前渐强. GMA 2A"""
    if k == 0.0:
        return perf.copy()
    h = _harmony(perf, harmony)
    zhuyin = _key_root(perf, h, key)
    if h is None or zhuyin is None:
        out = perf.copy()
        out.meta.setdefault("kth_skipped", []).append(
            "harmonic_charge: no harmony/key")
        return out

    evs = _events(perf)
    if not evs:
        return perf.copy()

    dianhe, huanle = [], []
    shangge = None
    for ev in evs:
        try:
            ch = h.at(ev.beat)
        except Exception:                                # pragma: no cover
            dianhe.append(0.0)
            huanle.append(False)
            continue
        dianhe.append(harmonic_charge_of(ch, zhuyin))
        huanle.append(shangge is None or ch != shangge)
        shangge = ch

    pianyi = [0.0] * len(evs)
    dl = [0.0] * len(evs)
    for i, ev in enumerate(evs):
        c = max(0.0, dianhe[i])
        ms = 2.0 * math.sqrt(c) * k
        if huanle[i]:
            ms += 8.0 * math.sqrt(c) * k
        pianyi[i] = _ms_to_beats(perf, ms, ev.beat)

    # 换和弦前 ramp_s 秒内渐强
    for i, ev in enumerate(evs):
        if not huanle[i]:
            continue
        mubiao = 1.5 * dianhe[i] * k
        if mubiao == 0.0:
            continue
        jianbian_pai = ramp_s * _bpm(perf, ev.beat) / 60.0
        for j in range(i, -1, -1):
            houtui = ev.beat - evs[j].beat
            if houtui > jianbian_pai:
                break
            if j < i and huanle[j]:
                break
            dl[j] = max(dl[j], mubiao * (1.0 - houtui / max(_EPS, jianbian_pai)))
        # 之后慢慢回落
        for j in range(i + 1, len(evs)):
            if huanle[j]:
                break
            qianjin = evs[j].beat - ev.beat
            xiayige = None
            for m in range(j, len(evs)):
                if huanle[m]:
                    xiayige = evs[m].beat - ev.beat
                    break
            if xiayige is None or xiayige <= _EPS:
                break
            dl[j] = max(dl[j], mubiao * (1.0 - qianjin / xiayige))

    out = _warp(perf, evs, pianyi)
    for i, ev in enumerate(evs):
        if dl[i] == 0.0:
            continue
        for j in ev.idx:
            out.notes[j].vel = _gain(out.notes[j].vel, dl[i])
    return out


RULES = {
    "high_loud":           (high_loud, "level"),
    "melodic_charge":      (melodic_charge, "level+timing"),
    "harmonic_charge":     (harmonic_charge, "level+timing"),
    "double_duration":     (double_duration, "timing"),
    "duration_contrast":   (duration_contrast, "level+timing"),
    "faster_uphill":       (faster_uphill, "timing"),
    "leap_tone_duration":  (leap_tone_duration, "timing"),
    "inegales":            (inegales, "timing"),
    "swing":               (swing, "timing"),
    "leap_articulation":   (leap_articulation, "articulation"),
    "punctuation":         (punctuation, "timing+articulation"),
    "phrase_arch":         (phrase_arch, "level+timing"),
    "final_ritardando":    (final_ritardando, "tempo"),
}


if __name__ == "__main__":
    from ..theory import Chord, Harmony, pc_near, stack

    def _piece():
        xuanlv = [
            (0.0, 1.0, 67), (1.0, 1.0, 69), (2.0, 2.0, 71),
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
            (60.0, 4.0, 67),
        ]
        hexian = [("G", ""), ("C", ""), ("D", "7"), ("G", ""),
                  ("E", "m"), ("A", "m"), ("D", "7"), ("G", ""),
                  ("C", ""), ("G", ""), ("A", "m"), ("D", "7"),
                  ("G", ""), ("C", ""), ("D", "7"), ("G", "")]
        harmony = Harmony({i * 4.0: Chord.parse(*c)
                           for i, c in enumerate(hexian)})
        perf = Performance(tempo=TempoMap([(0.0, 88.0)]))
        perf.meta["harmony"] = harmony
        perf.meta["key"] = "G"
        for onset, dur, pitch in xuanlv:
            perf.add(Note(onset, dur, pitch, 64, Voice.MELODY, "melody"))
        for bar, (root, qual) in enumerate(hexian):
            b0 = bar * 4.0
            ch = Chord.parse(root, qual)
            low = pc_near(ch.bass, 43, lo=36, hi=52)
            perf.add(Note(b0, 2.0, low, 64, Voice.BASS, "bass"),
                     Note(b0 + 2.0, 2.0, low + 12, 64, Voice.BASS, "bass"))
            for off in (0.0, 2.0):
                for p in stack(ch, 55, 67, 2, min_gap=3):
                    perf.add(Note(b0 + off, 2.0, p, 64, Voice.INNER, "inner"))
        return perf.sorted()

    def _sig(p):
        return [(round(n.onset, 9), round(n.dur, 9), n.pitch, n.vel)
                for n in p.notes]

    perf = _piece()
    jizhun = _sig(perf)
    print("piece: %d notes, %.0f beats, %d events"
          % (len(perf.notes), perf.length, len(_events(perf))))

    for name, (fn, _kind) in sorted(RULES.items()):
        out = fn(perf, 0.0)
        assert _sig(out) == jizhun, "%s k=0 moved something" % name
        assert out is not perf and out.notes[0] is not perf.notes[0]
        assert _sig(perf) == jizhun, "%s mutated its input" % name
    print("deadpan   : all %d rules bit-identical at k=0" % len(RULES))

    for name, (fn, _kind) in sorted(RULES.items()):
        assert _sig(fn(perf, 1.0)) == _sig(fn(perf, 1.0)), \
            "%s not deterministic" % name
    print("determinism: ok")

    def _report(name, out):
        b, a = _sig(perf), _sig(out)
        dv = [y[3] - x[3] for x, y in zip(b, a)]
        do = [(y[0] - x[0]) * 60000.0 / 88.0 for x, y in zip(b, a)]
        dd = [(y[1] - x[1]) * 60000.0 / 88.0 for x, y in zip(b, a)]
        print("  %-19s vel %+5.1f..%+5.1f   onset %+7.1f..%+7.1f ms"
              "   dur %+7.1f..%+7.1f ms"
              % (name, min(dv), max(dv), min(do), max(do), min(dd), max(dd)))
        return dv, do, dd

    print("effects at k=1:")
    dv, do, dd = _report("high_loud", high_loud(perf))
    assert max(do) == min(do) == 0.0 and max(dd) == min(dd) == 0.0, \
        "high_loud moved time"
    hi = [n for n in high_loud(perf).notes if n.pitch >= 72]
    lo = [n for n in high_loud(perf).notes if n.pitch <= 55]
    assert min(n.vel for n in hi) > max(n.vel for n in lo), \
        "high_loud not monotone"

    dv, do, dd = _report("melodic_charge", melodic_charge(perf))
    assert max(dv) > 0 and min(dv) == 0, "melodic_charge should only add level"
    assert melodic_charge_of(66, 0) == 6.0 and melodic_charge_of(60, 0) == 0.0

    dv, do, dd = _report("harmonic_charge", harmonic_charge(perf))
    assert max(dv) > 0, "harmonic_charge did nothing"

    dv, do, dd = _report("duration_contrast", duration_contrast(perf))
    assert min(dd) < 0 < max(dd), "duration contrast is not a contrast"

    dv, do, dd = _report("double_duration", double_duration(perf))
    assert max(abs(x) for x in dd) > 0.0

    dv, do, dd = _report("faster_uphill", faster_uphill(perf))
    dv, do, dd = _report("leap_tone_duration", leap_tone_duration(perf))
    dv, do, dd = _report("leap_articulation", leap_articulation(perf))
    assert max(abs(x) for x in do) == 0.0, \
        "leap_articulation moved an onset"
    assert min(dd) < 0 and max(dd) <= 0, \
        "leap_articulation should only shorten"

    dv, do, dd = _report("punctuation", punctuation(perf))
    assert min(dd) < 0, "punctuation inserted no micropause"
    dv, do, dd = _report("phrase_arch", phrase_arch(perf))
    assert min(dv) < 0, "phrase arch should dip the level at phrase ends"
    dv, do, dd = _report("inegales", inegales(perf))
    dv, do, dd = _report("swing", swing(perf, ratio=2.33))

    print("final ritardando, v(x) = [1 + (w^q - 1) x]^(1/q):")
    for q in (2.0, 3.0):
        cur = ritardando_curve(k=1.0, q=q, w=0.3, n=6)
        print("  q=%.0f w=0.30 " % q
              + "  ".join("x=%.1f v=%.3f" % (x, v) for x, v in cur))
        assert abs(cur[0][1] - 1.0) < 1e-12 and abs(cur[-1][1] - 0.3) < 1e-9
        assert all(a[1] >= b[1] for a, b in zip(cur, cur[1:])), \
            "curve not monotone"
    c2 = ritardando_curve(k=1.0, q=2.0, w=0.3, n=11)
    for x, v in c2:
        assert abs(v - math.sqrt(1.0 + (0.09 - 1.0) * x)) < 1e-12
    c3 = ritardando_curve(k=1.0, q=3.0, w=0.3, n=11)
    assert all(b >= a - 1e-12 for (_, a), (_, b) in zip(c2, c3))

    r = final_ritardando(perf, 1.0, q=3.0, w=0.6, span=8.0)
    t0 = perf.tempo.bpm(50.0)
    print("  tempo over the last 8 beats: "
          + "  ".join("%.0f" % r.tempo.bpm(b)
                      for b in (54, 56, 58, 60, 62, 64)))
    assert abs(r.tempo.bpm(50.0) - t0) < 1e-9, "ritardando leaked backwards"
    assert abs(r.tempo.bpm(64.0) - t0 * 0.6) < 1e-6, "final tempo wrong"

    p = perf
    for fn in (high_loud, melodic_charge, harmonic_charge, double_duration,
               duration_contrast, leap_tone_duration, leap_articulation,
               punctuation, phrase_arch):
        p = fn(p, 1.0)
    p = final_ritardando(p, 1.0)
    assert len(p.notes) == len(perf.notes)
    assert all(n.dur > 0 and n.onset >= 0 for n in p.notes)
    assert [n.pitch for n in p.notes] == [n.pitch for n in perf.notes]
    print("combined  : %d notes, length %.2f beats (nominal %.2f)"
          % (len(p.notes), p.length, perf.length))
    print("kth ok")
