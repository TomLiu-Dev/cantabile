# -*- coding: utf-8 -*-
"""从没标注的音里找出旋律, 再给它配伴奏: 和弦, 织体, 音区, 左右手。

旋律用 Viterbi 在起音组上解码, 不是简单 skyline (有 "hold" 状态, 下面声部重复
而上面长音没动的情况也能对)。伴奏上限放在旋律低音区下面小三度, 避开临界带。
参考: Uitdenbogerd & Zobel skyline; Nakamura & Sagayama ICMC 2015
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from cantabile.model import Note, Performance, TempoMap, Voice
from cantabile.theory import Chord, Harmony, QUALITIES, name
from cantabile.arrange import styles as _styles
from cantabile.arrange.textures import (BlockChords, BrokenChord,
                                        QuaverAlternation, Sustained,
                                        WalkingBass)
from cantabile.arrange.voicing import melody_harmony
from cantabile.arrange import playability as _play

__all__ = ["Allocation", "allocate", "choose_texture", "budget", "realise",
           "recover_melody", "register_plan", "infer_harmony", "cross_check",
           "MELODY_WEIGHTS", "BELOW_TUNE_SHARE", "REFERENCE_BANDS",
           "REFERENCE_TOTAL", "REFERENCE_BPM"]

EPS = 1e-6

# 参考录音: 总密度 (音/秒), 速度, 各音区密度
REFERENCE_TOTAL = (7.5, 10.2)

REFERENCE_BPM = (89.0, 108.0)

REFERENCE_BANDS = {
    "bass":  (3.45, 4.30),
    "tenor": (2.85, 3.23),
    "alto":  (0.80, 2.13),
    "top":   (0.40, 0.55),
}

# G3 = 55, E4 = 64, C5 = 72
BAND_EDGES = (("bass", 0, 55), ("tenor", 55, 64), ("alto", 64, 72),
              ("top", 72, 128))

# 旋律大约 1.5 音/秒, 总共 7.5-10.2, 所以 84% 左右的音在旋律下面
BELOW_TUNE_SHARE = 0.84

CEILING_GAP = 3

MELODY_LOW_Q = 0.10

CEILING_BOUNDS = (50, 67)


def _lazy(module, *attrs):
    """cantabile.analyze.<module> 里第一个能用的函数, 没有就 None。"""
    try:
        mod = __import__("cantabile.analyze." + module, fromlist=["*"])
    except Exception:
        return None
    for a in attrs:
        fn = getattr(mod, a, None)
        if callable(fn):
            return fn
    return None


def _as_note(n):
    if isinstance(n, Note):
        return replace(n)
    if hasattr(n, "pitch"):
        return Note(float(n.onset), float(n.dur), int(n.pitch),
                    int(getattr(n, "vel", 64)),
                    getattr(n, "voice", Voice.INNER), getattr(n, "tag", ""))
    o, d, p, v = n
    return Note(float(o), float(d), int(p), int(v), Voice.INNER, "input")


def _normalise(notes):
    jieguo = [_as_note(n) for n in notes]
    jieguo.sort(key=lambda n: (n.onset, n.pitch))
    return jieguo


def _onset_groups(notes, tol=0.05):
    zu = []
    for n in sorted(notes, key=lambda n: (n.onset, n.pitch)):
        if zu and n.onset - zu[-1][0] <= tol:
            zu[-1][1].append(n)
        else:
            zu.append((n.onset, [n]))
    return zu


def _quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    i = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[i]


def _span_of(notes):
    if not notes:
        return 0.0
    return max(n.end for n in notes) - min(n.onset for n in notes)


def _band_of(pitch):
    for label, lo, hi in BAND_EDGES:
        if lo <= pitch < hi:
            return label
    return "top"


def _mid(r):
    return (r[0] + r[1]) / 2.0 if isinstance(r, (tuple, list)) else float(r)


# 解码器的代价, 用坐标上升在三首核对过旋律的曲子上拟合的, 平均 F1 0.995
MELODY_WEIGHTS = {
    "onset": 0.5, "height": 0.7, "floor": 1.5, "vel": 0.6, "thin": 0.0,
    "hold": 0.8, "sustain": 1.0, "block": 3.0, "fullblock": 0.3, "below": 1.2,
    "leap": 0.15, "down": 0.0, "lone": 0.5, "maxfall": 2,
    "look": 9, "minchord": 3, "floor_q": 0.02, "floor_slack": 2,
    "prior": 0.0,
}


def _melody_floor(groups, w):
    """旋律音区的下沿: 满和弦顶音的低分位数, 再减一点。"""
    dingyin = [max(n.pitch for n in g) for _, g in groups
               if len(g) >= w["minchord"]]
    if not dingyin:
        dingyin = [max(n.pitch for n in g) for _, g in groups]
    q = _quantile(dingyin, w["floor_q"])
    return (q if q is not None else 0) - w["floor_slack"]


def recover_melody(notes, *, weights: dict = None) -> list:
    """从没标注的音里找出旋律, 返回的就是输入里的 Note 对象。"""
    w = dict(MELODY_WEIGHTS)
    if weights:
        w.update(weights)
    ns = list(notes)
    groups = _onset_groups(ns)
    if not groups:
        return []

    xiaxian = _melody_floor(groups, w)
    xianyan = _analysis_prior(ns)

    def emit(c, grp):
        ding = max(n.pitch for n in grp)
        zuixiang = max(n.vel for n in grp)
        cost = w["onset"] + w["height"] * (ding - c.pitch)
        cost += w["floor"] * max(0, xiaxian - c.pitch)
        cost += w["vel"] * max(0, zuixiang - c.vel) / 16.0
        if len(grp) < w["minchord"]:
            cost += w["thin"]
        if xianyan is not None and (round(c.onset, 4), c.pitch) in xianyan:
            cost -= w["prior"]
        return cost

    def hold(m, grp, beat):
        ding = max(n.pitch for n in grp)
        cost = w["hold"]
        if len(grp) >= w["minchord"]:
            cost += w["fullblock"]
        if ding >= m.pitch:
            cost += w["block"]
        cost += w["below"] * max(0, ding - m.pitch)
        if m.end > beat + EPS:
            cost -= w["sustain"]
        return cost

    def step(m, c, grp):
        gap = abs(c.pitch - m.pitch)
        cost = w["leap"] * max(0, gap - 5)
        if c.pitch < m.pitch:
            cost += w["down"] * max(0, gap - 2)
            if len(grp) == 1:
                cost += w["lone"] * max(0, gap - w["maxfall"])
        return cost

    _, g0 = groups[0]
    ding0 = max(n.pitch for n in g0)
    zhuangtai = {}
    for c in g0:
        if c.pitch >= ding0 - w["look"]:
            zhuangtai[id(c)] = (emit(c, g0), c, [c])

    for beat, grp in groups[1:]:
        ding = max(n.pitch for n in grp)
        houxuan = [c for c in grp if c.pitch >= ding - w["look"]]
        xia = {}
        for _, (sc, m, lujing) in zhuangtai.items():
            hc = sc + hold(m, grp, beat)
            cur = xia.get(id(m))
            if cur is None or hc < cur[0]:
                xia[id(m)] = (hc, m, lujing)
            for c in houxuan:
                if c is m:
                    continue
                cc = sc + emit(c, grp) + step(m, c, grp)
                cur = xia.get(id(c))
                if cur is None or cc < cur[0]:
                    xia[id(c)] = (cc, c, lujing + [c])
        zhuangtai = xia
        if not zhuangtai:
            zhuangtai = {id(c): (0.0, c, [c]) for c in houxuan}

    return min(zhuangtai.values(), key=lambda s: s[0])[2]


def cross_check(melody, notes) -> dict:
    """跟 analyze.melody 的结果对一下, 只是记录, 不影响结果。"""
    xianyan = _analysis_prior(notes)
    if xianyan is None:
        return {"source": "none (analyze.melody not available)"}
    wode = {(round(n.onset, 4), n.pitch) for n in melody}
    yizhi = wode & xianyan
    return {
        "source": "analyze.melody",
        "agreed": len(yizhi),
        "only_here": sorted(wode - xianyan)[:16],
        "only_there": sorted(xianyan - wode)[:16],
        "agreement": round(len(yizhi) / max(1, len(wode | xianyan)), 3),
    }


def _analysis_prior(notes):
    fn = _lazy("melody", "melody_line", "extract", "skyline", "line")
    if fn is None:
        return None
    try:
        got = fn(notes)
    except Exception:
        return None
    jieguo = set()
    for item in got or ():
        if isinstance(item, Note) or hasattr(item, "pitch"):
            jieguo.add((round(float(item.onset), 4), int(item.pitch)))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            jieguo.add((round(float(item[0]), 4), int(item[1])))
    return jieguo or None


def register_plan(melody, *, gap: int = CEILING_GAP) -> tuple:
    """返回 (ceiling, expose_below)。旋律没音就用默认 (59, 62)。"""
    yingao = [n.pitch if hasattr(n, "pitch") else int(n[1]) for n in melody]
    if not yingao:
        return 59, 62
    di = _quantile(yingao, MELODY_LOW_Q)
    shangxian = int(di) - int(gap)
    lo, hi = CEILING_BOUNDS
    shangxian = max(lo, min(hi, shangxian))
    return shangxian, shangxian + int(gap)


_PREF = {"": 1.00, "m": 1.00, "7": 0.97, "m7": 0.97, "maj7": 0.95, "6": 0.93,
         "m6": 0.90, "sus4": 0.90, "sus2": 0.88, "dim": 0.90, "dim7": 0.90,
         "m7b5": 0.88, "aug": 0.82, "add9": 0.88, "madd9": 0.86, "7sus4": 0.88,
         "9": 0.86, "m9": 0.86, "maj9": 0.84, "6/9": 0.82}


def _best_chord(quanzhong, bass_pc):
    zong = sum(quanzhong.values())
    if zong <= 0:
        return None
    best, best_score = None, float("-inf")
    for root in range(12):
        for quality in QUALITIES:
            tones = {(root + i) % 12 for i in QUALITIES[quality]}
            fugai = sum(v for pc, v in quanzhong.items() if pc in tones)
            queshao = sum(1 for t in tones if quanzhong.get(t, 0.0) <= 0.0)
            score = (fugai / zong) * _PREF.get(quality, 0.85)
            score -= 0.40 * ((zong - fugai) / zong)
            score -= 0.03 * queshao
            score -= 0.030 * len(tones)
            if bass_pc is not None:
                score += 0.09 if bass_pc == root else (
                    0.03 if bass_pc in tones else -0.10)
            if score > best_score:
                best_score, best = score, (root, quality)
    root, quality = best
    return Chord(root, quality, bass_pc) if (bass_pc is not None
                                             and bass_pc != root) \
        else Chord(root, quality)


def infer_harmony(accompaniment, melody=(), *, bpb: float = 4.0,
                  resolution: float = None) -> Harmony:
    """从伴奏里推和弦, 默认半小节一个。旋律音不算进去。"""
    res = float(bpb) / 2.0 if resolution is None else float(resolution)
    notes = list(accompaniment)
    if not notes:
        return Harmony({0.0: Chord(0, "")})
    end = max(n.end for n in notes)
    mel = {(round(n.onset, 4), n.pitch) for n in melody}

    slots, prev = {}, None
    n_slots = max(1, int(round(end / res + 0.5)))
    for i in range(n_slots):
        t0, t1 = i * res, (i + 1) * res
        quanzhong, bass_pc, diyin = {}, None, 128
        for n in notes:
            if (round(n.onset, 4), n.pitch) in mel:
                continue
            ov = min(n.end, t1) - max(n.onset, t0)
            if ov <= 1e-9:
                continue
            pc = n.pitch % 12
            quanzhong[pc] = quanzhong.get(pc, 0.0) + ov * (n.vel / 100.0)
            if n.pitch < diyin:
                diyin, bass_pc = n.pitch, pc
        ch = _best_chord(quanzhong, bass_pc) or prev or Chord(0, "")
        if prev is None or ch != prev:
            slots[float(t0)] = ch
        prev = ch
    return Harmony(slots or {0.0: Chord(0, "")})


def budget(melody, harmony, target_density, *, bpb: float = 4,
           bpm: float = None) -> dict:
    """扣掉旋律以后伴奏还能放多少音。target_density 可以是数或 (lo, hi)。"""
    if isinstance(target_density, (tuple, list)) and len(target_density) == 2:
        band = (float(min(target_density)), float(max(target_density)))
        mubiao = _mid(band)
    else:
        band = (float(target_density), float(target_density))
        mubiao = float(target_density)
    bpm = float(bpm) if bpm else _mid(REFERENCE_BPM)
    meimiao = bpm / 60.0

    mel = list(melody)
    beats = _span_of(mel) or (len(mel) or 1.0)
    mel_per_beat = len(mel) / max(beats, EPS)
    mel_nps = mel_per_beat * meimiao

    acc_nps = mubiao - mel_nps
    acc_per_beat = acc_nps / max(meimiao, EPS)

    # top 是旋律的, 伴奏分到其余三个音区
    fene = {k: _mid(REFERENCE_BANDS[k]) for k in ("bass", "tenor", "alto")}
    tot = sum(fene.values()) or 1.0
    bands = {k: round(max(0.0, acc_nps) * v / tot, 3) for k, v in fene.items()}
    bands["top"] = round(mel_nps, 3)

    below = 0.0 if mubiao <= 0 else max(0.0, acc_nps) / mubiao
    jieguo = {
        "bpm": round(bpm, 2),
        "beats": round(beats, 2),
        "target_nps": round(mubiao, 3),
        "target_band": band,
        "melody_notes": len(mel),
        "melody_per_beat": round(mel_per_beat, 3),
        "melody_nps": round(mel_nps, 3),
        "accomp_nps": round(acc_nps, 3),
        "accomp_per_beat": round(acc_per_beat, 3),
        "bands_nps": bands,
        "bands_per_beat": {k: round(v / max(meimiao, EPS), 3)
                           for k, v in bands.items()},
        "below_tune_share": round(below, 3),
        "below_tune_target": BELOW_TUNE_SHARE,
        "harmonic_rhythm": round(_harmonic_rhythm(harmony, beats), 3),
        "ok": acc_per_beat > 0.0,
    }
    if acc_per_beat <= 0.0:
        jieguo["note"] = ("the tune alone is at or over the target density; "
                          "no accompaniment fits under it")
    elif below < BELOW_TUNE_SHARE - 0.06:
        jieguo["note"] = ("only %.0f%% of notes would sit below the tune against "
                          "a measured %.0f%%: the melody is unusually busy for "
                          "this target" % (100 * below, 100 * BELOW_TUNE_SHARE))
    else:
        jieguo["note"] = "within the measured split"
    return jieguo


def _harmonic_rhythm(harmony, beats):
    """平均几拍换一个和弦。"""
    if harmony is None:
        return float(beats) or 4.0
    try:
        spans = list(harmony.spans())
    except Exception:
        return float(beats) or 4.0
    changdu = []
    for a, b, _ in spans:
        b = min(b, beats) if beats else b
        if b > a and b != float("inf"):
            changdu.append(b - a)
    return sum(changdu) / len(changdu) if changdu else (float(beats) or 4.0)


def _candidates():
    return [
        ("accompaniment",    QuaverAlternation(chord_notes=2)),
        ("four-part", BlockChords(subdivision=1.0, chord_notes=1)),
        ("ballad",  BrokenChord(notes_per_beat=2, span_octaves=2)),
        ("pad",     Sustained()),
        ("walking", WalkingBass(step=1.0)),
    ]


def _measure_texture(jili, style, harmony, ceiling, expose, melody,
                     beats, bpb):
    """真的渲染一遍, 数每拍几个伴奏音。"""
    if not beats:
        return 0.0
    st = replace(style, texture=jili, lh_ceiling=ceiling,
                 expose_melody_below=expose)
    notes = _render_all(st, harmony, {"ceiling": ceiling}, melody, beats, bpb)
    return len(notes) / beats


def _band_rates(perf_notes, beats, bpm):
    miao = beats * 60.0 / max(bpm, EPS)
    jishu = {}
    for n in perf_notes:
        b = _band_of(n.pitch)
        jishu[b] = jishu.get(b, 0) + 1
    return {k: jishu.get(k, 0) / max(miao, EPS) for k, _, _ in BAND_EDGES}


def _band_error(rates):
    err = 0.0
    for band, rng in REFERENCE_BANDS.items():
        v = rates.get(band, 0.0)
        lo, hi = rng
        if v < lo:
            err += (lo - v) / lo
        elif v > hi:
            err += (v - hi) / hi
    return err


def _render_all(style, harmony, ctx_extra, melody, beats, bpb, dyn=56):
    """一个风格在整个和弦表上出的所有伴奏音, 包括旋律下面的右手和声。"""
    jieguo = []
    for start, end, hexian in harmony.spans():
        a = max(0.0, start)
        b = min(end, beats) if beats else end
        if b <= a or b == float("inf"):
            continue
        extra = dict(ctx_extra)
        extra["bar_origin"] = a - (a % bpb)
        try:
            jieguo.extend(style.render(hexian, a, b - a, dyn, **extra))
        except Exception:
            return jieguo
    for n in melody:
        if n.dur < 0.5:
            continue
        xiamian = melody_harmony(n.pitch, harmony.at(n.onset), style.rh_harmony,
                                 expose_below=style.expose_melody_below)
        for j, pt in enumerate([q for q in xiamian if q is not None]):
            jieguo.append(Note(n.onset, n.dur * 0.94, pt, max(1, dyn - 4 - 3 * j),
                               Voice.INNER, "allocate.melody.harmony"))
    return jieguo


# 左手低音的几个候选范围, 第一个是默认
_LH_WINDOWS = ((38, 50), (36, 48), (35, 47), (33, 45), (40, 52))


def _choose_bass_window(style, harmony, melody, beats, bpb, bpm):
    """每个范围都渲染一遍, 挑各音区密度最接近参考的。返回 (target, lo, hi, report)。"""
    best, shiguo = None, []
    for lo, hi in _LH_WINDOWS:
        if hi > style.lh_ceiling:
            continue
        target = (lo + hi) // 2
        extra = {"lo": lo, "hi": hi, "target": target,
                 "ceiling": style.lh_ceiling}
        notes = _render_all(style, harmony, extra, melody, beats, bpb)
        rates = _band_rates(notes, beats, bpm)
        err = _band_error(rates)
        shiguo.append({"window": (lo, hi), "target": target,
                       "band_error": round(err, 3),
                       "rates": {k: round(v, 2) for k, v in rates.items()}})
        if best is None or err < best[0]:
            best = (err, target, lo, hi, rates)
    if best is None:
        return (style.bass_target, 38, 50,
                {"note": "the ceiling left no usable window; style default"})
    err, target, lo, hi, rates = best
    return target, lo, hi, {
        "chosen": (lo, hi), "target": target, "band_error": round(err, 3),
        "rates": {k: round(v, 2) for k, v in rates.items()},
        "reference": REFERENCE_BANDS,
        "tried": shiguo,
        "why": ("the window whose realised register split comes closest to the "
                "measured recordings, not the one with the right total"),
    }


def choose_texture(melody, harmony, *, bpb: float = 4, tempo=None) -> tuple:
    """每个候选织体都渲染打分, 返回 (texture, reasons)。tempo 是 BPM 或 TempoMap。"""
    mel = [n if isinstance(n, Note) else _as_note(n) for n in melody]
    bpm = _bpm_of(tempo)
    beats = _span_of(mel)
    if not beats:
        try:
            beats = max(b for b, _, _ in harmony.spans()) + bpb
        except Exception:
            beats = float(bpb)
    mel_per_beat = len(mel) / max(beats, EPS)
    shangxian, expose = register_plan(mel)
    hr = _harmonic_rhythm(harmony, beats)
    # 从常见左手低音 G#2 到上限的空间
    kongjian = shangxian - 44

    feat = {
        "melody_notes": len(mel),
        "beats": round(beats, 2),
        "melody_per_beat": round(mel_per_beat, 3),
        "melody_low": min((n.pitch for n in mel), default=None),
        "melody_high": max((n.pitch for n in mel), default=None),
        "ceiling": shangxian,
        "expose_below": expose,
        "room_semitones": kongjian,
        "harmonic_rhythm_beats": round(hr, 2),
        "bpm": round(bpm, 1),
    }

    defen = []
    for sname, tex in _candidates():
        st = _styles.get(sname)
        per_beat = _measure_texture(tex, st, harmony, shangxian, expose,
                                    mel, beats, bpb)
        nps = (per_beat + mel_per_beat) * bpm / 60.0
        lo, hi = st.notes_per_sec
        if nps < lo:
            midu = (lo - nps) / max(lo, EPS)
        elif nps > hi:
            midu = (nps - hi) / max(hi, EPS)
        else:
            midu = 0.0

        terms = {"density": round(3.0 * midu, 3)}
        liyou = []

        # 独奏织体薄一点, 给 10 个百分点余量
        share = per_beat / (per_beat + mel_per_beat) if per_beat + mel_per_beat \
            else 0.0
        short = max(0.0, (BELOW_TUNE_SHARE - 0.10) - share)
        terms["below_tune"] = round(2.5 * short, 3)
        if short:
            liyou.append("only %.0f%% of its notes would sit below the tune "
                         "(measured %.0f%%)" % (100 * share,
                                                100 * BELOW_TUNE_SHARE))

        need = _ROOM_NEEDED[sname]
        if kongjian < need:
            terms["room"] = round(0.45 * (need - kongjian), 3)
            liyou.append("needs %d semitones under the tune, has %d"
                         % (need, kongjian))
        else:
            terms["room"] = 0.0

        mang = max(0.0, mel_per_beat - 0.80)
        xishu = max(0.0, 0.55 - mel_per_beat)
        terms["melody"] = round(_BUSY_PENALTY[sname] * mang
                                + _SPARSE_PENALTY[sname] * xishu, 3)
        if mang and _BUSY_PENALTY[sname] > 0:
            liyou.append("the tune is already busy (%.2f notes/beat)"
                         % mel_per_beat)
        if xishu and _SPARSE_PENALTY[sname] > 0:
            liyou.append("the tune is sparse (%.2f notes/beat) and something "
                         "must carry the motion" % mel_per_beat)

        lo_h, hi_h = _HR_WINDOW[sname]
        if hr < lo_h:
            terms["harmony"] = round(0.8 * (lo_h - hr), 3)
            liyou.append("harmony moves every %.1f beats, faster than this "
                         "texture states it" % hr)
        elif hr > hi_h:
            terms["harmony"] = round(0.5 * (hr - hi_h), 3)
            liyou.append("harmony sits still for %.1f beats; this texture has "
                         "nothing to do" % hr)
        else:
            terms["harmony"] = 0.0

        lo_t, hi_t = _TEMPO_WINDOW[sname]
        if bpm < lo_t:
            terms["tempo"] = round(0.02 * (lo_t - bpm), 3)
            liyou.append("%.0f bpm is slow for this texture (%.0f-%.0f)"
                         % (bpm, lo_t, hi_t))
        elif bpm > hi_t:
            terms["tempo"] = round(0.02 * (bpm - hi_t), 3)
            liyou.append("%.0f bpm is fast for this texture (%.0f-%.0f)"
                         % (bpm, lo_t, hi_t))
        else:
            terms["tempo"] = 0.0

        zongfen = round(sum(terms.values()), 3)
        defen.append({
            "style": sname, "texture": tex, "score": zongfen, "terms": terms,
            "accomp_per_beat": round(per_beat, 3),
            "below_tune_share": round(share, 3),
            "total_nps": round(nps, 2),
            "style_band": st.notes_per_sec,
            "why": "; ".join(liyou) or "fits the material",
        })

    defen.sort(key=lambda c: c["score"])
    win = defen[0]
    dier = defen[1] if len(defen) > 1 else None

    reasons = {
        "features": feat,
        "chosen": win["style"],
        "texture": type(win["texture"]).__name__,
        "score": win["score"],
        "margin": round(dier["score"] - win["score"], 3) if dier else None,
        "runner_up": dier["style"] if dier else None,
        "candidates": [{k: v for k, v in c.items() if k != "texture"}
                       for c in defen],
        "verdict": _verdict(win, feat),
    }
    return win["texture"], reasons


# 每种织体: 低音 44 到上限要几个半音
_ROOM_NEEDED = {"accompaniment": 12, "four-part": 10, "ballad": 19, "pad": 7,
                "walking": 7}

_BUSY_PENALTY = {"accompaniment": 0.0, "four-part": 0.2, "ballad": 2.2, "pad": 0.0,
                 "walking": 1.6}

_SPARSE_PENALTY = {"accompaniment": 0.8, "four-part": 1.6, "ballad": 0.0, "pad": 2.2,
                   "walking": 0.4}

# 几拍一个和弦合适
_HR_WINDOW = {"accompaniment": (1.5, 4.0), "four-part": (0.5, 1.5), "ballad": (2.0, 8.0),
              "pad": (2.0, 16.0), "walking": (2.0, 8.0)}

_TEMPO_WINDOW = {"accompaniment": (80.0, 120.0), "four-part": (60.0, 108.0),
                 "ballad": (50.0, 92.0), "pad": (40.0, 80.0),
                 "walking": (100.0, 180.0)}


def _verdict(win, feat):
    return ("%s: %.2f melody notes/beat, %.1f beats per chord, %d semitones "
            "of room under the tune (ceiling %s), %.0f bpm -> %.2f notes/sec "
            "total, inside the measured band %s"
            % (win["style"], feat["melody_per_beat"],
               feat["harmonic_rhythm_beats"], feat["room_semitones"],
               name(feat["ceiling"]) if feat["ceiling"] else "-",
               feat["bpm"], win["total_nps"], win["style_band"]))


def _bpm_of(tempo):
    if tempo is None:
        return _mid(REFERENCE_BPM)
    if isinstance(tempo, TempoMap):
        return float(tempo.bpm(0.0))
    try:
        return float(tempo)
    except Exception:
        return _mid(REFERENCE_BPM)


@dataclass
class Allocation:
    """allocate 的结果。rationale 里记着每一步为什么这么选。"""
    melody: list = field(default_factory=list)
    accompaniment: list = field(default_factory=list)
    hands: dict = field(default_factory=dict)
    texture: object = None
    style: object = None
    rationale: dict = field(default_factory=dict)

    @property
    def notes(self) -> list:
        return sorted(self.melody + self.accompaniment,
                      key=lambda n: (n.onset, n.pitch))

    def hand_of(self, note) -> str:
        return self.hands.get(id(note), "?")

    def explain(self) -> str:
        hang = []
        r = self.rationale
        hang.append("melody       %d notes, %s-%s"
                    % (len(self.melody),
                       name(min(n.pitch for n in self.melody)) if self.melody else "-",
                       name(max(n.pitch for n in self.melody)) if self.melody else "-"))
        hang.append("accompaniment%4d notes" % len(self.accompaniment))
        hang.append("style        %s" % (self.style.name if self.style else "-"))
        hang.append("texture      %s" % type(self.texture).__name__)
        reg = r.get("register", {})
        hang.append("ceiling      %s  (tune's 10th percentile %s, minus %d)"
                    % (name(reg.get("ceiling", 0)), name(reg.get("melody_low_q", 0)),
                       reg.get("gap", CEILING_GAP)))
        hang.append("expose below %s  (melody notes at or under this get no "
                    "harmony packed beneath them)" % name(reg.get("expose_below", 0)))
        tex = r.get("texture", {})
        if tex.get("verdict"):
            hang.append("chose        %s" % tex["verdict"])
        for c in tex.get("candidates", []):
            hang.append("   %-8s score %6.2f  %s" % (c["style"], c["score"],
                                                     c["why"]))
        b = r.get("budget", {})
        if b:
            lhw = r.get("register", {}).get("left_hand_window", {})
            if lhw.get("chosen"):
                hang.append("left hand    window %s-%s, chosen because its "
                            "register split is closest to the recordings "
                            "(error %.2f)"
                            % (name(lhw["chosen"][0]), name(lhw["chosen"][1]),
                               lhw["band_error"]))
            hang.append("budget       %.2f notes/beat of accompaniment "
                        "(%.2f notes/sec); %.0f%% of notes below the tune "
                        "(measured %.0f%%)"
                        % (b["accomp_per_beat"], b["accomp_nps"],
                           100 * b["below_tune_share"],
                           100 * b["below_tune_target"]))
            hang.append("             %s" % b.get("note", ""))
        h = r.get("hands", {})
        if h:
            hang.append("hands        L %d / R %d, split %s"
                        % (h.get("left", 0), h.get("right", 0),
                           h.get("split_range", "-")))
        return "\n".join(hang)


def allocate(notes, *, harmony=None, style=None, bpb=4, tempo=None,
             span=None) -> Allocation:
    """分出旋律和伴奏, 定好编配。notes 可以是 Note 或 (onset, dur, pitch, vel)。"""
    hs = span or _play.HandSpan()
    ns = _normalise(notes)
    if not ns:
        return Allocation(rationale={"error": "no notes"})

    mel = recover_melody(ns)
    mel_ids = {id(n) for n in mel}
    acc = [n for n in ns if id(n) not in mel_ids]
    for n in mel:
        n.voice = Voice.MELODY
        n.tag = n.tag or "allocated.melody"
    for n in acc:
        if n.voice is Voice.MELODY:
            n.voice = Voice.INNER
        n.tag = n.tag or "allocated.accompaniment"

    shangxian, expose = register_plan(mel)
    mel_yingao = [n.pitch for n in mel]
    yinqu = {
        "melody_low": min(mel_yingao) if mel_yingao else None,
        "melody_low_q": _quantile(mel_yingao, MELODY_LOW_Q),
        "melody_high": max(mel_yingao) if mel_yingao else None,
        "quantile": MELODY_LOW_Q,
        "gap": CEILING_GAP,
        "ceiling": shangxian,
        "expose_below": expose,
        "why": ("the accompaniment stops a minor third below the register the "
                "tune habitually uses; inside that a critical band wide "
                "(2-4 semitones here) the tune is masked whatever its volume"),
    }

    beats = _span_of(ns)

    if harmony is None:
        hesheng = infer_harmony(acc, mel, bpb=bpb)
        laiyuan = "inferred from the accompaniment (melody excluded)"
    elif isinstance(harmony, Harmony):
        hesheng, laiyuan = harmony, "supplied"
    else:
        hesheng, laiyuan = Harmony(harmony), "supplied grid"

    jili, treasons = choose_texture(mel, hesheng, bpb=bpb, tempo=tempo)

    base = _styles.get(style) if style is not None \
        else _styles.get(treasons["chosen"])
    if style is not None:
        jili = base.texture
        treasons["chosen"] = base.name
        treasons["verdict"] = "style pinned by the caller: %s" % base.name
    st = replace(base, texture=jili, lh_ceiling=shangxian,
                 expose_melody_below=expose, beats_per_bar=float(bpb))

    lh_target, lh_lo, lh_hi, lh_report = _choose_bass_window(
        st, hesheng, mel, beats, bpb, _bpm_of(tempo))
    st = replace(st, bass_target=lh_target)
    yinqu["left_hand_window"] = lh_report

    yusuan = budget(mel, hesheng, st.notes_per_sec, bpb=bpb, bpm=_bpm_of(tempo))
    yusuan["left_hand_window"] = (lh_lo, lh_hi)

    hands = _play.assign_hands(ns, span=hs)
    fenjie = [_play.split_point(ns, b, span=hs)
              for b in _sample_beats(ns, beats)]
    hand_info = {
        "left": sum(1 for v in hands.values() if v == "L"),
        "right": sum(1 for v in hands.values() if v == "R"),
        "split_range": "%s-%s" % (name(min(fenjie)), name(max(fenjie)))
        if fenjie else "-",
        "split_moves": sum(1 for a, b in zip(fenjie, fenjie[1:]) if a != b),
        "melody_in_right": sum(1 for n in mel if hands.get(id(n)) == "R"),
    }

    duanluo = _sections(mel, bpb)

    return Allocation(
        melody=mel, accompaniment=acc, hands=hands, texture=jili, style=st,
        rationale={
            "input_notes": len(ns),
            "melody": {
                "notes": len(mel),
                "method": "viterbi over onset groups (hold state + full-chord "
                          "and sustain evidence); see MELODY_WEIGHTS",
                "weights": dict(MELODY_WEIGHTS),
                "floor": _melody_floor(_onset_groups(ns), MELODY_WEIGHTS),
                "cross_check": cross_check(mel, ns),
            },
            "register": yinqu,
            "harmony": {"source": laiyuan, "chords": len(hesheng),
                        "beats_per_chord": round(
                            _harmonic_rhythm(hesheng, beats), 2),
                        "grid": hesheng},
            "texture": treasons,
            "budget": yusuan,
            "hands": hand_info,
            "structure": duanluo,
            "constants": {"below_tune_share": BELOW_TUNE_SHARE,
                          "reference_total_nps": REFERENCE_TOTAL,
                          "reference_bands_nps": REFERENCE_BANDS},
        })


def _sample_beats(notes, beats, step=1.0):
    if not notes:
        return []
    t0 = min(n.onset for n in notes)
    jieguo, t = [], t0
    while t <= t0 + beats + EPS:
        jieguo.append(t)
        t += step
    return jieguo


def _sections(notes, bpb):
    """analyze.structure 有的话记一下乐句边界, 只是记录。"""
    fn = _lazy("structure", "phrases", "sections", "segment")
    if fn is None or not notes:
        return {"source": "none (analyze.structure not available)",
                "policy": "one texture for the whole allocation"}
    try:
        got = fn(notes)
    except Exception:
        return {"source": "analyze.structure raised; ignored",
                "policy": "one texture for the whole allocation"}
    return {"source": "analyze.structure.%s" % fn.__name__,
            "phrases": len(list(got or ())),
            "boundaries": list(got or ())[:16],
            "policy": "one texture for the whole allocation"}


def realise(alloc: Allocation, *, dyn: int = 56) -> Performance:
    """按选好的风格重新生成伴奏 (不用输入的伴奏), 力度留给演奏层。"""
    st = alloc.style or _styles.get("accompaniment")
    hesheng = _harmony_of(alloc)
    meixiaojie = float(getattr(st, "beats_per_bar", 4.0))
    bpm = alloc.rationale.get("budget", {}).get("bpm") or _mid(REFERENCE_BPM)
    lh = alloc.rationale.get("budget", {}).get("left_hand_window", (38, 50))

    perf = Performance(tempo=TempoMap([(0.0, float(bpm))]))
    perf.meta.update(style=st.name, texture=type(st.texture).__name__,
                     ceiling=st.lh_ceiling,
                     expose_melody_below=st.expose_melody_below,
                     allocation=alloc.rationale)

    beats = _span_of(alloc.melody + alloc.accompaniment)

    spans = list(hesheng.spans())
    for i, (start, end, hexian) in enumerate(spans):
        a = max(0.0, start)
        b = min(end, beats) if beats else end
        if b <= a or b == float("inf"):
            continue
        xiayige = spans[i + 1][2] if i + 1 < len(spans) else None
        perf.add(*st.render(hexian, a, b - a, dyn, bar_origin=a - (a % meixiaojie),
                            next_chord=xiayige, lo=lh[0], hi=lh[1],
                            target=st.bass_target))
        perf.pedal.append((a, 1))

    for n in sorted(alloc.melody, key=lambda x: x.onset):
        perf.add(Note(n.onset, n.dur * 1.02, n.pitch, dyn, Voice.MELODY,
                      "allocate.melody"))
        if n.dur < 0.5:
            continue
        xiamian = melody_harmony(n.pitch, hesheng.at(n.onset), st.rh_harmony,
                                 expose_below=st.expose_melody_below)
        for j, p in enumerate([p for p in xiamian if p is not None]):
            perf.add(Note(n.onset, n.dur * 0.94, p, max(1, dyn - 4 - 3 * j),
                          Voice.INNER, "allocate.melody.harmony"))

    perf.sorted()
    _enforce_below_melody(perf)
    perf.meta["hands"] = _play.assign_hands(perf.notes)
    return perf.sorted()


def _harmony_of(alloc):
    h = alloc.rationale.get("harmony", {})
    if isinstance(h, dict) and isinstance(h.get("grid"), Harmony):
        return h["grid"]
    return infer_harmony(alloc.accompaniment, alloc.melody)


def _enforce_below_melody(perf):
    """伴奏音高过同时响的旋律音就降八度, 低于 A0 就扔掉。"""
    mel = sorted([n for n in perf.notes if n.voice is Voice.MELODY],
                 key=lambda n: n.onset)
    if not mel:
        return perf
    folded = dropped = 0
    liu = []
    for n in perf.notes:
        if n.voice is Voice.MELODY:
            liu.append(n)
            continue
        chongtu = [m for m in mel
                   if m.onset < n.end - EPS and n.onset < m.end - EPS
                   and m.pitch <= n.pitch]
        if not chongtu:
            liu.append(n)
            continue
        xianzhi = min(m.pitch for m in chongtu) - 1
        p = n.pitch
        while p > xianzhi:
            p -= 12
        if p >= 21:
            n.pitch = p
            folded += 1
            liu.append(n)
        else:
            dropped += 1
    perf.notes = liu
    perf.meta["melody_protect_folded"] = folded
    perf.meta["melody_protect_dropped"] = dropped
    return perf


def _above_melody(perf):
    mel = [n for n in perf.notes if n.voice is Voice.MELODY]
    bad = []
    for n in perf.notes:
        if n.voice is Voice.MELODY:
            continue
        for m in mel:
            if m.onset < n.end - EPS and n.onset < m.end - EPS \
                    and n.pitch >= m.pitch:
                bad.append((n, m))
                break
    return bad


if __name__ == "__main__":
    import os

    def banner(s):
        print("\n" + s + "\n" + "-" * max(66, len(s)))

    # MIDI 不放在仓库里; CANTABILE_MIDI_DIR 下的 xxx.mid 对应曲库里的 XXX
    mdir = os.environ.get("CANTABILE_MIDI_DIR", "")
    MIDI = {}
    if os.path.isdir(mdir):
        from cantabile.corpus import names as qumu
        youde = set(qumu())
        for f in sorted(os.listdir(mdir)):
            k = os.path.splitext(f)[0].upper()
            if f.endswith(".mid") and k in youde:
                MIDI[k] = os.path.join(mdir, f)

    def prf(got, ref, tol=0.06):
        ref = list(ref)
        used = [False] * len(ref)
        tp = 0
        for n in got:
            for j, (b, p, _) in enumerate(ref):
                if not used[j] and p == n.pitch and abs(b - n.onset) <= tol:
                    used[j] = True
                    tp += 1
                    break
        pr = tp / max(1, len(got))
        rc = tp / max(1, len(ref))
        f1 = 0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc)
        return pr, rc, f1, tp

    have_midi = bool(MIDI)
    if have_midi:
        from cantabile.corpus import get as qupu
        from cantabile.ingest.midi import load

        diyi = next(iter(MIDI))

        banner("melody recovery from four-part MIDI")
        f1s = []
        allocs = {}
        for tune, path in MIDI.items():
            raw, _, tmap = load(path)
            a = allocate(raw, bpb=qupu(tune).beats_per_bar,
                         tempo=tmap.bpm(0.0))
            allocs[tune] = a
            p, r, f, tp = prf(a.melody, qupu(tune).melody)
            f1s.append(f)
            print("  %-13s %3d in -> %3d melody / %3d accomp   "
                  "P %.3f  R %.3f  F1 %.3f  (%d/%d)"
                  % (tune, len(raw), len(a.melody), len(a.accompaniment),
                     p, r, f, tp, len(qupu(tune).melody)))
        print("  mean F1 %.3f" % (sum(f1s) / len(f1s)))
        assert sum(f1s) / len(f1s) > 0.95

        banner("rationale (%s)" % diyi)
        print(allocs[diyi].explain())

        banner("realise(): density")
        print("  reference: %s notes/sec total, %.0f%% of them below the tune"
              % (REFERENCE_TOTAL, 100 * BELOW_TUNE_SHARE))
        for tune in MIDI:
            a = allocs[tune]
            perf = realise(a)
            bpm = a.rationale["budget"]["bpm"]
            secs = perf.length * 60.0 / bpm
            nps = len(perf.notes) / secs
            below = sum(1 for n in perf.notes if n.voice is not Voice.MELODY)
            share = below / len(perf.notes)
            bad = _above_melody(perf)
            print("  %-13s %-8s %4d notes / %5.1fs at %.0f bpm = %5.2f "
                  "notes/sec" % (tune, a.style.name, len(perf.notes), secs,
                                 bpm, nps))
            print("                %.0f%% below the tune;  accompaniment "
                  "sounding at or above a simultaneous melody note: %d"
                  % (100 * share, len(bad)))
            assert not bad
            assert REFERENCE_TOTAL[0] <= nps <= REFERENCE_TOTAL[1], (tune, nps)
            assert share >= BELOW_TUNE_SHARE - 0.04, (tune, share)

        banner("low melody notes get no harmony")
        a = allocs[diyi]
        perf = realise(a)
        exposed = bare = packed = 0
        for n in a.melody:
            if n.dur < 0.5:
                continue
            under = [x for x in perf.notes
                     if x.tag == "allocate.melody.harmony"
                     and abs(x.onset - n.onset) < 1e-6]
            if n.pitch < a.style.expose_melody_below:
                exposed += 1
                bare += 1 if not under else 0
            else:
                packed += 1 if under else 0
        print("  %s: expose_below = %s" % (diyi, name(a.style.expose_melody_below)))
        print("  %d melody notes below it, %d without harmony"
              % (exposed, bare))
        print("  %d melody notes above it with harmony" % packed)
        assert exposed and bare == exposed

        banner("register bands (%s)" % diyi)
        bpm = a.rationale["budget"]["bpm"]
        secs = perf.length * 60.0 / bpm
        counts = {}
        for n in perf.notes:
            b = _band_of(n.pitch)
            counts[b] = counts.get(b, 0) + 1
        for band, _, _ in BAND_EDGES:
            rate = round(counts.get(band, 0) / secs, 2)
            lo, hi = REFERENCE_BANDS[band]
            flag = "in band" if lo <= rate <= hi else (
                "below" if rate < lo else "above")
            print("  %-6s %5.2f notes/sec   reference %.2f-%.2f   %s"
                  % (band, rate, lo, hi, flag))

        banner("playability")
        for tune in MIDI:
            rep = _play.check(realise(allocs[tune]))
            print("  %-13s ok() = %-5s  widest L %2d / R %2d over %d chords; "
                  "%d stretches; a hand moves %.1f semitones between chords "
                  "on average, %.0f at most"
                  % (tune, rep.ok(), rep.stats["playability.widest_left"],
                     rep.stats["playability.widest_right"],
                     rep.stats["playability.slices"],
                     rep.stats["playability.stretches"],
                     rep.stats["playability.mean_hand_move"],
                     rep.stats["playability.max_hand_move"]))
            assert rep.ok(), rep.summary()

        banner("analysis layer")
        st = allocs[diyi].rationale["structure"]
        print("  analyze.structure: %s" % st["source"])
        print("  policy:            %s" % st["policy"])
        for tune in MIDI:
            cc = allocs[tune].rationale["melody"]["cross_check"]
            if cc.get("source") == "analyze.melody":
                print("  %-13s analyze.melody agrees on %.0f%% of the tune "
                      "(%d notes); %d only here, %d only there"
                      % (tune, 100 * cc["agreement"], cc["agreed"],
                         len(cc["only_here"]), len(cc["only_there"])))
            else:
                print("  %-13s %s" % (tune, cc["source"]))
    else:
        print("\n  (reference MIDI not found; skipping corpus checks)")

    banner("choose_texture()")

    def tune(pairs, dur):
        return [Note(b, dur, p, 80, Voice.MELODY) for b, p in pairs]

    grid_fast = Harmony(dict(
        (float(i), [("C",), ("F",), ("G", "7"), ("C",), ("A", "m"), ("F",),
                    ("D", "m7"), ("G", "7")][i % 8]) for i in range(32)))
    grid_slow = Harmony({0: ("C",), 8: ("F",), 16: ("G", "7"), 24: ("C",)})
    grid_bar = Harmony({0: ("C",), 4: ("F",), 8: ("G", "7"), 12: ("C",)})

    cases = [
        ("busy tune, chord every 2 beats",
         tune([(i * 0.5, 67 + (i % 5)) for i in range(32)], 0.5),
         Harmony({0: ("C",), 2: ("F",), 4: ("G", "7"), 6: ("C",),
                  8: ("A", "m"), 10: ("F",), 12: ("G", "7"), 14: ("C",)}), 95),
        ("sparse high ballad tune, chord every 8 beats",
         tune([(i * 4.0, 74 + (i % 4)) for i in range(8)], 4.0),
         grid_slow, 68),
        ("four-part: one note per beat, harmony per beat",
         tune([(float(i), 64 + (i % 6)) for i in range(32)], 1.0),
         grid_fast, 88),
        ("tune that dips to B3",
         tune([(float(i), 59 + (i % 3)) for i in range(16)], 1.0),
         grid_bar, 72),
    ]
    for label, mel, grid, bpm in cases:
        tex, why = choose_texture(mel, grid, bpb=4, tempo=bpm)
        f = why["features"]
        print("\n  %s" % label)
        print("    -> %-10s (%s), margin %.2f over %s"
              % (why["chosen"], type(tex).__name__, why["margin"] or 0.0,
                 why["runner_up"]))
        print("       %.2f notes/beat, %.1f beats/chord, ceiling %s, "
              "room %d semitones, %d bpm"
              % (f["melody_per_beat"], f["harmonic_rhythm_beats"],
                 name(f["ceiling"]), f["room_semitones"], f["bpm"]))
        for c in why["candidates"]:
            print("       %-8s %6.2f  %-5.2f n/s  %s"
                  % (c["style"], c["score"], c["total_nps"], c["why"]))

    banner("register_plan()")
    for label, pitches in (("tune sits high (G4-D5)", [67, 69, 71, 72, 74]),
                           ("tune dips to D4", [62, 64, 67, 69, 72, 74]),
                           ("low tune (B3-G4)", [59, 60, 62, 64, 65, 67])):
        mel = [Note(float(i), 1.0, p, 80, Voice.MELODY)
               for i, p in enumerate(pitches)]
        c, e = register_plan(mel)
        print("  %-24s -> ceiling %-4s expose_below %-4s"
              % (label, name(c), name(e)))
    hi = register_plan([Note(0, 1, p, 80) for p in (67, 69, 71, 72, 74)])[0]
    lo = register_plan([Note(0, 1, p, 80) for p in (59, 60, 62, 64, 65)])[0]
    assert hi > lo

    banner("budget()")
    mel = tune([(float(i), 67 + (i % 5)) for i in range(64)], 1.0)
    b = budget(mel, grid_bar, REFERENCE_TOTAL, bpb=4, bpm=95)
    for k in ("bpm", "target_nps", "melody_per_beat", "melody_nps",
              "accomp_nps", "accomp_per_beat", "below_tune_share",
              "below_tune_target", "note"):
        print("  %-18s %s" % (k, b[k]))
    print("  bands (notes/sec)  %s" % b["bands_nps"])
    assert abs(b["below_tune_share"] - BELOW_TUNE_SHARE) < 0.05

    banner("budget(): melody too busy")
    busy = tune([(i * 0.25, 67 + (i % 5)) for i in range(128)], 0.25)
    b2 = budget(busy, grid_bar, (4.0, 5.0), bpb=4, bpm=95)
    print("  accomp_per_beat %s   ok %s" % (b2["accomp_per_beat"], b2["ok"]))
    print("  %s" % b2["note"])
    assert not b2["ok"]
