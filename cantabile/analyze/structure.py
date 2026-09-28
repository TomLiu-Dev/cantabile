"""乐句切分和节拍权重。

参考: Cambouropoulos 2001 (LBDM); Temperley 2001; Lerdahl & Jackendoff 1983 (GTTM)。
"""
from __future__ import annotations

import numpy as np

from ..model import Finding, Note, Report

__all__ = [
    "lbdm", "phrases", "metric_weights", "metric_weight", "phrase_position",
    "segment_report", "gpr_strengths", "boundary_candidates", "as_events",
    "MAX_PITCH_INTERVAL", "MAX_TIME_INTERVAL",
]


MAX_PITCH_INTERVAL = 12.0
MAX_TIME_INTERVAL = 4.0

# 音程都加 1, 不然同音反复会让变化度跳到 1
PITCH_FLOOR = 1.0


def as_events(notes) -> list:
    """统一成排好序的 (onset, dur, pitch, vel)。"""
    jieguo = []
    for n in notes:
        if isinstance(n, Note):
            jieguo.append((float(n.onset), float(n.dur), int(n.pitch), int(n.vel)))
        elif isinstance(n, (tuple, list)):
            if len(n) == 3:
                beat, pitch, dur = n
                jieguo.append((float(beat), float(dur), int(pitch), 64))
            elif len(n) == 4:
                onset, dur, pitch, vel = n
                jieguo.append((float(onset), float(dur), int(pitch), int(vel)))
            else:
                raise ValueError(f"cannot read a {len(n)}-tuple as a note: {n!r}")
        else:
            raise TypeError(f"not a note: {n!r}")
    jieguo.sort(key=lambda e: (e[0], -e[2]))
    return jieguo


def _monophonic(events) -> list:
    # 和弦只留最高音, 不然和弦里 IOI=0 会被当成边界
    jieguo = []
    for onset, dur, pitch, vel in events:
        if jieguo and abs(onset - jieguo[-1][0]) < 1e-6:
            if pitch > jieguo[-1][2]:
                jieguo[-1] = (onset, dur, pitch, vel)
        else:
            jieguo.append((onset, dur, pitch, vel))
    return jieguo


def _degree_of_change(x: np.ndarray) -> np.ndarray:
    # r = |x[i] - x[i+1]| / (x[i] + x[i+1]); 两头是 0, 设成 1 会在最后一个音前多出一个边界
    n = len(x)
    r = np.zeros(n + 1)
    if n >= 2:
        a, b = x[:-1], x[1:]
        s = a + b
        with np.errstate(divide="ignore", invalid="ignore"):
            mid = np.where(s > 0, np.abs(a - b) / np.where(s > 0, s, 1.0), 0.0)
        r[1:-1] = mid
    return r


def _strength_profile(x: np.ndarray) -> np.ndarray:
    # s[i] = x[i] * (r[i-1, i] + r[i, i+1])
    if len(x) == 0:
        return x
    r = _degree_of_change(x)
    s = x * (r[:-1] + r[1:])
    top = float(s.max())
    return s / top if top > 0 else s


def lbdm(notes, *, w_pitch: float = 0.25, w_ioi: float = 0.5,
         w_rest: float = 0.25) -> list:
    """LBDM 边界强度, 返回每个音的 [(beat, strength)], 第一个音是 1.0。"""
    ev = _monophonic(as_events(notes))
    if len(ev) < 2:
        return [(e[0], 1.0) for e in ev]

    onset = np.array([e[0] for e in ev], dtype=float)
    dur = np.array([e[1] for e in ev], dtype=float)
    pitch = np.array([e[2] for e in ev], dtype=float)

    yincheng = np.minimum(np.abs(np.diff(pitch)) + PITCH_FLOOR, MAX_PITCH_INTERVAL)
    ioi = np.minimum(np.maximum(np.diff(onset), 0.0), MAX_TIME_INTERVAL)
    rest = np.minimum(np.maximum(onset[1:] - (onset[:-1] + dur[:-1]), 0.0),
                      MAX_TIME_INTERVAL)

    w = np.array([float(w_pitch), float(w_ioi), float(w_rest)], dtype=float)
    if w.sum() <= 0:
        raise ValueError("LBDM weights must not all be zero")
    w = w / w.sum()

    s = (w[0] * _strength_profile(yincheng)
         + w[1] * _strength_profile(ioi)
         + w[2] * _strength_profile(rest))
    top = float(s.max())
    if top > 0:
        s = s / top

    jieguo = [(float(onset[0]), 1.0)]
    jieguo.extend((float(onset[i + 1]), float(s[i])) for i in range(len(s)))
    return jieguo


def gpr_strengths(notes) -> list:
    """GTTM 的 GPR 2a/2b/3a-3d, 返回 [(beat, {rule: bool})], 记在 n3 上。"""
    ev = _monophonic(as_events(notes))
    n = len(ev)
    jieguo = []
    if n < 4:
        return jieguo

    onset = [e[0] for e in ev]
    dur = [e[1] for e in ev]
    pitch = [e[2] for e in ev]
    vel = [e[3] for e in ev]

    def ioi(i):
        return onset[i + 1] - onset[i]

    def rest(i):
        return max(0.0, onset[i + 1] - (onset[i] + dur[i]))

    def leap(i):
        return abs(pitch[i + 1] - pitch[i])

    def artic(i):
        step = ioi(i)
        return (dur[i] / step) if step > 0 else 1.0

    for j in range(1, n - 2):            # j 是 n2
        a, b, c = j - 1, j, j + 1
        r = {
            "gpr2a": rest(b) > rest(a) and rest(b) > rest(c),
            "gpr2b": ioi(b) > ioi(a) and ioi(b) > ioi(c),
            "gpr3a": leap(b) > leap(a) and leap(b) > leap(c),
            "gpr3b": (vel[b] != vel[c]) and (vel[a] == vel[b]) and (vel[c] == vel[c + 1]),
            "gpr3c": (abs(artic(b) - artic(a)) > 0.25
                      and abs(artic(a) - artic(max(a - 1, 0))) <= 0.25),
            "gpr3d": (abs(dur[b] - dur[c]) > 1e-6
                      and abs(dur[a] - dur[b]) <= 1e-6
                      and abs(dur[c] - dur[c + 1]) <= 1e-6),
        }
        jieguo.append((float(onset[c]), r))
    return jieguo


def _gpr_support(notes) -> dict:
    return {beat: sum(1 for v in rules.values() if v)
            for beat, rules in gpr_strengths(notes)}


def boundary_candidates(strengths, *, threshold: float = None,
                        peaks_only: bool = True) -> list:
    """LBDM 曲线上的候选边界, 强的在前。默认取 75 分位以上的峰。"""
    if not strengths:
        return []
    vals = np.array([s for _, s in strengths], dtype=float)
    if threshold is None:
        threshold = float(np.percentile(vals, 75.0)) if len(vals) > 3 else 0.0

    fengzhi = []
    for i in range(1, len(vals)):
        if vals[i] < threshold:
            continue
        if peaks_only:
            prev = vals[i - 1]
            nxt = vals[i + 1] if i + 1 < len(vals) else -1.0
            if not (vals[i] > prev and vals[i] >= nxt):
                continue
        fengzhi.append((strengths[i][0], float(vals[i])))
    fengzhi.sort(key=lambda p: (-p[1], p[0]))
    return fengzhi


def _greedy(houxuan, start, end, min_len, phase=None, bonus=0.0,
            beats_per_bar=4.0):
    # 从强到弱取, 每句至少 min_len 拍
    defen = []
    for beat, qiangdu in houxuan:
        if phase is not None and abs((beat % beats_per_bar) - phase) < 1e-6:
            qiangdu *= 1.0 + bonus
        defen.append((beat, qiangdu))
    defen.sort(key=lambda p: (-p[1], p[0]))

    fenjie = [start]
    for beat, _q in defen:
        if beat <= start + 1e-9 or beat >= end - 1e-9:
            continue
        if all(abs(beat - k) >= min_len - 1e-9 for k in fenjie) \
                and end - beat >= min_len - 1e-9:
            fenjie.append(beat)
    fenjie.sort()
    return fenjie


def phrases(notes, *, threshold: float = None, min_len: float = 2.0,
            parallelism: float = 0.4, beats_per_bar: int = 4) -> list:
    """把旋律切成乐句 [(start, end)]。parallelism > 0 时第二遍给落在同一小节位置的边界加分 (PSPR 3)。"""
    ev = _monophonic(as_events(notes))
    if not ev:
        return []
    start = ev[0][0]
    end = max(e[0] + e[1] for e in ev)
    if len(ev) < 3:
        return [(start, end)]

    qiangdu = lbdm(notes)
    bpb = float(beats_per_bar)

    if parallelism > 0:
        houxuan = boundary_candidates(qiangdu, threshold=threshold,
                                      peaks_only=False)
        fenjie = _greedy(houxuan, start, end, min_len, beats_per_bar=bpb)
        if len(fenjie) >= 3:
            weizhi = [round(b % bpb, 6) for b in fenjie[1:]]
            phase = max(set(weizhi), key=weizhi.count)
            fenjie = _greedy(houxuan, start, end, min_len, phase=phase,
                             bonus=parallelism, beats_per_bar=bpb)
    else:
        fenjie = _greedy(boundary_candidates(qiangdu, threshold=threshold),
                         start, end, min_len, beats_per_bar=bpb)

    bj = fenjie + [end]
    return [(bj[i], bj[i + 1]) for i in range(len(bj) - 1)]


def phrase_position(beat: float, phrases) -> float:
    """beat 在所在乐句里的位置 0..1。"""
    if not phrases:
        return 0.0
    for a, b in phrases:
        if a <= beat < b:
            return 0.0 if b <= a else float((beat - a) / (b - a))
    if beat < phrases[0][0]:
        return 0.0
    if beat >= phrases[-1][1]:
        return 1.0
    a, b = min(phrases, key=lambda p: min(abs(beat - p[0]), abs(beat - p[1])))
    return 0.0 if b <= a else float(min(1.0, max(0.0, (beat - a) / (b - a))))


def _levels(beats_per_bar: float, subdivisions: int, beat: float = 1.0) -> list:
    # 各层节拍的长度, 大的在前。4/4 细分4 -> [4, 2, 1, 0.5, 0.25]; 9/8 是 bpb=4.5, beat=1.5
    bar = float(beats_per_bar)
    b = float(beat) if beat else 1.0
    cengji = [bar]
    span = bar
    while span > b + 1e-9:
        half, third = span / 2.0, span / 3.0
        # 不用 %, 4.5 % 3 这种会出错
        if half >= b - 1e-9 and abs(half / b - round(half / b)) < 1e-6:
            span = half
        elif third >= b - 1e-9 and abs(third / b - round(third / b)) < 1e-6:
            span = third
        else:
            span = b
        cengji.append(span)
    if abs(cengji[-1] - b) > 1e-9:
        cengji.append(b)

    sub = int(subdivisions)
    span = b
    while sub > 1:
        if sub % 2 == 0:
            sub //= 2
            span /= 2.0
        elif sub % 3 == 0:
            sub //= 3
            span /= 3.0
        else:
            break
        cengji.append(span)
    return cengji


_GRID_CACHE = {}


def _grid(beats_per_bar, subdivisions, beat=1.0):
    # 返回的是缓存本身, 别改
    key = (float(beats_per_bar), int(subdivisions), float(beat))
    if key not in _GRID_CACHE:
        metric_weights(beats_per_bar, subdivisions, beat)
    return _GRID_CACHE[key]


def metric_weights(beats_per_bar: int = 4, subdivisions: int = 4,
                   beat: float = 1.0) -> dict:
    """一小节里每个位置的节拍权重 {offset: weight}, 强拍 1.0 (GTTM 打点数 / 层数)。"""
    key = (float(beats_per_bar), int(subdivisions), float(beat))
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    cengji = _levels(beats_per_bar, subdivisions, beat)
    n = len(cengji)
    step = cengji[-1]
    quanzhong = {}
    k = 0
    while k * step < beats_per_bar - 1e-9:
        # 用没四舍五入的值判断整除, 不然三连音会错
        raw = k * step
        dian = 0
        for L in cengji:
            q = raw / L
            if abs(q - round(q)) < 1e-6:
                dian += 1
        quanzhong[round(raw, 6)] = max(dian, 1) / n
        k += 1
    _GRID_CACHE[key] = quanzhong
    return dict(quanzhong)


def metric_weight(beat: float, beats_per_bar: int = 4,
                  beat_unit: float = 1.0) -> float:
    """某一拍的节拍权重 0..1; 不在格子上的给一个小的正数。"""
    biao = _grid(beats_per_bar, 4, beat_unit)
    off = round(float(beat) % float(beats_per_bar), 6)
    if off in biao:
        return biao[off]
    for grid, w in biao.items():
        if abs(grid - off) < 1e-6:
            return w
    zuidi = min(biao.values())
    xi = _grid(beats_per_bar, 12, beat_unit)    # 三连音和十六分
    for grid, w in xi.items():
        if abs(grid - off) < 1e-6:
            return max(zuidi * 0.5, min(zuidi, w))
    return zuidi * 0.5


def segment_report(notes, phrases) -> Report:
    """检查切分: 有没有空句、缝隙, 长短是否规整, 边界有没有 GPR 支持。"""
    rep = Report()
    ev = _monophonic(as_events(notes))
    if not ev:
        rep.add(Finding("segment", 0.0, "no notes to segment", "error"))
        return rep
    if not phrases:
        rep.add(Finding("segment", ev[0][0], "no phrases found", "error"))
        return rep

    qishi = [e[0] for e in ev]
    end = max(e[0] + e[1] for e in ev)
    changdu = [b - a for a, b in phrases]
    zhichi = _gpr_support(notes)

    for (a, b), (c, _d) in zip(phrases, phrases[1:]):
        if abs(b - c) > 1e-6:
            rep.add(Finding("segment.tile", b,
                            f"gap or overlap between phrases: {b:.3f} -> {c:.3f}",
                            "error"))
    if abs(phrases[-1][1] - end) > 1e-6:
        rep.add(Finding("segment.tile", phrases[-1][1],
                        f"last phrase ends at {phrases[-1][1]:.3f}, melody at {end:.3f}",
                        "error"))

    geshu = []
    for a, b in phrases:
        k = sum(1 for o in qishi if a - 1e-9 <= o < b - 1e-9)
        geshu.append(k)
        if k == 0:
            rep.add(Finding("segment.empty", a,
                            f"phrase {a:.2f}-{b:.2f} contains no notes", "error"))
        elif k == 1:
            rep.add(Finding("segment.short", a,
                            "single-note phrase (GPR 1 says avoid this)", "warn"))

    # GPR 5
    rounded = [round(L, 3) for L in changdu]
    modal = max(set(rounded), key=rounded.count) if rounded else 0.0
    for (a, b), L in zip(phrases, changdu):
        if modal > 0 and (L > 2.0 * modal or L < 0.5 * modal):
            rep.add(Finding("segment.irregular", a,
                            f"phrase is {L:.2f} beats against a modal {modal:.2f} - "
                            "likely a missed or spurious boundary", "warn"))

    tongyi = 0
    for a, _b in phrases[1:]:
        k = zhichi.get(a, 0)
        if k:
            tongyi += 1
        else:
            rep.add(Finding("segment.unsupported", a,
                            "LBDM boundary with no GTTM grouping rule firing "
                            "(GPR 2a/2b/3a-3d all silent)", "warn"))
    qidian = {a for a, _ in phrases}
    for beat, k in zhichi.items():
        if k >= 3 and beat not in qidian:
            rep.add(Finding("segment.missed", beat,
                            f"{k} grouping rules fire here but LBDM did not "
                            "put a boundary", "info"))

    inner = phrases[1:]
    rep.stats.update({
        "phrases": len(phrases),
        "phrase_len_mean": round(float(np.mean(changdu)), 3),
        "phrase_len_modal": modal,
        "phrase_len_min": round(float(min(changdu)), 3),
        "phrase_len_max": round(float(max(changdu)), 3),
        "notes_per_phrase": round(float(np.mean(geshu)), 2),
        "gpr_agreement": round(tongyi / len(inner), 3) if inner else 1.0,
    })
    return rep


if __name__ == "__main__":                                   # pragma: no cover
    from ..corpus import TUNES

    print("metric grid (GTTM dot count), 4/4 at 4 subdivisions:")
    print("  ", {off: round(w, 2) for off, w in sorted(metric_weights(4, 4).items())})
    print("metric grid, 3/4 at 4 subdivisions:")
    print("  ", {off: round(w, 2) for off, w in sorted(metric_weights(3, 4).items())})

    # 曲库在仓库外面 (~/.cantabile/corpus), 有就顺便切一下
    for nm in sorted(TUNES):
        tune = TUNES[nm]
        mel = tune.melody
        qiangdu = lbdm(mel)
        ph = phrases(mel, min_len=4.0)
        print(f"\n{nm}: {len(ph)} phrases over {tune.length:g} beats")
        print(f"   detected: {[round(a, 2) for a, _ in ph[1:]]}")
        print(segment_report(mel, ph).summary(limit=3))
        top = sorted(qiangdu, key=lambda x: -x[1])[:5]
        print("   strongest LBDM peaks:", [(round(b, 2), round(v, 2)) for b, v in top])
