"""旋律检查: 旋律全不全, 听不听得见。

check_integrity 每一遍都对一下旋律, 报缺的、错位的、被切短的;
check_never_masked 报伴奏高过旋律; check_unisons 报伴奏和旋律同音
(一个键不能响两次, 渲染时会切掉一个, 一般切的是旋律)。
protect_melody 修同度和重弹, 不会为了伴奏去缩短旋律。
"""
from __future__ import annotations

import bisect
from typing import List, Sequence, Tuple

from cantabile.model import Finding, Note, Performance, Report, Voice
from cantabile.theory import name

__all__ = [
    "check_integrity",
    "check_never_masked",
    "check_unisons",
    "protect_melody",
    "melody_register_report",
    "TRUNCATION_RATIO",
    "RESTRIKE_GAP",
]

# 旋律音短于写的时值的这个比例就算被切
TRUNCATION_RATIO = 0.70

# 同一个音松开到下次按下至少空这么多拍; 同一个 tick 上的 note-off 和 note-on 有的渲染器会丢一个
RESTRIKE_GAP = 0.02

# 剪完比这还短就不要了
MIN_KEEP = 0.03

_EPS = 1e-9


def _n(n: Note) -> str:
    return "%s(%d) vel=%d %s tag=%s" % (
        name(n.pitch), n.pitch, n.vel, n.voice.value, n.tag or "untagged")


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


class _Window:
    """按起始时间排好, 快速找可能和某个音重叠的音。"""

    def __init__(self, notes: Sequence[Note]):
        self.notes = sorted(notes, key=lambda n: n.onset)
        self.onsets = [n.onset for n in self.notes]
        self.max_dur = max((n.dur for n in self.notes), default=0.0)

    def around(self, n: Note) -> List[Note]:
        lo = bisect.bisect_left(self.onsets, n.onset - self.max_dur)
        hi = bisect.bisect_left(self.onsets, n.end)
        return self.notes[lo:hi]


def check_integrity(perf: Performance,
                    tune: Sequence[Tuple[float, int, float]],
                    offsets: Sequence[float] = (0.0,),
                    tol: float = 0.15) -> Report:
    """每一遍 (offsets) 旋律的每个音都要在, 起始差 tol 拍以内、音高一样才算。"""
    rep = Report()
    xuanlv = perf.melody()
    yongguo: set = set()
    expected = present = missing = truncated = wrong_oct = wrong_pitch = 0

    for off in offsets:
        for pai, yingao, shichang in tune:
            expected += 1
            want = pai + off
            fujin = [m for m in xuanlv if abs(m.onset - want) <= tol]
            duide = [m for m in fujin if m.pitch == yingao]
            # 优先用还没被认领的, 一个音不能顶两个
            chi = [m for m in duide if id(m) not in yongguo] or duide
            xuan = (min(chi, key=lambda m: abs(m.onset - want))
                    if chi else None)

            if xuan is not None:
                yongguo.add(id(xuan))
                present += 1
                if xuan.dur < shichang * TRUNCATION_RATIO:
                    truncated += 1
                    rep.add(Finding(
                        check="melody_integrity", beat=want,
                        detail="%s truncated to %.2f of %.2f beats (%.0f%% of "
                               "its value)" % (_n(xuan), xuan.dur, shichang,
                                               100.0 * xuan.dur / shichang),
                        severity="error", notes=(xuan,)))
                continue

            tongji = [m for m in fujin if m.pitch % 12 == yingao % 12]
            if tongji:
                wrong_oct += 1
                got = min(tongji, key=lambda m: abs(m.onset - want))
                rep.add(Finding(
                    check="melody_integrity", beat=want,
                    detail="expected %s(%d), found the right pitch class in the "
                           "wrong octave: %s" % (name(yingao), yingao, _n(got)),
                    severity="error", notes=(got,)))
            elif fujin:
                wrong_pitch += 1
                got = min(fujin, key=lambda m: abs(m.onset - want))
                rep.add(Finding(
                    check="melody_integrity", beat=want,
                    detail="expected %s(%d), found %s"
                           % (name(yingao), yingao, _n(got)),
                    severity="error", notes=(got,)))
            else:
                missing += 1
                rep.add(Finding(
                    check="melody_integrity", beat=want,
                    detail="MISSING %s(%d), %.2f beats - no melody note within "
                           "%.2f beats of here" % (name(yingao), yingao, shichang, tol),
                    severity="error", notes=()))

    rep.stats["melody_integrity.expected"] = expected
    rep.stats["melody_integrity.present"] = present
    rep.stats["melody_integrity.missing"] = missing
    rep.stats["melody_integrity.wrong_octave"] = wrong_oct
    rep.stats["melody_integrity.wrong_pitch"] = wrong_pitch
    rep.stats["melody_integrity.truncated"] = truncated
    rep.stats["melody_integrity.present_pct"] = (
        round(100.0 * present / expected, 1) if expected else 0.0)
    rep.stats["melody_integrity.extra_melody_notes"] = max(
        0, len(xuanlv) - len(yongguo))
    return rep


def check_never_masked(perf: Performance, min_overlap: float = 0.15) -> Report:
    """伴奏音高过同时在响的旋律音。耳朵会把最上面那条当旋律, 所以都算 error。"""
    rep = Report()
    xuanlv = perf.melody()
    banzou = _Window(perf.accompaniment())
    zuida = 0
    shouying: set = set()

    for m in xuanlv:
        for a in banzou.around(m):
            ov = a.overlap(m)
            if ov < min_overlap or a.pitch <= m.pitch:
                continue
            shouying.add(id(m))
            gao = a.pitch - m.pitch
            zuida = max(zuida, gao)
            rep.add(Finding(
                check="never_masked", beat=max(a.onset, m.onset),
                detail="%s sounds %d semitones ABOVE melody %s for %.2f beats"
                       % (_n(a), gao, _n(m), ov),
                severity="error", notes=(a, m)))

    rep.stats["never_masked.melody_notes"] = len(xuanlv)
    rep.stats["never_masked.masked_melody_notes"] = len(shouying)
    rep.stats["never_masked.offenders"] = len(rep.findings)
    rep.stats["never_masked.worst_semitones_above"] = zuida
    return rep


def check_unisons(perf: Performance, min_overlap: float = 0.10) -> Report:
    """伴奏和在响的旋律音同一个音高。钢琴一个音只有一个键, 通常被切的是旋律。"""
    rep = Report()
    xuanlv = perf.melody()
    banzou = _Window(perf.accompaniment())
    shouying: set = set()

    for m in xuanlv:
        for a in banzou.around(m):
            if a.pitch != m.pitch:
                continue
            ov = a.overlap(m)
            if ov < min_overlap:
                continue
            shouying.add(id(m))
            houlai = a.onset > m.onset
            rep.add(Finding(
                check="unisons", beat=max(a.onset, m.onset),
                detail="%s doubles melody %s for %.2f beats - %s"
                       % (_n(a), _n(m), ov,
                          "the melody is already sounding and will be the note "
                          "that gets cut" if houlai else
                          "the melody strikes into a held accompaniment note"),
                severity="error", notes=(a, m)))

    rep.stats["unisons.melody_notes"] = len(xuanlv)
    rep.stats["unisons.melody_notes_hit"] = len(shouying)
    rep.stats["unisons.pairs"] = len(rep.findings)
    return rep


def protect_melody(perf: Performance) -> Performance:
    """去掉同度, 处理重弹, 不缩短旋律。

    1. 删掉和在响的旋律音同度的伴奏音。
    2. 同一个音高, 前一个音在下次按下前 RESTRIKE_GAP 拍松开。前一个是旋律就删后来的伴奏音,
       不然剪短前一个, 剪得太短又能删就删。
    """
    jieguo = perf.copy()
    xuanlv = _Window([n for n in jieguo.notes if n.voice is Voice.MELODY])

    # 1: 同度
    shan: set = set()
    for a in jieguo.notes:
        if a.voice is Voice.MELODY:
            continue
        for m in xuanlv.around(a):
            if a.pitch == m.pitch and a.overlap(m) > _EPS:
                shan.add(id(a))
                break
    n_unison = len(shan)
    yinfu = [n for n in jieguo.notes if id(n) not in shan]

    # 2: 重弹
    n_trim = n_drop = n_forced = 0
    an_yingao: dict = {}
    for n in yinfu:
        an_yingao.setdefault(n.pitch, []).append(n)

    shan2: set = set()
    for yingao, zu in an_yingao.items():
        zu.sort(key=lambda n: (n.onset, -n.dur))
        qian = None
        for cur in zu:
            if qian is None:
                qian = cur
                continue
            limit = cur.onset - RESTRIKE_GAP
            if qian.end <= limit + _EPS:
                qian = cur
                continue
            # 同一个键上重叠了, 看谁让
            if qian.voice is Voice.MELODY and cur.voice is not Voice.MELODY:
                shan2.add(id(cur))
                n_drop += 1
                continue
            if cur.voice is Voice.MELODY and qian.voice is not Voice.MELODY:
                if limit - qian.onset < MIN_KEEP:
                    shan2.add(id(qian))
                    n_drop += 1
                else:
                    qian.dur = limit - qian.onset
                    n_trim += 1
                qian = cur
                continue
            # 两边同类: 剪前面那个
            if limit - qian.onset < MIN_KEEP:
                if qian.voice.droppable:
                    shan2.add(id(qian))
                    n_drop += 1
                else:
                    qian.dur = max(MIN_KEEP, limit - qian.onset)
                    n_forced += 1
            else:
                qian.dur = limit - qian.onset
                n_trim += 1
            qian = cur

    jieguo.notes = [n for n in yinfu if id(n) not in shan2]
    jieguo.meta = dict(jieguo.meta)
    jieguo.meta["protect_melody"] = {
        "unisons_dropped": n_unison,
        "restrike_trimmed": n_trim,
        "restrike_dropped": n_drop,
        "forced_overlaps": n_forced,
        "before": len(perf.notes), "after": len(jieguo.notes),
    }
    return jieguo.sorted()


def melody_register_report(perf: Performance) -> Report:
    """旋律音区的中位数、p10、p90。太低或者太窄 (可能被音区拟合压扁了) 给 warn。"""
    rep = Report()
    xuanlv = perf.melody()
    if not xuanlv:
        rep.add(Finding(check="melody_register", beat=0.0,
                        detail="no MELODY notes in this performance at all",
                        severity="error"))
        rep.stats["melody_register.count"] = 0
        return rep

    ps = [n.pitch for n in xuanlv]
    med, p10, p90 = _pct(ps, 0.5), _pct(ps, 0.10), _pct(ps, 0.90)
    rep.stats["melody_register.count"] = len(ps)
    rep.stats["melody_register.median"] = "%s (%.1f)" % (name(int(round(med))), med)
    rep.stats["melody_register.p10"] = "%s (%.1f)" % (name(int(round(p10))), p10)
    rep.stats["melody_register.p90"] = "%s (%.1f)" % (name(int(round(p90))), p90)
    rep.stats["melody_register.span"] = "%d semitones (%s-%s)" % (
        max(ps) - min(ps), name(min(ps)), name(max(ps)))

    rep.add(Finding(check="melody_register", beat=0.0, severity="info",
                    detail="%d melody notes, median %s, p10 %s, p90 %s"
                           % (len(ps), name(int(round(med))),
                              name(int(round(p10))), name(int(round(p90))))))
    if p90 - p10 < 4:
        rep.add(Finding(check="melody_register", beat=0.0, severity="warn",
                        detail="p10-p90 spread is only %.1f semitones - the tune "
                               "may have been flattened by a register fit"
                               % (p90 - p10)))
    if med < 55:
        rep.add(Finding(check="melody_register", beat=0.0, severity="warn",
                        detail="median melody pitch %s is below G3; the tune is "
                               "in the accompaniment's register and will be hard "
                               "to pick out" % name(int(round(med)))))
    return rep


def check_tune(perf: Performance, tune, repeats: Sequence[float],
               tol: float = 0.15) -> Report:
    """带弱起的曲子用这个: 最后一遍没有弱起, 最后一个音延长。"""
    ruoqi = getattr(tune, "pickup", 0.0) or 0.0
    reps = tuple(repeats)
    if not ruoqi or not reps:
        return check_integrity(perf, tune.melody, reps, tol=tol)
    cut = tune.length - ruoqi
    zuihou = [(b, p, d) for b, p, d in tune.melody if b < cut - 1e-6]
    b, p, d = zuihou[-1]
    zuihou[-1] = (b, p, d + ruoqi)
    r = check_integrity(perf, tune.melody, reps[:-1], tol=tol) if reps[:-1] \
        else Report()
    r.merge(check_integrity(perf, zuihou, reps[-1:], tol=tol))
    return r


if __name__ == "__main__":
    from cantabile.audit import demo, banner

    perf, harm, tune = demo()

    banner("melody.check_integrity  (tune is complete -> expect clean)")
    print(check_integrity(perf, tune).summary())

    banner("melody.check_integrity on a damaged copy  (delete 1, truncate 1)")
    huaide = perf.copy()
    huaide.notes = [n for n in huaide.notes
                    if not (n.voice is Voice.MELODY and n.onset == 4.0)]
    for n in huaide.notes:
        if n.voice is Voice.MELODY and n.onset == 2.0:
            n.dur = 0.8
    print(check_integrity(huaide, tune).summary())

    banner("melody.check_never_masked  (expect exactly 1: G5 over B4)")
    print(check_never_masked(perf).summary())

    banner("melody.check_unisons  (expect exactly 1: inner C5 on melody C5)")
    print(check_unisons(perf).summary())

    xiuhou = protect_melody(perf)
    banner("after protect_melody  %s" % xiuhou.meta["protect_melody"])
    print(check_unisons(xiuhou).summary())
    print("  integrity after repair:")
    print(check_integrity(xiuhou, tune).summary())

    banner("melody.melody_register_report")
    print(melody_register_report(perf).summary())
