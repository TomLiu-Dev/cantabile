"""和声检查: 长音是不是和弦音, 有没有持续的撞音; resolve_clashes 负责修。

修的时候伴奏给旋律让路, 两个伴奏音半音撞就删轻的那个。BASS 和 MELODY 不删。
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

from cantabile.model import Finding, Note, Performance, Report, Voice
from cantabile.theory import Harmony, chord_near, is_clash, name
from cantabile.audit.melody import _Window

__all__ = [
    "check_chord_tones",
    "check_clashes",
    "resolve_clashes",
    "CLASH_INTERVALS",
]

# 和旋律算撞的音程。同度也算, 一个键不能同时按两次
CLASH_INTERVALS = (0, 1, 11, 13)

_EPS = 1e-9


def _n(n: Note) -> str:
    return "%s(%d) vel=%d %s tag=%s" % (
        name(n.pitch), n.pitch, n.vel, n.voice.value, n.tag or "untagged")


def _pairs(notes: Sequence[Note], min_overlap: float):
    """重叠至少 min_overlap 拍的每一对音。"""
    paixu = sorted(notes, key=lambda n: n.onset)
    for i, a in enumerate(paixu):
        for j in range(i + 1, len(paixu)):
            b = paixu[j]
            if b.onset >= a.end:
                break
            if a.overlap(b) >= min_overlap:
                yield a, b


def check_chord_tones(perf: Performance,
                      harmony: Harmony,
                      min_dur: float = 1.0,
                      ignore: Iterable[Voice] = (Voice.MELODY,)) -> Report:
    """长于 min_dur 的音不在和弦里就报错 (短的经过音、辅助音不管)。"""
    rep = Report()
    ignore = set(ignore)
    jiancha = 0
    an_tag: dict = {}

    for n in sorted(perf.notes, key=lambda x: (x.onset, x.pitch)):
        if n.voice in ignore or n.dur < min_dur:
            continue
        jiancha += 1
        hexian = harmony.at(n.onset)
        if hexian.has(n.pitch):
            continue
        # 斜线和弦的低音 (比如 Cmaj7/D 的 D) 也算和声里的音
        if n.pitch % 12 == hexian.bass:
            continue
        tag = n.tag or "untagged"
        an_tag[tag] = an_tag.get(tag, 0) + 1
        zuijin = chord_near(hexian, n.pitch)
        rep.add(Finding(
            check="chord_tones",
            beat=n.onset,
            detail="%s is not in %s (held %.2f beats) - nearest chord tone %s"
                   % (_n(n), hexian.label(), n.dur, name(zuijin)),
            severity="error",
            notes=(n,),
        ))

    rep.stats["chord_tones.checked"] = jiancha
    rep.stats["chord_tones.bad"] = len(rep.findings)
    if an_tag:
        rep.stats["chord_tones.by_tag"] = dict(
            sorted(an_tag.items(), key=lambda kv: -kv[1]))
    return rep


def check_clashes(perf: Performance,
                  min_overlap: float = 0.2,
                  min_dur: float = 0.25) -> Report:
    """持续的半音 / 大七度 / 小九度。

    跟旋律撞的、还有纯半音, 算 error; 伴奏之间的大七小九算 warn, 开放排列里可能是故意的。
    """
    rep = Report()
    duishu = 0
    for a, b in _pairs(perf.notes, min_overlap):
        if a.dur < min_dur or b.dur < min_dur:
            continue
        if not is_clash(a.pitch, b.pitch):
            continue
        duishu += 1
        iv = abs(a.pitch - b.pitch)
        you_xuanlv = Voice.MELODY in (a.voice, b.voice)
        sev = "error" if (you_xuanlv or iv == 1) else "warn"
        zhonglei = {1: "semitone", 11: "major 7th", 13: "minor 9th"}[iv]
        rep.add(Finding(
            check="clashes",
            beat=max(a.onset, b.onset),
            detail="%s %s vs %s (overlap %.2f beats)%s"
                   % (zhonglei, _n(a), _n(b), a.overlap(b),
                      " - against the MELODY" if you_xuanlv else ""),
            severity=sev,
            notes=(a, b),
        ))

    rep.stats["clashes.pairs"] = duishu
    rep.stats["clashes.errors"] = len([f for f in rep.findings
                                       if f.severity == "error"])
    return rep


def _sounding_with(index, n, skip):
    return [m for m in index.around(n)
            if m is not n and id(m) not in skip and m.overlap(n) > _EPS]


def resolve_clashes(perf: Performance,
                    harmony: Optional[Harmony] = None,
                    max_move: int = 7) -> Performance:
    """修撞音, 返回新的 Performance。

    第一遍: 能删的音跟旋律撞了, 就挪到 max_move 半音以内、旋律下面、不跟别的撞的和弦音,
    挪不了就删。第二遍: 两个伴奏音差半音, 删轻的。BASS 和 MELODY 不删。
    """
    jieguo = perf.copy()
    yinfu = jieguo.notes
    xuanlv_suoyin = _Window([n for n in yinfu if n.voice is Voice.MELODY])
    quanbu_suoyin = _Window(yinfu)
    shandiao: set = set()
    n_moved = n_dropped = n_kept = 0

    # 第一遍: 伴奏 vs 旋律
    for n in list(yinfu):
        if not n.voice.droppable or id(n) in shandiao:
            continue
        xuanlv_zaixiang = [m for m in xuanlv_suoyin.around(n) if m.overlap(n) > _EPS]
        zhuang = [m for m in xuanlv_zaixiang
                  if abs(m.pitch - n.pitch) in CLASH_INTERVALS]
        if not zhuang:
            continue

        nuoguo = False
        if harmony is not None:
            hexian = harmony.at(n.onset)
            # 要低于所有在响的旋律音, 不然会触发 check_never_masked
            shangxian = min(m.pitch for m in xuanlv_zaixiang)
            qita = _sounding_with(quanbu_suoyin, n, shandiao)
            houxuan = [p for p in hexian.notes_in(max(21, n.pitch - max_move),
                                                  min(108, n.pitch + max_move))
                       if p < shangxian]
            houxuan.sort(key=lambda p: (abs(p - n.pitch), -p))
            for p in houxuan:
                if any(abs(m.pitch - p) in CLASH_INTERVALS
                       for m in xuanlv_zaixiang):
                    continue
                if any(abs(m.pitch - p) in CLASH_INTERVALS for m in qita):
                    continue
                n.pitch = p
                nuoguo = True
                n_moved += 1
                break
        if not nuoguo:
            shandiao.add(id(n))
            n_dropped += 1

    yinfu = [n for n in yinfu if id(n) not in shandiao]

    # 第二遍: 伴奏 vs 伴奏
    banzou = [n for n in yinfu if n.voice is not Voice.MELODY]
    shandiao2: set = set()
    for a, b in _pairs(banzou, _EPS):
        if id(a) in shandiao2 or id(b) in shandiao2:
            continue
        if abs(a.pitch - b.pitch) != 1:
            continue
        da, db = a.voice.droppable, b.voice.droppable
        if not da and not db:
            n_kept += 1
            continue
        if da and db:
            shan = a if (a.vel, a.dur, -a.pitch) <= (b.vel, b.dur, -b.pitch) else b
        else:
            shan = a if da else b
        shandiao2.add(id(shan))
        n_dropped += 1

    jieguo.notes = [n for n in yinfu if id(n) not in shandiao2]
    jieguo.meta = dict(jieguo.meta)
    jieguo.meta["resolve_clashes"] = {
        "moved": n_moved, "dropped": n_dropped,
        "kept_structural": n_kept,
        "before": len(perf.notes), "after": len(jieguo.notes),
    }
    return jieguo.sorted()


if __name__ == "__main__":
    from cantabile.audit import demo, banner

    perf, harm, tune = demo()
    banner("harmony.check_chord_tones  (expect exactly 1: D4 over a C chord)")
    print(check_chord_tones(perf, harm).summary())

    banner("harmony.check_clashes  (expect exactly 2: E4/F4, and G#4 vs melody)")
    print(check_clashes(perf).summary())

    xiuhou = resolve_clashes(perf, harm)
    banner("after resolve_clashes  %s" % xiuhou.meta["resolve_clashes"])
    print(check_clashes(xiuhou).summary())
    print("  chord tones after repair:")
    print(check_chord_tones(xiuhou, harm).summary())
