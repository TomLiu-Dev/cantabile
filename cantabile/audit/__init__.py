"""检查 Performance 里的错音, 每条结果带拍、音高和 tag。

    harmony    和弦音、撞音, 以及修复
    melody     旋律全不全、在不在最上面、有没有被切断
    balance    从音频里看谁更响
    reference  和参考钢琴录音比

balance / reference 要 numpy 和 librosa, 用到时才 import。
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

from cantabile.model import Finding, Note, Performance, Report, TempoMap, Voice
from cantabile.theory import Harmony

from cantabile.audit.harmony import (check_chord_tones, check_clashes,
                                     resolve_clashes)
from cantabile.audit.melody import (check_integrity, check_never_masked,
                                    check_unisons, melody_register_report,
                                    protect_melody)

__all__ = [
    "full_audit",
    "check_chord_tones", "check_clashes", "resolve_clashes",
    "check_integrity", "check_never_masked", "check_unisons",
    "protect_melody", "melody_register_report",
    "melody_prominence", "spectral_profile",
    "profile", "compare", "REFERENCE_ACCOMPANIMENT",
    "demo", "banner", "synth_wav", "smoke",
]


# 音频那一半用到时才 import
def melody_prominence(*a, **k) -> Report:
    from cantabile.audit.balance import melody_prominence as f
    return f(*a, **k)


def spectral_profile(*a, **k) -> dict:
    from cantabile.audit.balance import spectral_profile as f
    return f(*a, **k)


def profile(*a, **k) -> dict:
    from cantabile.audit.reference import profile as f
    return f(*a, **k)


def compare(*a, **k) -> Report:
    from cantabile.audit.reference import compare as f
    return f(*a, **k)


def synth_wav(*a, **k) -> str:
    from cantabile.audit.balance import synth_wav as f
    return f(*a, **k)


def __getattr__(attr):
    if attr == "REFERENCE_ACCOMPANIMENT":
        from cantabile.audit.reference import REFERENCE_ACCOMPANIMENT
        return REFERENCE_ACCOMPANIMENT
    raise AttributeError(attr)


def full_audit(perf: Performance,
               harmony: Optional[Harmony] = None,
               tune: Optional[Sequence[Tuple[float, int, float]]] = None,
               wav: Optional[str] = None,
               offsets: Sequence[float] = (0.0,),
               references: Optional[object] = None) -> Report:
    """给了什么输入就跑什么检查, 结果合在一起。

    references 默认不比: 随便一首曲子和参考录音的统计比, 报出来的错没意义。
    """
    rep = Report()
    paoguo = []

    if harmony is not None:
        rep.merge(check_chord_tones(perf, harmony))
        paoguo.append("chord_tones")
    rep.merge(check_clashes(perf))
    rep.merge(check_never_masked(perf))
    rep.merge(check_unisons(perf))
    rep.merge(melody_register_report(perf))
    paoguo += ["clashes", "never_masked", "unisons", "melody_register"]

    if tune is not None:
        rep.merge(check_integrity(perf, tune, offsets=offsets))
        paoguo.append("melody_integrity")

    if wav is not None:
        rep.merge(melody_prominence(wav, perf))
        paoguo.append("melody_prominence")
        if references is not None:
            rep.merge(compare(profile(wav, perf), references))
            paoguo.append("reference")
    elif references is not None:
        rep.add(Finding(check="reference", beat=0.0, severity="info",
                        detail="references supplied but no wav to measure"))

    rep.stats["audit.checks"] = ",".join(paoguo)
    rep.stats["audit.errors"] = len(rep.errors)
    rep.stats["audit.warnings"] = len(
        [f for f in rep.findings if f.severity == "warn"])
    rep.stats["audit.notes"] = len(perf.notes)
    return rep


def banner(text: str) -> None:
    print("\n" + "=" * 78 + "\n== " + text + "\n" + "=" * 78)


def demo():
    """八拍的小片段, 故意埋了六个错:

      A  非和弦音     INNER  D4  @6  持续2拍   C 和弦上
      B  半音撞       INNER  E4 + F4 @4       都是 Fmaj7 的音
      C  和旋律同度   INNER  C5  @0.5         旋律 C5
      D  高过旋律     COLOUR G5  @2           旋律 B4
      E  和旋律半音   COLOUR G#4 @4           旋律 A4
      F  悄悄重弹     COLOUR A4  @6.0         旋律 A4 在 6.0 松开

    B 和弦音检查查不出来, 只有撞音检查能抓到。F 哪个检查都不报,
    但渲染时会把旋律最后的 A4 切断, 靠 protect_melody 修。
    """
    harm = Harmony({0.0: ("C", ""), 2.0: ("G", "7"),
                    4.0: ("F", "maj7"), 6.0: ("C", "")})
    xuanlv = [(0.0, 72, 2.0), (2.0, 71, 2.0), (4.0, 69, 2.0), (6.0, 72, 2.0)]

    perf = Performance(tempo=TempoMap([(0.0, 90.0)]))
    for pai, yingao, shichang in xuanlv:
        perf.add(Note(pai, shichang, yingao, 82, Voice.MELODY, "tune"))
    for pai, yingao in ((0.0, 48), (2.0, 43), (4.0, 41), (6.0, 48)):
        perf.add(Note(pai, 2.0, yingao, 70, Voice.BASS, "bass.walk"))
    for pai, yingao, lidu in ((0.0, 60, 58), (0.0, 64, 58),
                              (2.0, 59, 56), (2.0, 65, 56),
                              (4.0, 60, 58), (4.0, 64, 60), (4.0, 65, 52),
                              (6.0, 60, 58), (6.0, 64, 58)):
        perf.add(Note(pai, 2.0, yingao, lidu, Voice.INNER, "inner.pad"))

    perf.add(Note(6.0, 2.0, 62, 55, Voice.INNER, "colour.arp"))    # A
    perf.add(Note(0.5, 1.0, 72, 54, Voice.INNER, "inner.pad"))     # C
    perf.add(Note(2.0, 1.0, 79, 58, Voice.COLOUR, "colour.arp"))   # D
    perf.add(Note(4.0, 0.5, 68, 50, Voice.COLOUR, "colour.arp"))   # E
    perf.add(Note(6.0, 0.5, 69, 50, Voice.COLOUR, "colour.arp"))   # F
    return perf.sorted(), harm, xuanlv


def smoke() -> Report:
    """python3 -c "from cantabile.audit import smoke; smoke()" """
    perf, harm, xuanlv = demo()

    banner("full_audit on the DEFECTIVE fragment")
    r = full_audit(perf, harm, xuanlv)
    print(r.summary())
    print("\n  ok() = %s" % r.ok())

    banner("repair: resolve_clashes -> protect_melody")
    xiuhou = protect_melody(resolve_clashes(perf, harm))
    print("  resolve_clashes: %s" % xiuhou.meta["resolve_clashes"])
    print("  protect_melody : %s" % xiuhou.meta["protect_melody"])

    banner("full_audit on the REPAIRED fragment")
    r2 = full_audit(xiuhou, harm, xuanlv)
    print(r2.summary())
    print("\n  ok() = %s" % r2.ok())
    print("  Two findings survive on purpose, and neither has a repair pass:\n"
          "    D4 not in the C chord  - fixing it means re-choosing a pitch,\n"
          "                             which is the arranger's job;\n"
          "    G5 above the melody    - fixing it means re-voicing the figure.\n"
          "  An audit that silently rewrote these would be hiding the bug it\n"
          "  was built to expose.  It names them and stops.")
    return r2


if __name__ == "__main__":       # pragma: no cover
    smoke()
