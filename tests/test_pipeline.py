"""整条流水线: 每首练习曲都要能生成, 并且通过检查。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lianxiqu import LIANXIQU as TUNES
from cantabile.piece import Form, Piece, default_tempo
from cantabile.arrange import styles
from cantabile.audit.harmony import check_chord_tones, check_clashes
from cantabile.audit.melody import (check_integrity, check_tune, check_never_masked,
                                    check_unisons, melody_register_report)

STYLE = styles.get("accompaniment")
_LIZI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples") + os.sep


def _build(tune, repair=True):
    form = Form.strophic(tune)
    qu = Piece(tune, form, STYLE, default_tempo(92, form.total_bars * form.beats_per_bar,
                               beats_per_bar=form.beats_per_bar),
               tune.title)
    return qu.build(repair=repair)


def _audit(perf, tune):
    r = check_chord_tones(perf, perf.meta["harmony"])
    r.merge(check_clashes(perf))
    r.merge(check_never_masked(perf))
    r.merge(check_unisons(perf))
    r.merge(check_tune(perf, tune, tuple(perf.meta["repeats"])))
    return r


def test_every_tune_compiles_and_audits_clean():
    for tune in TUNES.values():
        perf = _build(tune)
        baogao = _audit(perf, tune)
        assert baogao.ok(), f"{tune.name}:\n" + "\n".join(str(f) for f in baogao.errors[:8])


def test_repair_actually_reduces_clashes():
    for tune in TUNES.values():
        yuanshi = _build(tune, repair=False)
        xiuhou = _build(tune, repair=True)
        qian = check_clashes(yuanshi).stats.get("clashes.pairs", 0)
        hou = check_clashes(xiuhou).stats.get("clashes.pairs", 0)
        # 有的曲子本来就没有撞音
        assert hou <= qian, (tune.name, qian, hou)
        if qian:
            assert hou < qian, (tune.name, qian, hou)
        assert not check_clashes(xiuhou).errors, tune.name


def test_repair_never_costs_a_melody_note():
    """修的时候可以删 INNER 和 COLOUR, MELODY 和 BASS 一个都不能少。"""
    for tune in TUNES.values():
        yuanshi = _build(tune, repair=False)
        xiuhou = _build(tune, repair=True)
        a = sorted((round(n.onset, 3), n.pitch) for n in yuanshi.melody())
        b = sorted((round(n.onset, 3), n.pitch) for n in xiuhou.melody())
        assert a == b, tune.name
        assert len(xiuhou.of(*[v for v in yuanshi.notes[0].voice.__class__
                               if v.name == "BASS"])) == len(yuanshi.of(
              *[v for v in yuanshi.notes[0].voice.__class__ if v.name == "BASS"])), tune.name


def test_density_lands_in_the_measured_reference_band():
    """音符密度在参考范围里。范围本来就包括 3/4 和 4/4 的录音, 不按拍号缩放。"""
    for tune in TUNES.values():
        perf = _build(tune)
        miao = perf.tempo.seconds(perf.length)
        nps = len(perf.notes) / miao
        lo, hi = STYLE.notes_per_sec
        assert lo <= nps <= hi, (tune.name, round(nps, 2), (lo, hi))


def test_melody_never_dips_into_the_accompaniment_register():
    """伴奏型的音不超过左手上限。"""
    from cantabile.model import Voice
    for tune in TUNES.values():
        perf = _build(tune)
        zuigao = max(n.pitch for n in perf.notes
                     if n.voice in (Voice.BASS, Voice.INNER, Voice.COLOUR)
                     and n.tag.split(".")[0] in ("quaver", "vary", "broken",
                                                 "block", "pad", "walk"))
        assert zuigao <= STYLE.lh_ceiling, (tune.name, zuigao)


def test_accompaniment_does_not_repeat_itself():
    """左手伴奏型每小节有变化。参考录音的节奏型熵 4.91 bit, 固定模板大约 2.08。"""
    from cantabile.arrange.varying import pattern_entropy, REFERENCE_ENTROPY
    for tune in TUNES.values():
        perf = _build(tune)
        bpb = tune.beats_per_bar
        # 只看左手, 右手和弦跟着旋律节奏走
        e = pattern_entropy([n.onset for n in perf.accompaniment()
                             if n.tag.startswith("vary")], bar=float(bpb))
        # 每小节位置少, 熵自然低; 参考是在 4/4 上量的
        xiaxian = (REFERENCE_ENTROPY[0] - 0.5) * (bpb / 4.0) ** 0.5
        # 复拍子位置更少, 本来就有重复的律动
        shangxian = 0.25 if getattr(tune, "compound", False) else 0.20
        assert e["entropy"] >= xiaxian, (tune.name, e, round(xiaxian, 2))
        assert e["modal_share"] <= shangxian, (tune.name, e)


def test_form_uses_the_tune_s_own_metre():
    for tune in TUNES.values():
        form = Form.strophic(tune)
        assert form.beats_per_bar == tune.beats_per_bar, tune.name
        perf = _build(tune)
        bpb = tune.beats_per_bar
        # 伴奏型的音都在自己那个小节里
        weizhi = [round((n.onset % bpb) * 4) / 4 for n in perf.accompaniment()
                  if n.tag.startswith("vary")]
        assert max(weizhi) < bpb, (tune.name, max(weizhi), bpb)


def test_bianshu_dengyu_duanshu():
    """几段就弹几遍, 不转调。"""
    for tune in TUNES.values():
        form = Form.strophic(tune)
        bian = [s for s in form.sections if s.label.startswith("verse")]
        assert len(bian) == tune.verses, tune.name
        assert all(s.transpose == 0 for s in form.sections), tune.name


def test_chord_spread_matches_the_reference():
    """参考录音: 中位数 11 ms, p95 46 ms, 16% 的和弦超过 25 ms。"""
    from cantabile.play import zuo_yanzou
    from cantabile.perform.timing import measure_spread
    m = measure_spread(zuo_yanzou(_LIZI + "turkish_march.mid", bpm=126))
    assert abs(m["median_ms"] - 11.0) <= 4.0, m
    assert abs(m["p95_ms"] - 46.0) <= 12.0, m
    assert abs(100 * m["rolled_frac"] - 16.0) <= 6.0, m


def test_play_budiu_yinfu():
    from cantabile.play import du_midi, qu_chong, zuo_yanzou
    yingui, *_ = du_midi(_LIZI + "turkish_march.mid")
    yuanpu = qu_chong([n for _, ns in yingui for n in ns])
    perf = zuo_yanzou(_LIZI + "turkish_march.mid", bpm=126)
    assert len(perf.notes) == len(yuanpu)
    assert sorted(n.pitch for n in perf.notes) == sorted(p for _, _, p, _ in yuanpu)


def test_play_xuanlv_rendui():
    from cantabile.play import zuo_yanzou
    from cantabile.model import Voice
    perf = zuo_yanzou(_LIZI + "turkish_march.mid", bpm=126)
    xuanlv = sorted((n for n in perf.notes if n.voice is Voice.MELODY), key=lambda n: n.onset)
    # B A G# A C | D C B C E
    assert [n.pitch for n in xuanlv[:10]] == [71, 69, 68, 69, 72, 74, 72, 71, 72, 76]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f(); print(f"  pass  {f.__name__}")
        except Exception as e:
            bad += 1; print(f"  FAIL  {f.__name__}: {str(e)[:200]}")
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
