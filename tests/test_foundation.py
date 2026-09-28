"""model、theory、曲库的测试。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cantabile.model import Note, Performance, TempoMap, Voice, Report, Finding
from cantabile.theory import (Chord, Harmony, chord_near, pc_near, stack,
                              is_clash, name, pc_of, QUALITIES)


def test_pc_near_never_substitutes_a_different_note():
    """范围不到一个八度也要给出对的音级。"""
    for pcname in ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"):
        for lo, hi in ((40, 50), (38, 49), (60, 61), (21, 108)):
            got = pc_near(pcname, (lo + hi) // 2, lo, hi)
            assert got % 12 == pc_of(pcname), (pcname, lo, hi, name(got))


def test_chord_near_only_returns_chord_tones():
    for root in range(12):
        for q in QUALITIES:
            for bass in range(12):
                ch = Chord(root, q, bass)
                b = pc_near(bass, 44, 38, 50)
                for target in (b + 4, b + 7, b + 10, b + 14, b + 16, b + 19):
                    got = chord_near(ch, target, b + 2, b + 22)
                    assert ch.has(got), (ch.label(), name(got))


def test_stack_respects_minimum_gap_and_ceiling():
    ch = Chord.parse("C", "maj7")
    got = stack(ch, 50, 69, 3, min_gap=2)
    assert len(got) == 3 and all(ch.has(p) for p in got)
    assert all(abs(a - b) >= 2 for a, b in zip(got, got[1:]))
    assert max(got) <= 69


def test_is_clash_covers_the_intervals_that_read_as_wrong_notes():
    assert is_clash(60, 61) and is_clash(60, 71) and is_clash(60, 73)
    assert not is_clash(60, 64) and not is_clash(60, 67) and not is_clash(60, 72)


def test_harmony_lookup_is_piecewise_constant():
    h = Harmony({0: ("G", "", None), 4: ("C", "", "E"), 6: ("D", "7", "F#")})
    assert h.at(0).label() == "G" and h.at(3.99).label() == "G"
    assert h.at(4).label() == "C/E" and h.at(5.9).label() == "C/E"
    assert h.at(6).label() == "D7/F#" and h.at(999).label() == "D7/F#"


def test_harmony_transposition_keeps_inversions():
    h = Harmony({0: ("G", "", "B")}).transposed(3)
    assert h.at(0).label() == "A#/D"


def test_tempo_map_integrates_to_real_time():
    t = TempoMap([(0, 60.0)])
    assert abs(t.seconds(4) - 4.0) < 1e-6          # 60 bpm 一秒一拍
    t2 = TempoMap([(0, 60.0), (8, 120.0)])
    assert t2.seconds(8) < 8.0                     # 加速了用时更短


def test_performance_melody_at_picks_the_top_line():
    p = Performance()
    p.add(Note(0, 4, 67, 90, Voice.MELODY, "m"),
          Note(0, 4, 79, 80, Voice.MELODY, "m.oct"),
          Note(0, 4, 55, 60, Voice.BASS, "b"))
    assert p.melody_at(2).pitch == 79
    assert len(p.accompaniment()) == 1


def test_voice_droppability_protects_structure():
    assert Voice.INNER.droppable and Voice.COLOUR.droppable
    assert not Voice.MELODY.droppable and not Voice.BASS.droppable


def test_lianxiqu_shuju_duibuduei():
    from lianxiqu import LIANXIQU
    for t in LIANXIQU.values():
        assert t.melody, t.name
        zongchang = sum(d for _, _, d in t.melody)
        assert abs(zongchang - t.length) < 1e-6, (t.name, zongchang)
        for beat, pitch, dur in t.melody:
            assert 21 <= pitch <= 108 and dur > 0
        assert min(t.harmony) == 0 and max(t.harmony) < t.length


def test_meiyou_sirenqupu_ye_neng_yong():
    from cantabile.corpus import tunes
    assert isinstance(tunes.names(), list)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f(); print(f"  pass  {f.__name__}")
        except Exception as e:
            bad += 1; print(f"  FAIL  {f.__name__}: {e}")
    print(f"\n{len(fns)-bad}/{len(fns)} passed")
    sys.exit(1 if bad else 0)
