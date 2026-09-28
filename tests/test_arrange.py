"""编配层的测试。和弦排列要把所有根音、性质、低音都扫一遍, 错音往往只在转位和七和弦上出现。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cantabile.theory import Chord, QUALITIES, is_clash, name
from cantabile.arrange.voicing import open_left_hand, melody_harmony
from cantabile.arrange import styles
from cantabile.model import Voice

ALL = [Chord(r, q, b) for r in range(12) for q in QUALITIES for b in range(12)]


def test_left_hand_is_always_in_the_chord():
    """左手除了低音和它的八度, 都得是和弦音。"""
    for ch in ALL:
        v = open_left_hand(ch, ceiling=59)
        b = v.get("bass")
        assert b is not None and b % 12 == ch.bass, ch.label()
        for k, p in v.items():
            if p is None or k == "bass":
                continue
            if k == "octave" and p == b + 12:
                continue
            if k == "deep" and p == b - 12:
                continue
            assert ch.has(p), (ch.label(), k, name(p))


def test_left_hand_respects_the_ceiling():
    for ceiling in (55, 59, 64):
        for ch in ALL:
            for k, p in open_left_hand(ch, ceiling=ceiling).items():
                if p is None or k in ("bass", "deep"):
                    continue
                assert p <= ceiling, (ch.label(), k, name(p), ceiling)


def test_the_fifth_is_never_a_raw_interval():
    """G7/B 上 bass + 7 是 F#, 五音一定要取和弦音。"""
    ch = Chord.parse("G", "7", "B")
    fifth = open_left_hand(ch)["fifth"]
    assert ch.has(fifth), name(fifth)
    assert fifth % 12 != 6, "bass+7 would give F#, clashing with the chord's F"


def test_right_hand_never_reaches_or_clashes_with_the_melody():
    cuo = []
    for ch in (Chord(r, q) for r in range(12) for q in QUALITIES):
        for mel in range(60, 80):
            for p in melody_harmony(mel, ch, 3):
                if p >= mel or not ch.has(p) or is_clash(p, mel):
                    cuo.append((ch.label(), name(mel), name(p)))
    assert not cuo, cuo[:5]


def test_low_melody_notes_are_left_bare():
    """低于 expose_below 的旋律音下面不垫和声。"""
    ch = Chord.parse("G")
    assert melody_harmony(60, ch, 2, expose_below=62) == []
    assert melody_harmony(71, ch, 2, expose_below=62) != []


def test_accompaniment_texture_is_bar_to_bar_identical():
    """编配是确定的, 随机的东西放在演奏里。"""
    banzouxing = styles.get("accompaniment").texture
    a = banzouxing.render(Chord.parse("Bb"), 0, 2, 62, {"ceiling": 59})
    b = banzouxing.render(Chord.parse("Bb"), 8, 2, 62, {"ceiling": 59})
    assert [(n.pitch, round(n.onset, 6), n.vel) for n in a] == \
           [(n.pitch, round(n.onset - 8, 6), n.vel) for n in b]


def test_every_generated_note_carries_provenance():
    for st in styles.STYLES.values():
        for n in st.texture.render(Chord.parse("C", "maj7"), 0, 2, 60,
                                   {"ceiling": 59}):
            assert isinstance(n.voice, Voice) and n.tag, (st.name, n)
            assert 1 <= n.vel <= 127 and n.dur > 0


def test_banzou_fengge_canshu():
    st = styles.get("accompaniment")
    assert st.melody_gain == 14 and st.lh_ceiling == 55 and st.rh_harmony == 2
    assert st.accents == (3, 2)
    assert st.notes_per_sec == (5.8, 10.6)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f(); print(f"  pass  {f.__name__}")
        except Exception as e:
            bad += 1; print(f"  FAIL  {f.__name__}: {str(e)[:160]}")
    print(f"\n{len(fns)-bad}/{len(fns)} passed  ({len(ALL)} chords swept)")
    sys.exit(1 if bad else 0)
