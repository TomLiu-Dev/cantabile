# -*- coding: utf-8 -*-
"""伴奏织体: 一个和弦上两只手弹什么。节奏和用哪个和弦音在这里定, 音区归 voicing。"""
from __future__ import annotations

from cantabile.model import Note, Voice
from cantabile.theory import Chord, pc_near, name
from cantabile.arrange.voicing import open_left_hand, tone_between, spread_for_hand

__all__ = ["Texture", "QuaverAlternation", "BrokenChord", "BlockChords",
           "Sustained", "WalkingBass"]

EPS = 1e-9


def _ctx(ctx, key, default):
    if not ctx:
        return default
    v = ctx.get(key, default)
    return default if v is None else v


def _vel(dyn, delta, lo=1, hi=127):
    return max(lo, min(hi, int(round(dyn + delta))))


class Texture:
    """织体基类。render(chord, start, span, dyn, ctx=None) -> [Note]

    ctx 可选的键: ceiling, lo, hi, target, beats_per_bar, bar_origin,
    accents, next_chord
    """
    name = "texture"

    def render(self, chord: Chord, start: float, span: float, dyn: int,
               ctx=None) -> list:
        raise NotImplementedError

    def _hand(self, chord, ctx):
        return open_left_hand(chord,
                              target=_ctx(ctx, "target", 44),
                              lo=_ctx(ctx, "lo", 38),
                              hi=_ctx(ctx, "hi", 50),
                              ceiling=_ctx(ctx, "ceiling", 59))

    def _accent(self, beat, ctx):
        zhongyin = _ctx(ctx, "accents", (0, 0))
        if not zhongyin:
            return 0
        meixiaojie = float(_ctx(ctx, "beats_per_bar", 4.0))
        weizhi = (beat - float(_ctx(ctx, "bar_origin", 0.0))) % meixiaojie
        if abs(weizhi) < EPS:
            return zhongyin[0]
        if len(zhongyin) > 1 and abs(weizhi - meixiaojie / 2.0) < EPS:
            return zhongyin[1]
        return 0

    @staticmethod
    def _grid(start, span, step):
        jieguo = []
        t = float(start)
        jieshu = start + span
        while t < jieshu - EPS:
            jieguo.append((t, min(step, jieshu - t)))
            t += step
        return jieguo

    def __repr__(self):
        return f"<{type(self).__name__}>"


class QuaverAlternation(Texture):
    """八分音符: 低音, 和弦, 五度, 和弦。"""
    name = "quaver"

    def __init__(self, chord_notes: int = 2, step: float = 0.5):
        self.chord_notes = chord_notes
        self.step = step

    def render(self, chord, start, span, dyn, ctx=None):
        h = self._hand(chord, ctx)
        diyin, wudu, badu = h["bass"], h["fifth"], h["octave"]
        if wudu is None:
            wudu = diyin
        shangmian = [p for p in (wudu, badu, h["colour"]) if p is not None]
        shangmian = spread_for_hand(shangmian or [diyin], 12)[: max(1, self.chord_notes)]
        shangmian = sorted(shangmian)

        # 没五度就用八度
        jiaoti = wudu if wudu != diyin else (badu if badu is not None else diyin)
        jieguo = []
        for i, (t, shichang) in enumerate(self._grid(start, span, self.step)):
            zhong = self._accent(t, ctx)
            wei = i % 4
            if wei == 0:
                jieguo.append(Note(t, shichang, diyin, _vel(dyn, zhong),
                                   Voice.BASS, "quaver.bass"))
            elif wei == 2:
                jieguo.append(Note(t, shichang, jiaoti, _vel(dyn, zhong - 4),
                                   Voice.BASS, "quaver.fifth"))
            else:
                for p in shangmian:
                    jieguo.append(Note(t, shichang, p, _vel(dyn, zhong - 9),
                                       Voice.INNER, "quaver.chord"))
        return jieguo


class BrokenChord(Texture):
    """分解和弦, 上去再下来。"""
    name = "broken"

    def __init__(self, notes_per_beat: int = 2, span_octaves: int = 2):
        self.notes_per_beat = max(1, int(notes_per_beat))
        self.span_octaves = max(1, int(span_octaves))

    def _ladder(self, chord, ctx):
        di = pc_near(chord.bass, _ctx(ctx, "target", 44),
                     _ctx(ctx, "lo", 38), _ctx(ctx, "hi", 50))
        ding = min(di + 12 * self.span_octaves, _ctx(ctx, "ceiling", 59))
        tiji = [di] + list(chord.notes_in(di + 1, ding))
        return sorted(set(tiji))

    def render(self, chord, start, span, dyn, ctx=None):
        tiji = self._ladder(chord, ctx)
        if not tiji:
            return []
        step = 1.0 / self.notes_per_beat
        n = len(tiji)
        zhouqi = max(1, 2 * n - 2)
        jieguo = []
        for i, (t, shichang) in enumerate(self._grid(start, span, step)):
            k = i % zhouqi
            idx = k if k < n else zhouqi - k
            p = tiji[idx]
            zhong = self._accent(t, ctx)
            if idx == 0:
                jieguo.append(Note(t, min(shichang * 2, start + span - t), p,
                                   _vel(dyn, zhong), Voice.BASS, "broken.bass"))
            elif p >= tiji[-1] - 4 and p > tiji[0] + 12:
                jieguo.append(Note(t, shichang, p, _vel(dyn, zhong - 10),
                                   Voice.COLOUR, "broken.top"))
            else:
                jieguo.append(Note(t, shichang, p, _vel(dyn, zhong - 6),
                                   Voice.INNER, "broken.arp"))
        return jieguo


class BlockChords(Texture):
    """低音加柱式和弦, 每 subdivision 拍一下。"""
    name = "block"

    def __init__(self, subdivision: float = 1.0, chord_notes: int = 3):
        self.subdivision = float(subdivision)
        self.chord_notes = int(chord_notes)

    def render(self, chord, start, span, dyn, ctx=None):
        h = self._hand(chord, ctx)
        shangxian = _ctx(ctx, "ceiling", 59)
        kuai = [p for p in (h["fifth"], h["octave"], h["colour"], h["colour2"])
                if p is not None and p <= shangxian]
        if not kuai:
            t2 = tone_between(chord, h["bass"] + 2, shangxian, h["bass"] + 7)
            kuai = [t2] if t2 else []
        kuai = spread_for_hand(kuai, 12)[: self.chord_notes]

        jieguo = []
        for t, shichang in self._grid(start, span, self.subdivision):
            zhong = self._accent(t, ctx)
            jieguo.append(Note(t, shichang, h["bass"], _vel(dyn, zhong),
                               Voice.BASS, "block.bass"))
            for p in sorted(kuai):
                jieguo.append(Note(t, shichang, p, _vel(dyn, zhong - 7),
                                   Voice.INNER, "block.chord"))
        return jieguo


class Sustained(Texture):
    """开头弹一个和弦, 一直按住。"""
    name = "sustained"

    def render(self, chord, start, span, dyn, ctx=None):
        h = self._hand(chord, ctx)
        shangxian = _ctx(ctx, "ceiling", 59)
        shangmian = [p for p in (h["fifth"], h["octave"], h["colour"])
                     if p is not None and p <= shangxian]
        zhong = self._accent(start, ctx)
        jieguo = [Note(start, span, h["bass"], _vel(dyn, zhong), Voice.BASS, "pad.bass")]
        for p in sorted(spread_for_hand(shangmian, 12)):
            jieguo.append(Note(start, span, p, _vel(dyn, zhong - 8),
                               Voice.INNER, "pad.chord"))
        return jieguo


class WalkingBass(Texture):
    """走动低音。有 next_chord 时最后一个音半音接过去 (唯一的非和弦音)。"""
    name = "walking"

    def __init__(self, step: float = 1.0):
        self.step = float(step)

    def render(self, chord, start, span, dyn, ctx=None):
        lo, hi = _ctx(ctx, "lo", 38), _ctx(ctx, "hi", 50)
        target = _ctx(ctx, "target", 44)
        xiayige = _ctx(ctx, "next_chord", None)
        wangge = self._grid(start, span, self.step)
        if not wangge:
            return []

        genyin = pc_near(chord.bass, target, lo, hi)
        shangxian = _ctx(ctx, "ceiling", 59)
        tiji = list(chord.notes_in(genyin, min(genyin + 12, shangxian)))
        if len(tiji) < 2:
            tiji = list(chord.notes_in(min(lo, genyin - 7),
                                       min(genyin + 12, shangxian)))
        tiji = tiji or [genyin]

        jieguo = []
        zuihou = len(wangge) - 1
        for i, (t, shichang) in enumerate(wangge):
            zhong = self._accent(t, ctx)
            if i == zuihou and xiayige is not None and zuihou > 0:
                mubiao = pc_near(xiayige.bass, target, lo, hi)
                qian = jieguo[-1].pitch
                p = mubiao - 1 if qian <= mubiao else mubiao + 1
                tag = "walk.bass" if chord.has(p) else "walk.approach"
                jieguo.append(Note(t, shichang, p, _vel(dyn, zhong - 2), Voice.BASS, tag))
                continue
            p = tiji[i % len(tiji)]
            tag = "walk.bass" if i % len(tiji) == 0 else "walk.step"
            jieguo.append(Note(t, shichang, p, _vel(dyn, zhong if i == 0 else zhong - 3),
                               Voice.BASS, tag))
        return jieguo


if __name__ == "__main__":
    def show(tex, chord, span=2.0, ctx=None):
        ns = tex.render(chord, 0.0, span, 64, ctx)
        print(f"{type(tex).__name__:18s} {str(chord):8s} {len(ns):2d} notes")
        for n in sorted(ns, key=lambda x: (x.onset, -x.pitch)):
            print(f"    {n.onset:5.2f} +{n.dur:4.2f}  {name(n.pitch, True):5s}"
                  f" v{n.vel:3d}  {n.voice.value:6s} {n.tag}")
        return ns

    ctx = {"accents": (3, 2), "beats_per_bar": 4.0, "ceiling": 59}
    bb = Chord.parse("Bb")
    g7b = Chord.parse("G", "7", "B")

    a = show(QuaverAlternation(), bb, 2.0, ctx)
    b = show(QuaverAlternation(), bb, 2.0, dict(ctx, bar_origin=0.0))
    assert [(n.onset, n.pitch, n.vel, n.tag) for n in a] == \
           [(n.onset, n.pitch, n.vel, n.tag) for n in b]
    print()

    show(BrokenChord(notes_per_beat=2, span_octaves=2), bb, 2.0, ctx)
    print()
    show(BlockChords(), g7b, 2.0, ctx)
    print()
    show(Sustained(), bb, 2.0, ctx)
    print()
    show(WalkingBass(), bb, 2.0, dict(ctx, next_chord=Chord.parse("Eb")))
