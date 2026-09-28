# -*- coding: utf-8 -*-
"""几种编配风格: 织体 + 音区 + 力度平衡。"""
from __future__ import annotations

from dataclasses import dataclass

from cantabile.arrange.textures import (Texture, QuaverAlternation, BrokenChord,
                                        BlockChords, Sustained, WalkingBass)
from cantabile.arrange.voicing import melody_harmony, open_left_hand

__all__ = ["Style", "ACCOMPANIMENT", "SOLO_BALLAD", "FOUR_PART",
           "SUSTAINED_PAD", "WALKING_BASS", "STYLES", "get", "names"]


@dataclass
class Style:
    name: str
    texture: Texture
    rh_harmony: int            # notes under the melody
    lh_ceiling: int            # highest left-hand pitch
    expose_melody_below: int   # no harmony under melody notes below this
    melody_gain: int           # melody velocity above the accompaniment
    accents: tuple             # (downbeat, mid-bar) velocity bumps
    notes_per_sec: tuple       # (min, max) target density for the audit
    beats_per_bar: float = 4.0
    bass_target: int = 40      # preferred left-hand bass pitch

    def ctx(self, bar_origin: float = 0.0, **extra) -> dict:
        c = {"ceiling": self.lh_ceiling,
             "target": self.bass_target,
             "beats_per_bar": self.beats_per_bar,
             "bar_origin": bar_origin,
             "accents": self.accents}
        c.update(extra)
        return c

    def render(self, chord, start: float, span: float, dyn: int, **extra) -> list:
        return self.texture.render(chord, start, span, dyn,
                                   self.ctx(extra.pop("bar_origin", 0.0), **extra))

    def harmony_under(self, mel_pitch: int, chord) -> list:
        return melody_harmony(mel_pitch, chord, self.rh_harmony,
                              expose_below=self.expose_melody_below)

    def left_hand(self, chord) -> dict:
        return open_left_hand(chord, target=self.bass_target,
                              lo=self.bass_target - 5, hi=self.bass_target + 9,
                              ceiling=self.lh_ceiling)

    def melody_velocity(self, dyn: int) -> int:
        return max(1, min(127, dyn + self.melody_gain))

    def density_ok(self, notes_per_sec: float) -> bool:
        lo, hi = self.notes_per_sec
        return lo <= notes_per_sec <= hi

    def __repr__(self):
        return (f"<Style {self.name}: {type(self.texture).__name__}, "
                f"rh={self.rh_harmony}, ceiling={self.lh_ceiling}, "
                f"gain=+{self.melody_gain}>")


ACCOMPANIMENT = Style(
    name="accompaniment",
    texture=QuaverAlternation(chord_notes=2),
    rh_harmony=2,
    lh_ceiling=55,            # G3, 放到 B3 中音区太挤
    expose_melody_below=62,   # D4
    melody_gain=14,
    accents=(3, 2),
    notes_per_sec=(5.8, 10.6),
)

SOLO_BALLAD = Style(
    name="ballad",
    texture=BrokenChord(notes_per_beat=2, span_octaves=2),
    rh_harmony=1,
    lh_ceiling=64,            # E4: room for the arpeggio to climb
    expose_melody_below=67,
    melody_gain=16,
    accents=(2, 1),
    notes_per_sec=(4.5, 7.5),
)

FOUR_PART = Style(
    name="four-part",
    texture=BlockChords(subdivision=1.0, chord_notes=1),
    rh_harmony=1,             # soprano + alto
    lh_ceiling=60,            # tenor can reach C4
    expose_melody_below=59,
    melody_gain=9,            # blend rather than project
    accents=(2, 1),
    notes_per_sec=(3.5, 7.0),  # four voices in crotchets, 80-105 bpm
)

SUSTAINED_PAD = Style(
    name="pad",
    texture=Sustained(),
    rh_harmony=1,
    lh_ceiling=57,
    expose_melody_below=60,
    melody_gain=18,
    accents=(1, 0),
    notes_per_sec=(1.5, 5.0),
)

WALKING_BASS = Style(
    name="walking",
    texture=WalkingBass(step=1.0),
    rh_harmony=2,
    lh_ceiling=55,
    expose_melody_below=60,
    melody_gain=14,
    accents=(3, 1),
    notes_per_sec=(3.0, 6.5),
)

STYLES = {s.name: s for s in (ACCOMPANIMENT, SOLO_BALLAD, FOUR_PART,
                              SUSTAINED_PAD, WALKING_BASS)}
_ALIASES = {
    "solo_ballad": "ballad", "solo": "ballad", "ballade": "ballad",
    "satb": "four-part", "four_part": "four-part",
    "sustained": "pad",
    "walking_bass": "walking", "walk": "walking",
}


def names() -> list:
    return sorted(STYLES)


def get(name) -> Style:
    if isinstance(name, Style):
        return name
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    try:
        return STYLES[key]
    except KeyError:
        raise KeyError(f"unknown style {name!r}; known: {', '.join(names())}")


if __name__ == "__main__":
    from cantabile.theory import Chord, name as pname

    for k in names():
        print(" ", repr(STYLES[k]))
    print()
    assert get("ACCOMPANIMENT") is ACCOMPANIMENT
    assert get("solo") is SOLO_BALLAD
    assert get(FOUR_PART) is FOUR_PART
    try:
        get("boogie")
    except KeyError as e:
        print("  unknown style raises:", e)

    st = ACCOMPANIMENT
    bb = Chord.parse("Bb")
    print("\naccompaniment, one bar of Bb at dyn 62, melody D5:")
    for n in sorted(st.render(bb, 0.0, 4.0, 62), key=lambda x: (x.onset, -x.pitch)):
        print(f"    {n.onset:5.2f} {pname(n.pitch, True):5s} v{n.vel:3d} "
              f"{n.voice.value:6s} {n.tag}")
    print("  RH under D5 :", [pname(p, True) for p in st.harmony_under(74, bb)])
    print("  RH under B3 :", st.harmony_under(59, bb))
    print("  melody vel  :", st.melody_velocity(62))
    assert max(n.pitch for n in st.render(bb, 0.0, 4.0, 62)) <= st.lh_ceiling
