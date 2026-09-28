# -*- coding: utf-8 -*-
"""编配: 哪只手弹哪些音, 在哪个八度, 什么时候弹。"""
from __future__ import annotations

from cantabile.arrange.voicing import (melody_harmony, open_left_hand,
                                       spread_for_hand, avoid_clashes,
                                       tone_between)
from cantabile.arrange.textures import (Texture, QuaverAlternation, BrokenChord,
                                        BlockChords, Sustained, WalkingBass)
from cantabile.arrange.styles import (Style, ACCOMPANIMENT, SOLO_BALLAD,
                                      FOUR_PART, SUSTAINED_PAD, WALKING_BASS,
                                      STYLES, get, names)

__all__ = [
    "melody_harmony", "open_left_hand", "spread_for_hand", "avoid_clashes",
    "tone_between",
    "Texture", "QuaverAlternation", "BrokenChord", "BlockChords", "Sustained",
    "WalkingBass",
    "Style", "ACCOMPANIMENT", "SOLO_BALLAD", "FOUR_PART", "SUSTAINED_PAD",
    "WALKING_BASS", "STYLES", "get", "names",
]
