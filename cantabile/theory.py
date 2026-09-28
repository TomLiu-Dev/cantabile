"""音高和和弦。

声部要用 chord_near 取和弦音, 不要按音程硬加 (转位或者七和弦时 bass + 7 不一定是五音)。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence

NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NAMES_FLAT  = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

PC = {"C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
      "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7,
      "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11}

QUALITIES = {
    "":       (0, 4, 7),            "m":      (0, 3, 7),
    "6":      (0, 4, 7, 9),         "m6":     (0, 3, 7, 9),
    "7":      (0, 4, 7, 10),        "m7":     (0, 3, 7, 10),
    "maj7":   (0, 4, 7, 11),        "m7b5":   (0, 3, 6, 10),
    "dim":    (0, 3, 6),            "dim7":   (0, 3, 6, 9),
    "aug":    (0, 4, 8),
    "sus2":   (0, 2, 7),            "sus4":   (0, 5, 7),
    "7sus4":  (0, 5, 7, 10),
    "add9":   (0, 2, 4, 7),         "madd9":  (0, 2, 3, 7),
    "9":      (0, 2, 4, 7, 10),     "m9":     (0, 2, 3, 7, 10),
    "maj9":   (0, 2, 4, 7, 11),     "6/9":    (0, 2, 4, 7, 9),
}


def name(pitch: int, flats: bool = False) -> str:
    biao = NAMES_FLAT if flats else NAMES_SHARP
    return f"{biao[pitch % 12]}{pitch // 12 - 1}"


def pc_of(token) -> int:
    if isinstance(token, int):
        return token % 12
    try:
        return PC[token]
    except KeyError:
        raise ValueError(f"unknown pitch class {token!r}")


def pc_near(pc, target: int, lo: int = 21, hi: int = 108) -> int:
    """离 target 最近的、音级是 pc 的音, 优先在 [lo, hi] 里找。"""
    p = pc_of(pc)
    houxuan = [n for n in range(lo, hi + 1) if n % 12 == p]
    if not houxuan:
        base = target - ((target - p) % 12)
        houxuan = [n for n in (base, base + 12, base - 12) if 0 <= n <= 127] or [base % 12 + 60]
    jieguo = min(houxuan, key=lambda n: abs(n - target))
    assert jieguo % 12 == p, "pc_near broke its invariant"
    return jieguo


@dataclass(frozen=True)
class Chord:
    root: int                     # 音级 0-11
    quality: str = ""
    bass: int = None              # 音级, 默认等于 root

    @classmethod
    def parse(cls, root, quality: str = "", bass=None) -> "Chord":
        r = pc_of(root)
        return cls(r, quality, pc_of(bass) if bass not in (None, "") else r)

    def __post_init__(self):
        if self.quality not in QUALITIES:
            raise ValueError(f"unknown chord quality {self.quality!r}")
        if self.bass is None:
            object.__setattr__(self, "bass", self.root)

    @property
    def tones(self) -> frozenset:
        return frozenset((self.root + i) % 12 for i in QUALITIES[self.quality])

    def has(self, pitch: int) -> bool:
        return pitch % 12 in self.tones

    def notes_in(self, lo: int, hi: int) -> list:
        return [n for n in range(lo, hi + 1) if n % 12 in self.tones]

    def label(self, flats: bool = False) -> str:
        biao = NAMES_FLAT if flats else NAMES_SHARP
        s = biao[self.root] + self.quality
        return s if self.bass == self.root else f"{s}/{biao[self.bass]}"

    def __str__(self) -> str:
        return self.label()


def chord_near(chord: Chord, target: int, lo: int = 21, hi: int = 108) -> int:
    """[lo, hi] 里离 target 最近的和弦音, 找不到就放宽范围。"""
    c = chord.notes_in(lo, hi) or chord.notes_in(lo - 12, hi + 12) \
        or chord.notes_in(21, 108)
    return min(c, key=lambda n: abs(n - target))


def stack(chord: Chord, lo: int, hi: int, count: int,
          min_gap: int = 2, descending: bool = True) -> list:
    """在 [lo, hi] 里取 count 个和弦音, 间隔至少 min_gap; descending 时从上往下排。"""
    c = chord.notes_in(lo, hi)
    if descending:
        c = c[::-1]
    xuan = []
    for n in c:
        if xuan and abs(xuan[-1] - n) < min_gap:
            continue
        xuan.append(n)
        if len(xuan) == count:
            break
    return xuan


def is_clash(a: int, b: int) -> bool:
    """小二度 / 大七度 / 小九度, 听起来像错音。"""
    return abs(a - b) in (1, 11, 13)


class Harmony:
    """拍 -> 和弦, 一直到下一个和弦为止。"""

    def __init__(self, grid: dict):
        self.grid = {float(k): (v if isinstance(v, Chord) else Chord.parse(*v))
                     for k, v in grid.items()}
        self._keys = sorted(self.grid)
        if not self._keys:
            raise ValueError("empty harmony grid")

    def at(self, beat: float) -> Chord:
        lo, hi = 0, len(self._keys) - 1
        if beat < self._keys[0]:
            return self.grid[self._keys[0]]
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._keys[mid] <= beat:
                lo = mid
            else:
                hi = mid - 1
        return self.grid[self._keys[lo]]

    def spans(self):
        ks = self._keys
        for i, k in enumerate(ks):
            yield k, (ks[i + 1] if i + 1 < len(ks) else float("inf")), self.grid[k]

    def shifted(self, dt: float) -> "Harmony":
        return Harmony({k + dt: v for k, v in self.grid.items()})

    def transposed(self, semitones: int) -> "Harmony":
        return Harmony({k: Chord((c.root + semitones) % 12, c.quality,
                                 (c.bass + semitones) % 12)
                        for k, c in self.grid.items()})

    def __len__(self):
        return len(self.grid)
