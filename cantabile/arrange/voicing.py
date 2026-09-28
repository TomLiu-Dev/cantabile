# -*- coding: utf-8 -*-
"""和弦音怎么摆到两只手上。所有音都从 chord_near / pc_near 取, 不按音程硬加。"""
from __future__ import annotations

from cantabile.theory import Chord, chord_near, pc_near, is_clash, name

__all__ = [
    "melody_harmony", "open_left_hand", "spread_for_hand", "avoid_clashes",
    "tone_between",
]


def tone_between(chord: Chord, lo: int, hi: int, target: int):
    """[lo, hi] 里离 target 最近的和弦音, 没有就 None (不会放宽范围)。"""
    if lo > hi:
        return None
    if not chord.notes_in(lo, hi):
        return None
    return chord_near(chord, target, lo, hi)


def melody_harmony(mel_pitch: int, chord: Chord, count: int, *,
                   gap_below_melody: int = 3, min_gap: int = 2,
                   lowest: int = None, expose_below: int = None) -> list:
    """旋律音下面最多 count 个和弦音, 从高到低。"""
    if count <= 0:
        return []
    if expose_below is not None and mel_pitch < expose_below:
        return []

    hi = mel_pitch - gap_below_melody
    xiaxian = mel_pitch - 12 if lowest is None else lowest

    jieguo: list = []
    # 上面和弦音不够才一个八度一个八度往下找
    for jia in (0, 12, 24):
        jieguo = []
        lo = xiaxian - jia
        for n in reversed(chord.notes_in(lo, hi)):
            if n == mel_pitch:
                continue
            if is_clash(n, mel_pitch):
                continue
            if jieguo and abs(jieguo[-1] - n) < min_gap:
                continue
            jieguo.append(n)
            if len(jieguo) == count:
                break
        if len(jieguo) == count:
            break
    return jieguo


def open_left_hand(chord: Chord, *, target: int = 44, lo: int = 38, hi: int = 50,
                   ceiling: int = 59) -> dict:
    """左手开放排列 根音/五度/八度, 比如 Bb2-F3-Bb3。"""
    diyin = pc_near(chord.bass, target, lo, hi)

    wudu = tone_between(chord, diyin + 3, min(diyin + 11, ceiling), diyin + 7)

    badu = None
    if diyin + 12 <= ceiling:
        badu = pc_near(diyin % 12, diyin + 12, diyin + 6, max(diyin + 12, ceiling))
        if badu > ceiling or badu <= diyin:
            badu = None

    shangmian = badu if badu is not None else (wudu if wudu is not None else diyin)
    secai = tone_between(chord, shangmian + 2, ceiling, shangmian + 4)
    secai2 = None
    if secai is not None:
        secai2 = tone_between(chord, secai + 2, ceiling, secai + 3)

    shen = diyin - 12 if diyin - 12 >= 28 else None

    return {"bass": diyin, "fifth": wudu, "octave": badu,
            "colour": secai, "colour2": secai2, "deep": shen}


def spread_for_hand(pitches: list, max_span: int = 12) -> list:
    """把音收进一只手的跨度里, 最高音不动, 返回从高到低。"""
    ps = sorted({int(p) for p in pitches if p is not None}, reverse=True)
    if not ps:
        return []
    ding = ps[0]
    jieguo = [ding]
    for p in ps[1:]:
        q = p
        while ding - q > max_span:
            q += 12
        if q > ding:
            continue
        if any(abs(q - o) < 1 for o in jieguo):
            continue
        jieguo.append(q)
    return sorted(set(jieguo), reverse=True)


def avoid_clashes(pitches: list, against: list) -> list:
    """去掉跟 against 里任何音冲突的音, 顺序不变。"""
    liu = [p for p in against if p is not None]
    return [p for p in pitches
            if p is not None and not any(is_clash(p, a) for a in liu)]


if __name__ == "__main__":
    bb = Chord.parse("Bb")
    lh = open_left_hand(bb)
    print("open_left_hand(Bb):",
          {k: (name(v, True) if v is not None else None) for k, v in lh.items()})

    g7b = Chord.parse("G", "7", "B")
    lh2 = open_left_hand(g7b)
    print("open_left_hand(G7/B):",
          {k: (name(v) if v is not None else None) for k, v in lh2.items()})
    assert lh2["fifth"] is not None and g7b.has(lh2["fifth"])
    assert all(v is None or g7b.has(v) for v in lh2.values())

    mel = 74
    h = melody_harmony(mel, bb, 2)
    print("melody_harmony(D5, Bb, 2):", [name(p, True) for p in h])

    low = melody_harmony(59, bb, 2, expose_below=62)
    print("melody_harmony(B3, Bb, 2, expose_below=62):", low)

    wide = spread_for_hand([46, 53, 58, 36], max_span=12)
    print("spread_for_hand([36,46,53,58]):", [name(p, True) for p in wide])

    cl = avoid_clashes([65, 66, 71], against=[65])
    print("avoid_clashes([F4,F#4,B4] vs F4):", [name(p) for p in cl])
