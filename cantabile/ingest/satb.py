"""从四声部 (SATB) MIDI 导入: 旋律取高音声部, 和弦按每格里响的声部标。

每一遍都连着录的文件可以用 one_pass 只截一遍。
"""
from __future__ import annotations
from collections import defaultdict

import mido

from ..theory import Chord, QUALITIES

__all__ = ["voices", "one_pass", "soprano", "harmony_from_voices", "chart"]


def voices(path) -> list:
    """每轨的 [(onset, dur, pitch)], 空轨去掉, 按平均音高从高到低排。"""
    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat or 480
    shengbu = []
    for tr in mid.tracks:
        t, on, yin = 0, {}, []
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                on[(msg.channel, msg.note)] = t
            elif msg.type in ("note_off", "note_on") and (msg.channel, msg.note) in on:
                a = on.pop((msg.channel, msg.note))
                yin.append((a / tpb, (t - a) / tpb, msg.note))
        if yin:
            shengbu.append(sorted(yin))
    shengbu.sort(key=lambda v: -sum(p for _, _, p in v) / len(v))
    return shengbu


def one_pass(tracks, length: float, start: float = 0.0) -> list:
    """只留 [start, start + length) 里开始的音, 也就是一遍。"""
    return [[(o - start, min(d, start + length - o), p) for o, d, p in v
             if start - 1e-6 <= o < start + length - 1e-6] for v in tracks]


def soprano(track, grid: float = 0.25, split_top: bool = False) -> list:
    """旋律 [(beat, pitch, dur)], 对齐到 grid。空隙不到一格就接到下一个音。

    split_top: 一个谱表上两个声部, 每个起点只留最高音。
    """
    yin = sorted(track)
    if split_top:
        # 上面还按着更高的音时, 是下声部在动, 跳过
        top = {}
        for o, d, p in yin:
            shangmian = any(q > p and oo < o - 1e-6 and oo + dd > o + 1e-3
                            for oo, dd, q in yin)
            if shangmian:
                continue
            k = round(o / grid) * grid
            if k not in top or p > top[k][2]:
                top[k] = (k, d, p)
        yin = sorted(top.values())

    def q(x):
        return round(x / grid) * grid

    xuanlv = []
    for i, (o, d, p) in enumerate(yin):
        nxt = q(yin[i + 1][0]) if i + 1 < len(yin) else None
        dur = q(d) if q(d) > 0 else grid
        if nxt is not None and nxt - q(o) - dur <= grid * 1.01:
            dur = nxt - q(o)
        xuanlv.append((q(o), p, dur))
    return xuanlv


def harmony_from_voices(tracks, tune_melody, *, resolution: float = 1.0,
                        length: float = None, transpose: int = 0,
                        key_pc: int = None) -> dict:
    """四个声部一起标和弦 {beat: (root, quality, bass)}, 高音也算 (常常是三音)。tune_melody 没用到。"""
    quanbu = [n for v in tracks for n in v]
    end = length or max(o + d for o, d, _ in quanbu)
    gezi, prev = {}, None
    t = 0.0
    while t < end - 1e-6:
        w, bass = defaultdict(float), (None, 999)
        for o, d, p in quanbu:
            ov = min(o + d, t + resolution) - max(o, t)
            if ov <= 1e-9:
                continue
            # 格子开头就在响的音比后来的 (多半是经过音) 重
            w[p % 12] += ov * (3.0 if o <= t + 1e-6 else 0.4)
            if o <= t + 1e-6 < o + d and p < bass[1]:
                bass = (p % 12, p)
        ch = _sibu_hexian(w, bass[0], key_pc) or prev
        if transpose and ch is not None:
            ch = Chord((ch.root + transpose) % 12, ch.quality,
                       (ch.bass + transpose) % 12)
        if ch is not None and (prev is None or ch.label() != prev.label()):
            gezi[round(t, 3)] = ch
        prev = ch
        t += resolution
    return {b: (_pc_name(c.root), c.quality, _pc_name(c.bass))
            for b, c in gezi.items()}


# 四声部写法用的和弦; add9、6/9 这类多半是把经过音当成了和声
_SIBU = {"": 1.0, "m": 1.0, "7": 0.97, "m7": 0.93, "dim": 0.9, "dim7": 0.9,
         "m7b5": 0.88, "maj7": 0.86, "sus4": 0.84}


def _sibu_hexian(quanzhong, bass_pc, key_pc=None):
    total = sum(quanzhong.values())
    if total <= 0:
        return None
    best, best_score = None, float("-inf")
    for root in range(12):
        for quality, pref in _SIBU.items():
            if quality not in QUALITIES:
                continue
            hexianyin = {(root + i) % 12 for i in QUALITIES[quality]}
            fugai = sum(v for pc, v in quanzhong.items() if pc in hexianyin)
            queshao = sum(1 for x in hexianyin if quanzhong.get(x, 0.0) <= 0.0)
            defen = (fugai / total) * pref - 0.45 * (1 - fugai / total)
            defen -= 0.04 * queshao + 0.03 * len(hexianyin)
            if bass_pc is not None:
                defen += 0.09 if bass_pc == root else (0.03 if bass_pc in hexianyin else -0.2)
            if key_pc is not None:
                deg = (root - key_pc) % 12
                defen += {0: 0.05, 5: 0.04, 7: 0.05, 2: 0.02, 9: 0.02}.get(deg, 0.0)
            if defen > best_score:
                best_score, best = defen, (root, quality)
    root, quality = best
    return Chord(root, quality, bass_pc if bass_pc is not None else root)


_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def _pc_name(pc: int) -> str:
    return _NAMES[pc % 12]


def chart(harmony: dict, beats_per_bar: float, bars: int) -> str:
    """按小节列和弦, 跟印刷谱对着看。"""
    hang = []
    for b in range(bars):
        ge = []
        for t, (r, q, bs) in sorted(harmony.items()):
            if b * beats_per_bar - 1e-6 <= t < (b + 1) * beats_per_bar - 1e-6:
                lab = r + q + ("/" + bs if bs != r else "")
                ge.append(f"{t - b * beats_per_bar:g}:{lab}")
        hang.append(f"{b + 1:3d} | " + "  ".join(ge))
    return "\n".join(hang)
