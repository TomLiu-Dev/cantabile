"""从 MIDI 读四声部谱。

MIDI 多半是演奏录的: 放键时间不等于音符时值 (踏板撑着), 最高音也不一定是
高音声部 (旋律长音时伴奏在下面重敲)。所以节奏按起音间隔算, 伴奏高度按
上一个完整和弦的第二高音算。
"""
from __future__ import annotations

import math
from collections import defaultdict
import mido

from ..model import TempoMap
from ..theory import Chord, Harmony, QUALITIES, name

__all__ = ["load", "extract_soprano", "extract_harmony", "quantize",
           "pedal_events", "bar_lines", "as_names"]


def load(path):
    """返回 (notes, ticks_per_beat, tempo_map), notes 是 (onset, dur, pitch, vel), dur 是原始放键时间。"""
    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat or 480

    notes = []
    dengdai = defaultdict(list)          # (channel, pitch) -> [(tick, vel), ...]
    sudu = []                            # (tick, usec_per_beat)
    tick = 0

    for msg in mido.merge_tracks(mid.tracks):
        tick += msg.time
        if msg.type == "set_tempo":
            sudu.append((tick, msg.tempo))
        elif msg.type == "note_on" and msg.velocity > 0:
            dengdai[(msg.channel, msg.note)].append((tick, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            q = dengdai.get((msg.channel, msg.note))
            if q:
                # 先进先出, 同音重叠时才配得对
                start, vel = q.pop(0)
                notes.append((start / tpb, max(tick - start, 1) / tpb, msg.note, vel))

    end = tick
    for (ch, pitch), q in dengdai.items():         # 没放开的音
        for start, vel in q:
            notes.append((start / tpb, max(end - start, 1) / tpb, pitch, vel))

    notes.sort(key=lambda n: (n[0], n[2]))
    return notes, tpb, _tempo_map(sudu, tpb)


def _tempo_map(sudu, tpb):
    if not sudu:
        return TempoMap([(0.0, 120.0)])
    maodian, yijian = [], set()
    for tick, usec in sorted(sudu):
        beat = round(tick / tpb, 6)
        bpm = 60_000_000.0 / usec
        if beat in yijian:                          # 同一拍上以最后一个为准
            maodian[-1] = (beat, bpm)
        else:
            maodian.append((beat, bpm))
            yijian.add(beat)
    if maodian[0][0] > 0:
        maodian.insert(0, (0.0, maodian[0][1]))
    return TempoMap(maodian)


def pedal_events(path, controller: int = 64):
    """某个控制器的 [(beat, value)], 默认 CC64。"""
    mid = mido.MidiFile(str(path))
    tpb = mid.ticks_per_beat or 480
    jieguo, tick = [], 0
    for msg in mido.merge_tracks(mid.tracks):
        tick += msg.time
        if msg.type == "control_change" and msg.control == controller:
            jieguo.append((tick / tpb, msg.value))
    return jieguo


def bar_lines(path):
    """(beats_per_bar, first_downbeat), 没拍号就当 4/4。"""
    mid = mido.MidiFile(str(path))
    for track in mid.tracks:
        for msg in track:
            if msg.type == "time_signature":
                return msg.numerator * 4.0 / msg.denominator, 0.0
    return 4.0, 0.0


def _by_onset(notes, tol=1e-6):
    qishi_zu = []
    for onset, dur, pitch, vel in sorted(notes, key=lambda n: (n[0], n[2])):
        if qishi_zu and onset - qishi_zu[-1][0] <= tol:
            qishi_zu[-1][1].append((pitch, dur, vel))
        else:
            qishi_zu.append((onset, [(pitch, dur, vel)]))
    return [(t, sorted(g)) for t, g in qishi_zu]


def extract_soprano(notes, min_chord=3, floor=None, spread=0.06,
                    max_fall=2, hold_tol=0.10, voiced=True):
    """四声部 (SATB) 的最高声部, [(beat, pitch, dur)]。时值按起音间隔算, 同音不合并。

    比当前高音低的音, 只有高音已经放开、而且够到伴奏上沿时才算旋律。
    max_fall: 单个音比高音低这么多半音就当左手过渡音。
    voiced: 完整和弦里最高音也最响时, 可以不管伴奏上沿。
    """
    qishi_zu = _by_onset(notes, tol=spread)
    if not qishi_zu:
        return []

    # 最后一个音拖到最后一拍结束
    quwei = math.ceil(max(o + d for o, d, p, v in notes) - 1e-6)

    xuanzhong = []            # (beat, pitch, gate)
    gaoyin = None             # 当前高音声部的音
    gaoyin_end = float("-inf")
    shangyan = float("-inf")  # 伴奏上沿

    for beat, group in qishi_zu:
        yingao = [p for p, d, v in group]
        top = yingao[-1]
        if floor is not None and top < floor:
            continue

        if gaoyin is None:
            take = True
        elif top >= gaoyin:
            take = True
        else:
            fangle = beat >= gaoyin_end
            dandu = len(group) <= 1 and gaoyin - top > max_fall
            take = fangle and not dandu and (
                top >= shangyan
                or (voiced and len(group) >= min_chord
                    and max(v for p, d, v in group) <= group[-1][2]))

        if not take:
            # 高音还按着时下面的薄音也会抬高上沿
            if len(group) < min_chord:
                shangyan = max(shangyan, top)
            continue

        gate = max(d for p, d, v in group if p == top)
        xuanzhong.append((beat, top, gate))
        gaoyin = top
        gaoyin_end = beat + gate - max(0.1, hold_tol * gate)

        if len(yingao) >= 2:
            dier = yingao[-2]
            if len(yingao) >= min_chord:
                shangyan = dier
            else:
                shangyan = max(shangyan, dier)

    jieguo = []
    for i, (beat, pitch, gate) in enumerate(xuanzhong):
        if i + 1 < len(xuanzhong):
            dur = xuanzhong[i + 1][0] - beat
        else:
            dur = max(gate, quwei - beat)
        jieguo.append((beat, pitch, dur))
    return jieguo


def as_names(melody, beats_per_bar=4.0, flats=False):
    """{小节号: ['G4', 'A4', ...]}, 肉眼核对用。"""
    xiaojie = defaultdict(list)
    for beat, pitch, dur in melody:
        xiaojie[int(beat // beats_per_bar) + 1].append(name(pitch, flats))
    return dict(sorted(xiaojie.items()))


def _soprano_spans(soprano):
    return [(b, b + d, p) for b, p, d in soprano]


def extract_harmony(notes, soprano, resolution=2.0, lo=None, hi=None) -> Harmony:
    """每 resolution 拍一个和弦。高音声部不算进去, 免得旋律里的经过音被当成和声。"""
    if not notes:
        raise ValueError("no notes to analyse")

    gaoyin_duan = _soprano_spans(soprano)
    end = max(o + d for o, d, p, v in notes)

    def is_soprano(onset, pitch):
        for s, e, p in gaoyin_duan:
            if p == pitch and s - 1e-6 <= onset < e - 1e-6:
                return True
        return False

    gezi = {}
    prev = None
    n_slots = max(1, int(round(end / resolution + 0.5)))
    for i in range(n_slots):
        t0 = i * resolution
        t1 = t0 + resolution
        quanzhong = defaultdict(float)
        bass_pc, bass_pitch = None, 128
        for onset, dur, pitch, vel in notes:
            ov = min(onset + dur, t1) - max(onset, t0)
            if ov <= 1e-9:
                continue
            if lo is not None and pitch < lo:
                continue
            if hi is not None and pitch > hi:
                continue
            if is_soprano(onset, pitch):
                continue
            quanzhong[pitch % 12] += ov * (vel / 100.0)
            if pitch < bass_pitch:
                bass_pitch, bass_pc = pitch, pitch % 12
        hexian = _best_chord(quanzhong, bass_pc)
        if hexian is None:
            hexian = prev if prev is not None else Chord(0, "")
        gezi[float(t0)] = hexian
        prev = hexian
    return Harmony(gezi)


# 分数差不多时简单和弦优先
_PREF = {"": 1.00, "m": 1.00, "7": 0.97, "m7": 0.97, "maj7": 0.95, "6": 0.93,
         "m6": 0.90, "sus4": 0.90, "sus2": 0.88, "dim": 0.90, "dim7": 0.90,
         "m7b5": 0.88, "aug": 0.82, "add9": 0.88, "madd9": 0.86, "7sus4": 0.88,
         "9": 0.86, "m9": 0.86, "maj9": 0.84, "6/9": 0.82}


def _best_chord(weight, bass_pc):
    total = sum(weight.values())
    if total <= 0:
        return None
    best, best_score = None, float("-inf")
    for root in range(12):
        for quality in QUALITIES:
            hexianyin = {(root + i) % 12 for i in QUALITIES[quality]}
            fugai = sum(w for pc, w in weight.items() if pc in hexianyin)
            queshao = sum(1 for t in hexianyin if weight.get(t, 0.0) <= 0.0)
            waimian = total - fugai
            defen = (fugai / total) * _PREF.get(quality, 0.85)
            defen -= 0.40 * (waimian / total)
            defen -= 0.03 * queshao
            defen -= 0.030 * len(hexianyin)
            if bass_pc is not None:
                if bass_pc == root:
                    defen += 0.09
                elif bass_pc in hexianyin:
                    defen += 0.03
                else:
                    defen -= 0.10
            if defen > best_score:
                best_score, best = defen, (root, quality)
    root, quality = best
    if bass_pc is not None and bass_pc != root:
        return Chord(root, quality, bass_pc)
    return Chord(root, quality)


def quantize(events, grid=0.25, min_dur=None):
    """起点和时值对齐到 grid; 时值最少一格 (或 min_dur)。三元组、四元组都行。"""
    if not events:
        return []
    floor_dur = grid if min_dur is None else min_dur

    def snap(x):
        return round(round(x / grid) * grid, 6)

    jieguo = []
    for ev in events:
        if len(ev) == 3:
            beat, pitch, dur = ev
            jieguo.append((snap(beat), pitch, max(floor_dur, snap(dur))))
        elif len(ev) == 4:
            onset, dur, pitch, vel = ev
            jieguo.append((snap(onset), max(floor_dur, snap(dur)), pitch, vel))
        else:
            raise ValueError(f"cannot quantise a {len(ev)}-tuple: {ev!r}")
    return jieguo
