"""把一个 MIDI 文件弹得像真人。

谱子上写的音一个都不改，只加演奏：旋律突出、力度起伏、踏板、
速度的呼吸、按键先后的时间差。

用法:
    python -m cantabile play mozart.mid -o mozart.mp3
    python -m cantabile play mozart.mid -o mozart.mp3 --sfz 钢琴采样.sfz

    from cantabile.play import play
    play("mozart.mid", "mozart.mp3")
"""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import mido

from .model import Note, Performance, Voice
from .piece import default_tempo


# 默认去哪里找钢琴采样, 也可以用环境变量 CANTABILE_SFZ 指定
SFZ_MOREN = os.environ.get("CANTABILE_SFZ", "")


def du_midi(lujing):
    """读 MIDI, 返回 (每个音轨的音符, 速度bpm, 每小节几拍, 是不是复拍子)。

    音符是 (起始拍, 时长拍, 音高, 力度), 拍子按四分音符算。
    """
    mid = mido.MidiFile(str(lujing))
    tpb = mid.ticks_per_beat or 480
    yingui = []
    bpm = None
    fenzi, fenmu = 4, 4
    for tr in mid.tracks:
        t = 0
        an_xia = {}
        yinfu = []
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo" and bpm is None:
                bpm = mido.tempo2bpm(msg.tempo)
            elif msg.type == "time_signature" and t == 0:
                fenzi, fenmu = msg.numerator, msg.denominator
            elif msg.type == "note_on" and msg.velocity > 0:
                an_xia[(msg.channel, msg.note)] = (t, msg.velocity)
            elif msg.type in ("note_off", "note_on") and (msg.channel, msg.note) in an_xia:
                t0, v = an_xia.pop((msg.channel, msg.note))
                if t > t0:
                    yinfu.append((t0 / tpb, (t - t0) / tpb, msg.note, v))
        if yinfu:
            yingui.append((tr.name or "", sorted(yinfu)))
    meixiaojie = fenzi * 4.0 / fenmu
    fupaizi = fenmu == 8 and fenzi % 3 == 0
    return yingui, (bpm or 100.0), meixiaojie, fupaizi


def qu_chong(yinfu):
    """同一时刻同一个音只留一个 (合奏谱里经常有两个声部弹同一个音)。"""
    zuihao = {}
    for o, d, p, v in yinfu:
        k = (round(o, 3), p)
        if k not in zuihao or v > zuihao[k][3] or d > zuihao[k][1]:
            zuihao[k] = (o, max(d, zuihao.get(k, (0, 0))[1]), p, max(v, zuihao.get(k, (0, 0, 0, 0))[3]))
    return sorted(zuihao.values())


def zhao_xuanlv(yingui):
    """音轨名有 solo / melody 就用它; 不然取最高那个音轨每个时刻的最高音。"""
    for ming, yinfu in yingui:
        if any(k in ming.lower() for k in ("solo", "melody", "melodie")):
            return set((round(o, 3), p) for o, d, p, v in yinfu)
    zui_gao = max(yingui, key=lambda x: sum(n[2] for n in x[1]) / len(x[1]))[1]
    ding = {}
    for o, d, p, v in zui_gao:
        k = round(o, 3)
        if k not in ding or p > ding[k]:
            ding[k] = p
    return set(ding.items())


def zuo_yanzou(lujing, bpm=None, seed=0):
    """把 MIDI 文件变成一次演奏 (Performance), 还没有渲染成声音。"""
    from .analyze.chords import infer
    from .perform import (voice_balance, phrase_arc, accents, clamp,
                          duck_under_sustain, harmonic_pedal, humanize)
    from .perform.timing import NATURAL
    from .perform.rules import PALETTES
    from .perform.dispersion import (motor_noise, intentional,
                                     match_distribution, REFERENCE_PROFILE)

    yingui, wenjian_bpm, meixiaojie, fupaizi = du_midi(lujing)
    if not yingui:
        raise ValueError(f"{lujing} 里面没有音符")
    if bpm is None and abs(wenjian_bpm - 60) < 0.01:
        # 60 一般是导出软件没写速度时的默认值
        print("MIDI 里没有写速度, 先用 100, 可以用 --bpm 改")
        wenjian_bpm = 100.0
    xuanlv = zhao_xuanlv(yingui)
    quanbu = qu_chong([n for _, ns in yingui for n in ns])
    changdu = max(o + d for o, d, p, v in quanbu)

    perf = Performance(tempo=default_tempo(bpm or wenjian_bpm, changdu,
                                           beats_per_bar=meixiaojie))
    # 每个时刻: 旋律音标 MELODY, 剩下最低的标 BASS, 其他是 INNER
    tongshi = defaultdict(list)
    for n in quanbu:
        tongshi[round(n[0], 3)].append(n)
    for k, zu in tongshi.items():
        feixuanlv = [n for n in zu if (k, n[2]) not in xuanlv]
        zuidi = min(n[2] for n in feixuanlv) if feixuanlv else None
        for o, d, p, v in zu:
            if (k, p) in xuanlv:
                shengbu, biaoqian = Voice.MELODY, "melody"
            elif p == zuidi:
                shengbu, biaoqian = Voice.BASS, "bass"
            else:
                shengbu, biaoqian = Voice.INNER, "inner"
            perf.add(Note(o, d, p, v, shengbu, biaoqian))
    perf.sorted()

    # 和声只用来决定什么时候换踏板
    hesheng = infer([(n.onset, n.dur, n.pitch, n.vel) for n in perf.notes],
                    resolution=meixiaojie / 2 if meixiaojie >= 4 else meixiaojie,
                    beats_per_bar=int(meixiaojie))
    perf.meta["harmony"] = hesheng

    perf = clamp(accents(phrase_arc(voice_balance(perf, melody_gain=6,
                                                  inner_cut=2, colour_cut=4)),
                         meter=meixiaojie, compound=fupaizi))
    perf = PALETTES["accompaniment"].apply(perf)
    # 长音下面伴奏让一让, 但是一个音都不删 (thin_after 设成很大)
    perf = clamp(duck_under_sustain(perf, grace=0.3, max_cut=18, rate=12.0,
                                    thin_after=1e9))
    perf = harmonic_pedal(perf, hesheng)
    perf = intentional(motor_noise(perf, seed=seed), hesheng,
                       beats_per_bar=meixiaojie)
    if REFERENCE_PROFILE:
        perf = match_distribution(perf, REFERENCE_PROFILE, strength=0.5)
    return humanize(perf, NATURAL, seed=seed)


def xuanran(perf, shuchu, sfz=None, gongming=0.08):
    """把演奏渲染成 mp3/wav。没有钢琴采样就只输出 MIDI。"""
    import wave
    import numpy as np
    shuchu = Path(shuchu)
    if not sfz and not SFZ_MOREN:
        from .xiazai import zhao_sfz
        sfz = zhao_sfz()
    sfz = sfz or SFZ_MOREN
    if not sfz:
        from .render import to_midi
        mid = shuchu.with_suffix(".mid")
        to_midi(perf, mid)
        print(f"没有钢琴采样, 只写了 {mid}。先运行: python -m cantabile download-piano")
        return mid
    from .render.sfz import SFZSampler
    from .render import master, audible_end
    y = SFZSampler(sfz).render(perf)
    if gongming > 0:
        from .render.resonance import render as add_res, ResonanceConfig, events_from
        y = add_res(np.asarray(y, dtype=np.float64), events_from(perf), 44100,
                    ResonanceConfig(gain=gongming, max_voices=260))
        y = y / max(1e-9, np.abs(y).max()) * 0.9
    wav = shuchu.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes((np.asarray(y) * 32767).astype("<i2").tobytes())
    if shuchu.suffix.lower() != ".wav":
        master(wav, shuchu, trim=(0.0, audible_end(wav) + 2))
        wav.unlink()
    return shuchu


def play(lujing, shuchu=None, sfz=None, bpm=None, seed=0):
    """一步到位: MIDI 进去, 像真人弹的音频出来。"""
    shuchu = shuchu or Path(lujing).with_suffix(".mp3")
    perf = zuo_yanzou(lujing, bpm=bpm, seed=seed)
    return xuanran(perf, shuchu, sfz=sfz)
