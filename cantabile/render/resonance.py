"""共鸣: 琴弦之间的共振, 还有踩踏板时整台琴一起响。

采样器每个键都是单独录的, 没有其他弦的共鸣。这里把每根没被制音的弦当成一组共振器
(按非谐和的泛音频率), 用干声去激励, 再加回去。pedal_contrast 检查踩没踩踏板衰减听起来是不是不一样。

参考: Weinreich (1977) "Coupled piano strings", JASA 62(6);
Fletcher & Rossing, The Physics of Musical Instruments, ch. 12;
Lehtonen, Askenfelt & Valimaki (2009), JASA 126(2)。
"""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np


def f0_of(pitch: int) -> float:
    return 440.0 * 2.0 ** ((pitch - 69) / 12.0)


def inharmonicity(pitch: int) -> float:
    """非谐系数 B, f_n = n * f0 * sqrt(1 + B * n^2)。三角琴的典型曲线: 低音高, 中间最低, 最高八度又上去。"""
    if pitch < 40:
        return 2.2e-4 * (2.0 ** ((40 - pitch) / 12.0))
    if pitch < 76:
        return 2.2e-4 * (2.0 ** (-(pitch - 40) / 22.0))
    return 5.0e-5 * (2.0 ** ((pitch - 76) / 16.0))


def partials(pitch: int, n: int = 8, ceiling: float = 9000.0) -> list:
    f0, B = f0_of(pitch), inharmonicity(pitch)
    fanyin = []
    for k in range(1, n + 1):
        f = k * f0 * math.sqrt(1.0 + B * k * k)
        if f >= ceiling:
            break
        fanyin.append(f)
    return fanyin


def decay_time(pitch: int, partial: int = 1) -> float:
    """衰减到 -60 dB 的秒数, 低音弦响得久得多。"""
    jichu = 26.0 * math.exp(-(pitch - 21) / 34.0) + 1.2
    return max(0.25, jichu / (1.0 + 0.55 * (partial - 1)))


@dataclass
class ResonanceConfig:
    strings: tuple = (21, 96)     # 模拟哪些弦
    n_partials: int = 6
    gain: float = 0.16            # 音板把弦耦合得多强
    pedal_gain: float = 1.0       # 抬起制音器时额外的耦合
    damped_gain: float = 0.06     # 制音了的弦也还会响一点
    max_voices: int = 340
    detune_cents: float = 1.2     # 同音弦之间的音差, 产生拍频和双段衰减


def _resonator(x: np.ndarray, f: float, sr: int, t60: float,
               detune: float = 0.0) -> np.ndarray:
    """一个泛音 = 二阶共振器, 在 +/- detune 音分各跑一次, 像真的同音弦那样有拍。"""
    jieguo = np.zeros_like(x)
    for cents in (-detune, detune):
        fr = f * (2.0 ** (cents / 1200.0))
        if fr >= sr * 0.49 or fr <= 0:
            continue
        r = math.exp(-6.9078 / (t60 * sr))
        w = 2.0 * math.pi * fr / sr
        a1, a2 = -2.0 * r * math.cos(w), r * r
        b0 = (1.0 - r) * math.sin(w)
        y = np.zeros_like(x)
        z1 = z2 = 0.0
        for i in range(x.shape[0]):
            v = b0 * x[i] - a1 * z1 - a2 * z2
            y[i] = v
            z2, z1 = z1, v
        jieguo += y
    return jieguo * 0.5


def _resonator_fast(x, f, sr, t60, detune=0.0):
    """有 scipy 就用 lfilter, 没有就退回上面的循环。"""
    try:
        from scipy.signal import lfilter
    except Exception:
        return _resonator(x, f, sr, t60, detune)
    jieguo = np.zeros_like(x)
    for cents in (-detune, detune):
        fr = f * (2.0 ** (cents / 1200.0))
        if fr >= sr * 0.49 or fr <= 0:
            continue
        r = math.exp(-6.9078 / (max(t60, 1e-3) * sr))
        w = 2.0 * math.pi * fr / sr
        jieguo += lfilter([(1.0 - r) * math.sin(w)], [1.0, -2.0 * r * math.cos(w), r * r], x)
    return jieguo * 0.5


def undamped_at(beat_events, t: float) -> set:
    """t 时刻哪些弦没被制音 (踩着踏板, 或者键按着)。"""
    taban, anzhe = False, set()
    for kind, when, what in beat_events:
        if when > t:
            break
        if kind == "pedal":
            taban = bool(what)
        elif kind == "on":
            anzhe.add(what)
        elif kind == "off":
            anzhe.discard(what)
    return (set(range(21, 97)) if taban else set()) | anzhe


def render(dry: np.ndarray, events, sr: int,
           cfg: ResonanceConfig = None) -> np.ndarray:
    """给干声加上没制音的弦的共鸣。

    events 是按时间排好的 ("pedal"|"on"|"off", 秒, 值)。弦直接用干声激励,
    算是对琴马耦合的线性近似, 不是完整的音板模型。
    """
    cfg = cfg or ResonanceConfig()
    n = dry.shape[0]
    gongming = np.zeros(n, dtype=np.float64)
    lo, hi = cfg.strings
    # 每根弦按没被制音的时间比例加权
    wangge = np.linspace(0, n / sr, 64)
    quanzhong = {}
    for t in wangge:
        for p in undamped_at(events, t):
            quanzhong[p] = quanzhong.get(p, 0) + 1
    xuan = sorted(quanzhong, key=lambda p: -quanzhong[p])
    xuan = [p for p in xuan if lo <= p <= hi][: cfg.max_voices // cfg.n_partials]
    x = dry.astype(np.float64)
    for p in xuan:
        w = quanzhong[p] / len(wangge)
        for k, f in enumerate(partials(p, cfg.n_partials), start=1):
            gongming += _resonator_fast(x, f, sr, decay_time(p, k),
                                        cfg.detune_cents) * (w / k ** 1.4)
    # 故意不归一化: 没制音的弦越多共鸣越大, 要的就是这个
    return dry + cfg.gain * gongming / max(1.0, cfg.n_partials)


def events_from(perf) -> list:
    """从 Performance 取踏板和按键事件, 单位秒。"""
    ev = []
    for beat, down in perf.pedal:
        ev.append(("pedal", perf.tempo.seconds(max(0.0, beat - 0.03)), 0))
        ev.append(("pedal", perf.tempo.seconds(beat + 0.06), 1))
    for note in perf.notes:
        ev.append(("on", perf.tempo.seconds(note.onset), note.pitch))
        ev.append(("off", perf.tempo.seconds(note.end), note.pitch))
    ev.sort(key=lambda e: e[1])
    return ev


def pedal_contrast(render_fn, *, pitch: int = 60, sr: int = 44100) -> dict:
    """同一个音踩和不踩踏板的衰减比一比。频谱相关接近 1 说明踏板只是让音变长了, 真钢琴不是这样。"""
    up, down = render_fn(False), render_fn(True)
    n = min(len(up), len(down), int(1.0 * sr))
    def spec(y):
        S = np.abs(np.fft.rfft(y[:n] * np.hanning(n)))
        return S / max(S.max(), 1e-12)
    a, b = spec(up), spec(down)
    f = np.fft.rfftfreq(n, 1 / sr)
    m = (f > 50) & (f < 6000)
    return {"spectral_r": float(np.corrcoef(a[m], b[m])[0, 1]),
            "energy_ratio": float(np.abs(down[:n]).mean() /
                                  max(np.abs(up[:n]).mean(), 1e-12)),
            "partials_up": int(sum(1 for i in range(2, len(a) - 2)
                                   if a[i] > a[i-1] and a[i] > a[i+1] and a[i] > .02)),
            "partials_down": int(sum(1 for i in range(2, len(b) - 2)
                                     if b[i] > b[i-1] and b[i] > b[i+1] and b[i] > .02))}
