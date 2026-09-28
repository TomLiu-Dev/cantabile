"""单层采样的力度音色。

真钢琴弹得重不光是响, 还更亮: 从 pp 到 ff 频谱重心大概升 2-4 倍, 每键一个采样的
SoundFont 只有 1.1-1.2 倍。这里按力度分几段分别渲染, 每段加不同的高频倾斜, 再混起来。
"""
from __future__ import annotations
import math, shutil, subprocess, tempfile
from dataclasses import dataclass, field
from pathlib import Path

from ..model import Performance, Voice


def measure_velocity_timbre(soundfont, *, pitch: int = 60,
                            velocities=(30, 50, 70, 90, 110, 127),
                            sr: int = 44100, gap: float = 3.0) -> dict:
    """同一个音用不同力度渲染, 量频谱重心和 2k 以上占比随力度怎么变。"""
    import numpy as np
    from midiutil import MIDIFile
    linshi = Path(tempfile.mkdtemp())
    mid, wav = linshi / "v.mid", linshi / "v.wav"
    mf = MIDIFile(1)
    mf.addTempo(0, 0, 60)
    mf.addProgramChange(0, 0, 0, 0)
    for i, v in enumerate(velocities):
        mf.addNote(0, 0, pitch, i * gap, gap * 0.8, v)
    with open(mid, "wb") as fh:
        mf.writeFile(fh)
    subprocess.run(["fluidsynth", "-ni", "-F", str(wav), "-r", str(sr), "-g", "0.6",
                    "-o", "synth.reverb.active=no", "-o", "synth.chorus.active=no",
                    str(soundfont), str(mid)], check=True, capture_output=True)
    import wave
    with wave.open(str(wav)) as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        ch = w.getnchannels()
        y = raw.reshape(-1, ch).mean(axis=1) / 32768.0 if ch > 1 else raw / 32768.0
        sr = w.getframerate()
    cen, hi, amp = [], [], []
    for i, v in enumerate(velocities):
        duan = y[int((i * gap + .03) * sr):int((i * gap + .5) * sr)]
        if len(duan) < 256:
            continue
        S = np.abs(np.fft.rfft(duan * np.hanning(len(duan))))
        f = np.fft.rfftfreq(len(duan), 1 / sr)
        cen.append(float((S * f).sum() / max(S.sum(), 1e-12)))
        hi.append(float(S[f > 2000].sum() / max(S.sum(), 1e-12)))
        amp.append(float(np.abs(duan).max()))
    return {"velocities": list(velocities), "centroid_hz": cen,
            "above_2k_share": hi, "peak_amp": amp,
            "centroid_ratio": cen[-1] / cen[0] if cen and cen[0] else 0.0,
            "hf_ratio": hi[-1] / hi[0] if hi and hi[0] else 0.0,
            "amp_ratio": amp[-1] / amp[0] if amp and amp[0] else 0.0,
            "monotonic": all(b >= a * .95 for a, b in zip(cen, cen[1:]))}


REAL_PIANO = {"centroid_ratio": (2.0, 4.0), "hf_ratio": (6.0, 14.0)}


def verdict(m: dict) -> str:
    lo, hi = REAL_PIANO["centroid_ratio"]
    if m["centroid_ratio"] >= lo:
        return "expressive"
    if m["centroid_ratio"] >= 1.4:
        return "partial"
    return "loudness-only"


@dataclass
class Band:
    lo: int
    hi: int
    tilt_db: float          # 这一段的高架增益
    shelf_hz: float = 1800.0
    lp_hz: float = None     # 最轻那段可以加低通

    def contains(self, v: int) -> bool:
        return self.lo <= v <= self.hi


# pp 到 ff 分五段, 倾斜的差让重心比例落在真钢琴范围里 (calibrate 可以按音色库拟合)。
# 中心在 0 dB 以下, 整体不会变亮
DEFAULT_BANDS = [
    Band(1,   44, -11.0, 1500, lp_hz=3000),
    Band(45,  66,  -7.5, 1600),
    Band(67,  84,  -4.0, 1800),
    Band(85, 102,   0.0, 2000),
    Band(103, 127,  4.5, 2200),
]


def split_by_velocity(perf: Performance, bands=DEFAULT_BANDS) -> list:
    """每个力度段一个 Performance, 时间一样, 踏板共用。"""
    fenduan = []
    for b in bands:
        p = Performance([n for n in perf.notes if b.contains(n.vel)],
                        list(perf.pedal), perf.tempo, dict(perf.meta))
        fenduan.append((b, p))
    return fenduan


def _band_filter(b: Band) -> str:
    f = [f"treble=g={b.tilt_db}:f={b.shelf_hz}"]
    if b.lp_hz:
        f.append(f"lowpass=f={b.lp_hz}")
    # 弹得重起音也更尖, 响的几段加点临场感
    if b.tilt_db > 2:
        f.append("equalizer=f=3500:t=q:w=1.4:g=%.1f" % (b.tilt_db * 0.35))
    return ",".join(f)


def render_expressive(perf: Performance, cfg, out_wav, *, bands=DEFAULT_BANDS,
                      keep_parts: bool = False):
    """按力度段分别渲染、加倾斜, 再混起来。要 ffmpeg 和 fluidsynth。"""
    from .synth import render
    out_wav = Path(out_wav)
    linshi = Path(tempfile.mkdtemp())
    fenduan, lvjing = [], []
    for i, (b, p) in enumerate(split_by_velocity(perf, bands)):
        if not p.notes:
            continue
        w = linshi / f"band{i}.wav"
        render(p, cfg, w)
        fenduan.append(w)
        lvjing.append(f"[{len(fenduan)-1}:a]{_band_filter(b)}[b{len(fenduan)-1}]")
    if not fenduan:
        raise RuntimeError("nothing to render")
    args = ["ffmpeg", "-y", "-v", "error"]
    for w in fenduan:
        args += ["-i", str(w)]
    hunyin = "".join(f"[b{i}]" for i in range(len(fenduan)))
    chain = ";".join(lvjing) + f";{hunyin}amix=inputs={len(fenduan)}:normalize=0[out]"
    args += ["-filter_complex", chain, "-map", "[out]",
             "-c:a", "pcm_s24le", str(out_wav)]
    subprocess.run(args, check=True, capture_output=True)
    if keep_parts:
        return out_wav, fenduan
    shutil.rmtree(linshi, ignore_errors=True)
    return out_wav


def calibrate(soundfont, *, target=2.6, bands=None, spread=None) -> list:
    """拟合每段的倾斜, 让重心比例到 target, 返回新的 bands。"""
    import numpy as np
    jizhun = measure_velocity_timbre(soundfont)
    bands = list(bands or DEFAULT_BANDS)
    xianzai = max(jizhun["centroid_ratio"], 1.0)
    haicha = target / xianzai
    # g dB 的高架大概让重心乘 1 + g/24, 总量线性分到各段
    zong = 24.0 * (haicha - 1.0) if spread is None else spread
    n = len(bands)
    xin = []
    for i, b in enumerate(bands):
        t = (i / (n - 1)) - 0.5 if n > 1 else 0.0
        xin.append(Band(b.lo, b.hi, round(zong * t, 2), b.shelf_hz, b.lp_hz))
    return xin
