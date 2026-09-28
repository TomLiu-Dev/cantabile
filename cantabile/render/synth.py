"""Performance -> MIDI 文件 -> fluidsynth 出音频。

速度曲线是连续的, MIDI 只能存阶跃, 所以每拍取一次样。踏板写成 CC64。
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np

from ..model import Performance

__all__ = ["to_midi", "SynthConfig", "render", "audible_end", "read_wav"]

_MAX_BPM_STEP = 0.25        # 拍, 再细只会让文件变大


def to_midi(perf: Performance, path, program=0, tpb=480, channel=0,
            beats_per_bar=4.0, tempo_step=1.0) -> Path:
    """写成 MIDI 文件。速度每 tempo_step 拍一个 set_tempo, 都落在拍上, 所以 tick 就是 拍 * tpb。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    changdu = perf.length
    if changdu <= 0:
        raise ValueError("performance has no notes")

    mid = mido.MidiFile(type=1, ticks_per_beat=tpb)

    # 速度轨
    sudu_gui = mido.MidiTrack()
    sudu_gui.append(mido.MetaMessage("track_name", name=perf.meta.get("title", "cantabile"),
                                     time=0))
    num = int(round(beats_per_bar)) if beats_per_bar == int(beats_per_bar) else 4
    sudu_gui.append(mido.MetaMessage("time_signature", numerator=num, denominator=4, time=0))

    bu = max(_MAX_BPM_STEP, float(tempo_step))
    shijian, pai, shangci = [], 0.0, None
    while pai <= changdu + bu:
        bpm = perf.tempo.bpm(pai)
        usec = int(round(60_000_000.0 / max(1e-6, bpm)))
        if usec != shangci:                    # 速度没变就不写
            shijian.append((pai, usec))
            shangci = usec
        pai += bu
    if not shijian:
        shijian = [(0.0, int(round(60_000_000.0 / perf.tempo.bpm(0.0))))]
    if shijian[0][0] > 0:
        shijian.insert(0, (0.0, shijian[0][1]))

    prev = 0
    for pai, usec in shijian:
        tick = int(round(pai * tpb))
        sudu_gui.append(mido.MetaMessage("set_tempo", tempo=usec, time=tick - prev))
        prev = tick
    mid.tracks.append(sudu_gui)

    # 音符轨
    yinfu_gui = mido.MidiTrack()
    yinfu_gui.append(mido.Message("program_change", channel=channel,
                                  program=int(program), time=0))

    # 同一时刻: 踏板、note_off 在 note_on 前面, 重弹的音先松开再按
    raw = []
    for pai, zhi in sorted(perf.pedal):
        raw.append((int(round(pai * tpb)), 0,
                    mido.Message("control_change", channel=channel, control=64,
                                 value=int(max(0, min(127, zhi))), time=0)))
    for n in perf.notes:
        on = int(round(n.onset * tpb))
        off = max(on + 1, int(round(n.end * tpb)))
        lidu = int(max(1, min(127, round(n.vel))))
        raw.append((off, 1, mido.Message("note_off", channel=channel,
                                         note=int(n.pitch), velocity=0, time=0)))
        raw.append((on, 2, mido.Message("note_on", channel=channel,
                                        note=int(n.pitch), velocity=lidu, time=0)))
    raw.sort(key=lambda t: (t[0], t[1]))

    prev = 0
    for tick, _, msg in raw:
        msg.time = tick - prev
        prev = tick
        yinfu_gui.append(msg)
    yinfu_gui.append(mido.MetaMessage("end_of_track", time=tpb))
    mid.tracks.append(yinfu_gui)

    mid.save(str(path))
    return path


@dataclass
class SynthConfig:
    """fluidsynth 的设置。polyphony 给得很大: 复音数不够时它会偷最老的音, 一般是踩着踏板的低音。"""
    soundfont: str
    sr: int = 48000
    gain: float = 0.5
    polyphony: int = 1024
    reverb: tuple = (0.72, 0.52, 0.55, 0.40)   # 房间大小, 阻尼, 宽度, 电平
    chorus: bool = False
    tail: float = 4.0                          # 保留几秒尾音

    def options(self):
        daxiao, zuni, kuandu, dianping = self.reverb
        xuanxiang = [
            "-o", f"synth.polyphony={int(self.polyphony)}",
            "-o", "synth.reverb.active=" + ("yes" if dianping > 0 else "no"),
            "-o", f"synth.reverb.room-size={daxiao}",
            "-o", f"synth.reverb.damp={zuni}",
            "-o", f"synth.reverb.width={kuandu}",
            "-o", f"synth.reverb.level={dianping}",
            "-o", "synth.chorus.active=" + ("yes" if self.chorus else "no"),
        ]
        return xuanxiang


def render(perf, cfg: SynthConfig, out_wav) -> Path:
    """Performance 或者 .mid 路径 -> WAV。没装 fluidsynth、没音色库、出来是静音, 都直接报错。"""
    if shutil.which("fluidsynth") is None:
        raise RuntimeError("fluidsynth is not on PATH")
    sf = Path(cfg.soundfont)
    if not sf.is_file():
        raise FileNotFoundError(f"soundfont not found: {sf}")

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    linshi = None
    if isinstance(perf, Performance):
        linshi = tempfile.mkdtemp(prefix="cantabile-synth-")
        midi_lujing = to_midi(perf, Path(linshi) / "perf.mid")
    else:
        midi_lujing = Path(perf)
        if not midi_lujing.is_file():
            raise FileNotFoundError(midi_lujing)

    cmd = (["fluidsynth", "-ni", "-F", str(out_wav),
            "-r", str(int(cfg.sr)), "-g", str(cfg.gain)]
           + cfg.options() + [str(sf), str(midi_lujing)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if linshi:
        shutil.rmtree(linshi, ignore_errors=True)
    if proc.returncode != 0 or not out_wav.is_file():
        raise RuntimeError(f"fluidsynth failed ({proc.returncode}):\n"
                           f"{proc.stderr[-2000:]}")

    data, sr = read_wav(out_wav)
    if data.size == 0 or float(np.abs(data).max()) < 1e-4:
        raise RuntimeError(f"fluidsynth produced a silent file: {out_wav}. "
                           "Check the soundfont has a bank 0 preset for the "
                           "program this performance asks for.")
    return out_wav


def read_wav(path):
    """返回 (采样, sr), float32, -1..1, 立体声混成单声道。"""
    from scipy.io import wavfile
    sr, data = wavfile.read(str(path))
    data = np.asarray(data)
    if data.dtype.kind == "i":
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    elif data.dtype.kind == "u":
        info = np.iinfo(data.dtype)
        data = (data.astype(np.float32) - info.max / 2.0) / (info.max / 2.0)
    else:
        data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def audible_end(wav, threshold=12, block=0.05, floor_pct=5.0) -> float:
    """几秒以后就听不见了。threshold 是比本身底噪高多少 dB, 用来剪掉 fluidsynth 结尾的静音。"""
    data, sr = read_wav(wav)
    if data.size == 0:
        return 0.0
    n = max(1, int(round(block * sr)))
    keyong = (data.size // n) * n
    if keyong == 0:
        return data.size / float(sr)
    kuai = data[:keyong].reshape(-1, n)
    rms = np.sqrt((kuai.astype(np.float64) ** 2).mean(axis=1)) + 1e-12
    db = 20.0 * np.log10(rms)
    dizao = float(np.percentile(db, floor_pct))
    xiang = np.flatnonzero(db > dizao + threshold)
    if xiang.size == 0:
        return data.size / float(sr)
    return float((xiang[-1] + 1) * n) / sr
