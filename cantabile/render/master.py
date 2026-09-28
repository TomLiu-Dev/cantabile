"""母带处理: EQ、压缩、电平、淡入淡出, 都用 ffmpeg。

注意 atrim 后面接 asetpts=N/SR/TB 会把时间轴归零, 之后的淡出要用片段里的时间。
剪过的输出都会量一下, 是静音就抛 SilentOutput, 不会悄悄返回一个空文件。
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["MasterChain", "WARM_PIANO", "NEUTRAL", "master", "loudness",
           "excerpt", "decode", "SilentOutput"]


class SilentOutput(RuntimeError):
    pass


@dataclass
class MasterChain:
    """钢琴母带链, 按处理顺序:

    highpass (Hz), low_shelf (Hz, dB), mid_cut (Hz, Q, dB), treble (Hz, dB),
    comp (阈值dB, 比率, 起音ms, 释放ms, knee), gain_db (压缩后补偿),
    room_wet (房间声, 在压缩和限幅之间, 0 关), target_db (最后校正到的平均电平),
    limit (限幅, 线性, 最后一步)
    """
    highpass: float = 32
    # 采样库比参考录音暗、低频多, 稍微切一点
    low_shelf: tuple = (130, -1.5)
    mid_cut: tuple = (380, 1.2, -0.6)
    # 调到频谱重心和录音差不多 (663 Hz 对 673 Hz)
    treble: tuple = (2600, 2.0)
    # 轻压缩: 只柔化句子里的起音, 不压平段与段之间的强弱
    comp: tuple = (-18, 1.4, 60, 400, 9)
    gain_db: float = 0.0
    # 混响放在压缩后面、限幅前面, 压缩混响会把衰减压平
    room_wet: float = 0.0
    target_db: float = -23.0     # 真实录音量出来的电平
    limit: float = 0.97

    def filters(self, stage: str = "all"):
        """ffmpeg 滤镜列表。stage: "pre" 房间之前, "post" 增益和限幅, "all" 全部。"""
        if stage == "post":
            lvjing = []
            if self.gain_db:
                lvjing.append(f"volume={self.gain_db:g}dB")
            if self.limit:
                lvjing.append(f"alimiter=limit={self.limit:g}:level=disabled")
            return lvjing
        lvjing = []
        if self.highpass:
            lvjing.append(f"highpass=f={self.highpass:g}")
        if self.low_shelf:
            f, g = self.low_shelf
            if g:
                lvjing.append(f"bass=g={g:g}:f={f:g}:width_type=q:w=0.7")
        if self.mid_cut:
            f, q, g = self.mid_cut
            if g:
                lvjing.append(f"equalizer=f={f:g}:width_type=q:w={q:g}:g={g:g}")
        if self.treble:
            f, g = self.treble
            if g:
                lvjing.append(f"treble=g={g:g}:f={f:g}:width_type=q:w=0.7")
        if self.comp:
            yuzhi, bilv, qiyin, shifang, knee = self.comp
            # acompressor 的阈值是线性的, 不是 dB
            xianxing = max(0.000977, min(1.0, 10.0 ** (float(yuzhi) / 20.0)))
            lvjing.append(f"acompressor=threshold={xianxing:.6f}:ratio={bilv:g}"
                          f":attack={qiyin:g}:release={shifang:g}"
                          f":knee={max(1.0, min(8.0, float(knee))):g}:makeup=1")
        if stage == "pre":
            return lvjing
        if self.gain_db:
            lvjing.append(f"volume={self.gain_db:g}dB")
        if self.limit:
            lvjing.append(f"alimiter=limit={self.limit:g}:level=disabled")
        return lvjing


WARM_PIANO = MasterChain()
NEUTRAL = MasterChain(highpass=24, low_shelf=(150, 0.0), mid_cut=(380, 1.2, 0.0),
                      treble=(5500, 0.0), comp=(-20, 1.6, 20, 400, 3),
                      gain_db=6.0, limit=0.99)


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg is not on PATH")
    return exe


def decode(path, sr=48000):
    """ffmpeg 能读的都解成单声道 float32。"""
    cmd = [_ffmpeg(), "-v", "error", "-i", str(path),
           "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sr), "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"could not decode {path}:\n"
                           f"{proc.stderr.decode('utf-8', 'replace')[-1500:]}")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0, sr


def loudness(path, window=1.0, sr=48000) -> dict:
    """量电平。profile 是每 window 秒的 RMS (dBFS), 只看平均值会漏掉中间静音的地方。"""
    data, sr = decode(path, sr)
    if data.size == 0:
        return {"mean_db": -np.inf, "max_db": -np.inf, "peak": 0.0,
                "duration": 0.0, "window": window, "profile": []}

    n = max(1, int(round(window * sr)))
    keyong = (data.size // n) * n
    quxian = []
    if keyong:
        kuai = data[:keyong].reshape(-1, n).astype(np.float64)
        rms = np.sqrt((kuai ** 2).mean(axis=1)) + 1e-12
        quxian = [round(float(v), 2) for v in 20.0 * np.log10(rms)]
    if data.size > keyong:                      # 最后不满的一段也算
        weiba = data[keyong:].astype(np.float64)
        quxian.append(round(float(20.0 * np.log10(np.sqrt((weiba ** 2).mean()) + 1e-12)), 2))

    zong = np.sqrt((data.astype(np.float64) ** 2).mean()) + 1e-12
    fengzhi = float(np.abs(data).max())
    return {"mean_db": round(float(20.0 * np.log10(zong)), 2),
            "max_db": round(max(quxian) if quxian else -np.inf, 2),
            "peak": round(fengzhi, 5),
            "peak_db": round(float(20.0 * np.log10(fengzhi + 1e-12)), 2),
            "duration": round(data.size / float(sr), 3),
            "window": window,
            "profile": quxian}


def _duration(path):
    exe = shutil.which("ffprobe")
    if exe:
        proc = subprocess.run([exe, "-v", "error", "-show_entries",
                               "format=duration", "-of", "json", str(path)],
                              capture_output=True, text=True)
        if proc.returncode == 0:
            try:
                return float(json.loads(proc.stdout)["format"]["duration"])
            except (KeyError, ValueError, TypeError):
                pass
    data, sr = decode(path)
    return data.size / float(sr)


def _verify(path, min_db=-70.0, window=1.0):
    info = loudness(path, window=window)
    if not info["profile"] or info["mean_db"] < min_db:
        raise SilentOutput(f"{path} is silent (mean {info['mean_db']} dBFS)")
    return info


def _codec_args(out_path, bitrate):
    houzhui = Path(out_path).suffix.lower()
    if houzhui == ".wav":
        return ["-c:a", "pcm_s16le"]
    if houzhui == ".flac":
        return ["-c:a", "flac"]
    if houzhui in (".m4a", ".aac"):
        return ["-c:a", "aac", "-b:a", str(bitrate or "256k")]
    b = str(bitrate or "q2")
    if b.startswith("q"):                        # VBR: 'q2' -> -q:a 2
        return ["-c:a", "libmp3lame", "-q:a", b[1:]]
    return ["-c:a", "libmp3lame", "-b:a", b]


def _run(args, what):
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed:\n{' '.join(args)}\n"
                           f"{proc.stderr[-2500:]}")


def _trim_filters(trim, src_duration):
    """返回 (滤镜, 片段时长)。trim 是 (开始, 时长) 或者只给开始。

    atrim + asetpts 之后片段从 0 开始, 后面的滤镜要用片段时间, 所以把片段时长也返回。
    """
    if trim is None:
        return [], src_duration
    if isinstance(trim, (int, float)):
        kaishi, shichang = float(trim), None
    else:
        t = tuple(trim)
        kaishi = float(t[0])
        shichang = float(t[1]) if len(t) > 1 and t[1] is not None else None

    kaishi = max(0.0, kaishi)
    shengxia = max(0.0, src_duration - kaishi)
    pianduan = shengxia if shichang is None else max(0.0, min(shichang, shengxia))
    if pianduan <= 0:
        raise ValueError(f"trim {trim!r} selects nothing from a "
                         f"{src_duration:.2f}s file")

    f = [f"atrim=start={kaishi:g}" + ("" if shichang is None else f":duration={pianduan:g}"),
         "asetpts=N/SR/TB"]
    return f, pianduan


def _fade_filters(clip_duration, fade_in, fade_out):
    lvjing = []
    if fade_in and fade_in > 0:
        lvjing.append(f"afade=t=in:st=0:d={min(fade_in, clip_duration):g}")
    if fade_out and fade_out > 0:
        d = min(float(fade_out), clip_duration)
        st = max(0.0, clip_duration - d)
        lvjing.append(f"afade=t=out:st={st:g}:d={d:g}")
    return lvjing


def _measured_gain(path, target_db):
    """要加多少增益平均电平才到 target_db。"""
    shuchu = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-af",
                             "volumedetect", "-f", "null", "-"],
                            capture_output=True, text=True).stderr
    import re
    m = re.search(r"mean_volume:\s*(-?[\d.]+) dB", shuchu)
    return (target_db - float(m.group(1))) if m else 0.0


def _through_room(in_wav: Path, chain) -> Path:
    """先 EQ 和压缩, 再过房间; 限幅在 master() 里最后做。"""
    import numpy as np, wave
    from .room import apply as add_room, RoomConfig
    pre = in_wav.with_name(in_wav.stem + ".pre.wav")
    _run([_ffmpeg(), "-y", "-v", "error", "-i", str(in_wav), "-af",
          ",".join(chain.filters("pre")), "-ac", "1", "-ar", "44100",
          "-c:a", "pcm_f32le", str(pre)], "pre")
    raw = subprocess.run([_ffmpeg(), "-v", "error", "-i", str(pre), "-f", "f32le",
                          "-ac", "1", "-ar", "44100", "-"],
                         capture_output=True).stdout
    pre.unlink(missing_ok=True)
    y = np.frombuffer(raw, dtype="<f4").astype(np.float64)
    z = add_room(y, 44100, RoomConfig(wet=chain.room_wet))
    fengzhi = float(np.abs(z).max()) or 1.0
    if fengzhi > 0.99:
        z *= 0.99 / fengzhi
    shuchu = in_wav.with_name(in_wav.stem + ".room.wav")
    with wave.open(str(shuchu), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
        w.writeframes((z * 32767).astype("<i2").tobytes())
    return shuchu


def master(in_wav, out_path, chain=WARM_PIANO, trim=None, fade_out=5.0,
           fade_in=0.0, bitrate="q2", tags=None, verify=True) -> Path:
    """母带处理 in_wav -> out_path。

    trim 是 (开始秒, 时长); fade_out 是对剪好的片段结尾; bitrate 可以是 'q2' (VBR)
    或 '192k'; 格式看扩展名; tags 写进元数据。verify 时是静音就抛 SilentOutput。
    """
    in_wav = Path(in_wav)
    if not in_wav.is_file():
        raise FileNotFoundError(in_wav)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    yuan = _duration(in_wav)
    trim_f, pianduan = _trim_filters(trim, yuan)
    if getattr(chain, "room_wet", 0.0) > 0:
        in_wav = _through_room(in_wav, chain)
        lvjing = trim_f + chain.filters("post") + _fade_filters(pianduan, fade_in, fade_out)
    else:
        lvjing = trim_f + list(chain.filters()) + _fade_filters(pianduan, fade_in, fade_out)

    args = [_ffmpeg(), "-y", "-v", "error", "-i", str(in_wav)]
    if lvjing:
        args += ["-af", ",".join(lvjing)]
    args += _codec_args(out_path, bitrate)
    for k, v in (tags or {}).items():
        args += ["-metadata", f"{k}={v}"]
    args.append(str(out_path))

    _run(args, "master")

    # 量一下结果再校正一次电平, 不靠固定的补偿增益
    mubiao = getattr(chain, "target_db", None)
    if mubiao is not None:
        tiaozheng = _measured_gain(out_path, mubiao)
        if abs(tiaozheng) > 0.3:
            tmp = out_path.with_suffix(".lvl" + out_path.suffix)
            a2 = [_ffmpeg(), "-y", "-v", "error", "-i", str(out_path),
                  "-af", f"volume={tiaozheng:.2f}dB"] + _codec_args(out_path, bitrate)
            for k, v in (tags or {}).items():
                a2 += ["-metadata", f"{k}={v}"]
            a2.append(str(tmp))
            _run(a2, "level")
            tmp.replace(out_path)
    if verify:
        _verify(out_path)
    return out_path


def excerpt(in_wav, out_path, start, dur, chain=WARM_PIANO,
            fade_in=0.35, fade_out=1.2, bitrate="q2", tags=None) -> Path:
    """从 start 开始剪 dur 秒再做母带。每秒的 RMS 都查一遍, 有声音的太少就抛 SilentOutput。"""
    shuchu = master(in_wav, out_path, chain=chain, trim=(start, dur),
                    fade_out=fade_out, fade_in=fade_in, bitrate=bitrate,
                    tags=tags, verify=False)

    info = loudness(shuchu, window=1.0)
    youshengyin = [db for db in info["profile"] if db > -60.0]
    if info["duration"] <= 0.05:
        raise SilentOutput(f"excerpt {shuchu} came out empty ({info['duration']}s)")
    if not youshengyin:
        raise SilentOutput(
            f"excerpt {shuchu} is silent (mean {info['mean_db']} dBFS over "
            f"{info['duration']}s) - the classic cause is a fade scheduled in "
            f"source time after asetpts reset the clip to zero")
    if len(youshengyin) < max(1, int(0.5 * len(info["profile"]))):
        raise SilentOutput(
            f"excerpt {shuchu}: only {len(youshengyin)} of {len(info['profile'])} seconds "
            f"carry signal; profile={info['profile']}")
    return shuchu
