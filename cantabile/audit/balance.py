"""从音频看旋律是不是比伴奏响。

光看谱上的力度不够, 音区、踏板、重复都会影响最后多响。这里在每个在响的音的基频上读
FFT 幅度, 算 旋律dB - 最响的伴奏音dB。按频段比能量不行, 伴奏的泛音和踏板会把结果淹掉。
要 numpy 和 librosa。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from cantabile.model import Finding, Note, Performance, Report, Voice
from cantabile.theory import name

__all__ = [
    "melody_prominence",
    "spectral_profile",
    "reverb_tail",
    "load_audio",
    "synth_wav",
    "HELD_BEATS",
    "BAND_EDGES",
]

# 至少这么长算长音, 长音低于 0 dB 是 error
HELD_BEATS = 1.0

# spectral_profile 的分频点 (Hz), audit.reference 里的数就是按这个量的
BAND_EDGES = (250.0, 1200.0)

_EPS = 1e-12


def load_audio(wav_path: str, sr: int = 22050):
    import librosa
    y, sr_out = librosa.load(wav_path, sr=sr, mono=True)
    return np.asarray(y, dtype=np.float64), int(sr_out)


def _hz(pitch: int) -> float:
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def _db(a: float) -> float:
    return 20.0 * math.log10(max(float(a), _EPS))


class _Analyser:
    """短时频谱, 按采样位置缓存。"""

    def __init__(self, y, sr: int, win: float, nfft: int = 8192):
        self.y, self.sr = y, sr
        self.n = max(64, int(round(win * sr)))
        self.nfft = max(nfft, 1 << int(math.ceil(math.log2(max(self.n, 2)))))
        self.window = np.hanning(self.n)
        self.freqs = np.fft.rfftfreq(self.nfft, 1.0 / sr)
        self._cache: Dict[int, np.ndarray] = {}

    def spectrum(self, t: float) -> np.ndarray:
        kaishi = int(round(t * self.sr)) - self.n // 2
        kaishi = max(0, min(kaishi, max(0, len(self.y) - self.n)))
        if kaishi in self._cache:
            return self._cache[kaishi]
        duan = self.y[kaishi:kaishi + self.n]
        if len(duan) < self.n:
            duan = np.pad(duan, (0, self.n - len(duan)))
        pu = np.abs(np.fft.rfft(duan * self.window, self.nfft))
        self._cache[kaishi] = pu
        return pu

    def amp_at(self, spec: np.ndarray, f0: float,
               halfwidth_semitones: float = 0.5) -> float:
        """f0 附近窄窗口里的峰值 (留点余量给音准误差和泛音偏移)。"""
        if f0 <= 0 or f0 >= self.sr / 2:
            return 0.0
        lo = f0 * (2.0 ** (-halfwidth_semitones / 12.0))
        hi = f0 * (2.0 ** (halfwidth_semitones / 12.0))
        i0 = int(np.searchsorted(self.freqs, lo)) - 1
        i1 = int(np.searchsorted(self.freqs, hi)) + 1
        i0 = max(0, i0)
        i1 = min(len(spec), max(i1, i0 + 2))
        return float(spec[i0:i1].max()) if i1 > i0 else 0.0


def melody_prominence(wav_path: str,
                      perf: Performance,
                      sample_points: int = 4,
                      win: float = 0.22,
                      sr: int = 22050) -> Report:
    """每个旋律音取 sample_points 个点, 算 旋律dB - 最响伴奏dB, 正的表示旋律在上面。

    和旋律同音高的伴奏跳过 (那是 check_unisons 管的)。长音最差的点是负的算 error, 短音 warn。
    """
    rep = Report()
    y, sr = load_audio(wav_path, sr=sr)
    an = _Analyser(y, sr, win)
    sudu = perf.tempo

    # tempo.seconds() 每次都从 0 积分, 缓存一下
    miao_huancun: Dict[float, float] = {}

    def secs(beat: float) -> float:
        key = round(beat, 6)
        if key not in miao_huancun:
            miao_huancun[key] = sudu.seconds(beat)
        return miao_huancun[key]

    xuanlv = perf.melody()
    banzou = perf.accompaniment()
    quanbu_db: List[float] = []
    meige_yin: List[float] = []
    n_measured = n_no_acc = 0

    for m in xuanlv:
        t0, t1 = secs(m.onset), secs(m.end)
        if t1 <= t0:
            continue
        f_mel = _hz(m.pitch)
        zuicha = None
        zuicha_banzou: Optional[Note] = None
        zuicha_t = t0
        liangle = 0

        for i in range(max(1, sample_points)):
            frac = (i + 0.5) / max(1, sample_points)
            beat = m.onset + frac * m.dur
            t = t0 + frac * (t1 - t0)
            if t >= len(y) / sr:
                break
            zaixiang = [a for a in banzou
                        if a.onset <= beat < a.end and a.pitch != m.pitch]
            if not zaixiang:
                n_no_acc += 1
                continue
            pu = an.spectrum(t)
            mel_db = _db(an.amp_at(pu, f_mel))
            zuixiang, zuixiang_db = None, -999.0
            for a in zaixiang:
                d = _db(an.amp_at(pu, _hz(a.pitch)))
                if d > zuixiang_db:
                    zuixiang, zuixiang_db = a, d
            cha = mel_db - zuixiang_db
            quanbu_db.append(cha)
            liangle += 1
            if zuicha is None or cha < zuicha:
                zuicha, zuicha_banzou, zuicha_t = cha, zuixiang, t

        if zuicha is None:
            continue
        n_measured += 1
        meige_yin.append(zuicha)
        if zuicha < 0.0:
            changyin = m.dur >= HELD_BEATS
            rep.add(Finding(
                check="melody_prominence", beat=m.onset,
                detail="melody %s(%d) is %.1f dB BELOW %s(%d) [%s tag=%s] at "
                       "t=%.2fs%s"
                       % (name(m.pitch), m.pitch, -zuicha,
                          name(zuicha_banzou.pitch), zuicha_banzou.pitch,
                          zuicha_banzou.voice.value, zuicha_banzou.tag or "untagged",
                          zuicha_t, " (held note)" if changyin else ""),
                severity="error" if changyin else "warn",
                notes=(m, zuicha_banzou)))

    if quanbu_db:
        arr = np.asarray(quanbu_db)
        rep.stats["worst_db"] = round(float(arr.min()), 2)
        rep.stats["median_db"] = round(float(np.median(arr)), 2)
        rep.stats["pct_below_zero"] = round(
            100.0 * float((arr < 0).sum()) / len(arr), 1)
        rep.stats["prominence.samples"] = int(len(arr))
        rep.stats["prominence.melody_notes"] = n_measured
        rep.stats["prominence.notes_below_zero"] = int(
            sum(1 for v in meige_yin if v < 0))
        rep.stats["prominence.p10_db"] = round(float(np.percentile(arr, 10)), 2)
    else:
        rep.stats["worst_db"] = float("nan")
        rep.stats["median_db"] = float("nan")
        rep.stats["pct_below_zero"] = float("nan")
        rep.add(Finding(check="melody_prominence", beat=0.0, severity="warn",
                        detail="nothing measurable: no melody note ever sounds "
                               "against an accompaniment note"))
    rep.stats["prominence.samples_without_accompaniment"] = n_no_acc
    return rep


def spectral_profile(wav_path: str, sr: int = 22050) -> dict:
    """整段录音的频谱重心、滚降、低中高三段能量占比。"""
    import librosa
    y, sr = load_audio(wav_path, sr=sr)
    if not len(y):
        return {}
    n_fft, hop = 2048, 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    pinlv = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    zhongxin = librosa.feature.spectral_centroid(S=S, sr=sr, freq=pinlv)[0]
    r85 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    r90 = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.90)[0]
    rms = librosa.feature.rms(S=S, frame_length=n_fft, hop_length=hop)[0]

    # 比峰值低 60 dB 以下的静音不算
    youshengyin = rms > (rms.max() * 1e-3 + _EPS)
    if not youshengyin.any():
        youshengyin = np.ones_like(rms, dtype=bool)

    nengliang = (S ** 2).sum(axis=1)
    lo, hi = BAND_EDGES
    zong = float(nengliang.sum()) + _EPS
    di = float(nengliang[pinlv < lo].sum()) / zong
    zhong = float(nengliang[(pinlv >= lo) & (pinlv < hi)].sum()) / zong
    gao = float(nengliang[pinlv >= hi].sum()) / zong

    db = 20.0 * np.log10(np.maximum(rms[youshengyin], _EPS))
    return {
        "duration_s": round(len(y) / sr, 3),
        "spectral_centroid_hz": round(float(np.mean(zhongxin[youshengyin])), 1),
        "rolloff85_hz": round(float(np.mean(r85[youshengyin])), 1),
        "rolloff90_hz": round(float(np.mean(r90[youshengyin])), 1),
        "weight_bass": round(di, 4),
        "weight_mid": round(zhong, 4),
        "weight_treble": round(gao, 4),
        "band_edges_hz": BAND_EDGES,
        "dynamic_spread_db": round(
            float(np.percentile(db, 95) - np.percentile(db, 5)), 2),
        "rms_peak_db": round(float(db.max()), 2),
    }


def reverb_tail(wav_path: str, drop_db: float = 30.0, sr: int = 22050) -> float:
    """最后一个起音衰减 drop_db 要几秒, 录音先结束就按斜率外推。粗略的, 只用来互相比。"""
    import librosa
    y, sr = load_audio(wav_path, sr=sr)
    if len(y) < sr // 10:
        return float("nan")
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    if not rms.any():
        return float("nan")
    db = 20.0 * np.log10(np.maximum(rms, _EPS))
    shijian = librosa.frames_to_time(np.arange(len(db)), sr=sr, hop_length=hop)

    try:
        qiyin = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop,
                                           units="frames")
    except Exception:
        qiyin = np.array([], dtype=int)
    kaishi = int(qiyin[-1]) if len(qiyin) else int(np.argmax(rms))
    kaishi = min(kaishi, len(db) - 3)
    fengzhi = float(db[kaishi:kaishi + 4].max())
    weiba = db[kaishi:]
    tt = shijian[kaishi:]

    diyu = np.nonzero(weiba <= fengzhi - drop_db)[0]
    if len(diyu):
        return round(float(tt[diyu[0]] - tt[0]), 3)
    if len(weiba) < 4:
        return float("nan")
    xielv, jieju = np.polyfit(tt - tt[0], weiba, 1)   # dB/秒
    if xielv >= -1e-6:
        return float("nan")
    return round(float((fengzhi - drop_db - jieju) / xielv), 3)


def synth_wav(perf: Performance, path: str, sr: int = 22050,
              melody_gain: float = 1.0, tail: float = 2.5,
              partials: Tuple[float, ...] = (1.0, 0.45, 0.22, 0.10)) -> str:
    """用很简单的加法合成渲染成 wav, 测试用, 不需要采样器。melody_gain 只乘旋律。"""
    import wave
    sudu = perf.tempo
    huancun: Dict[float, float] = {}

    def secs(beat: float) -> float:
        key = round(beat, 6)
        if key not in huancun:
            huancun[key] = sudu.seconds(beat)
        return huancun[key]

    zongchang = int((secs(perf.length) + tail) * sr) + 1
    buf = np.zeros(zongchang, dtype=np.float64)

    for n in perf.notes:
        t0, t1 = secs(n.onset), secs(n.end)
        i0 = int(t0 * sr)
        i1 = min(zongchang, int((t1 + 1.2) * sr))
        if i1 <= i0:
            continue
        t = np.arange(i1 - i0, dtype=np.float64) / sr
        env = np.exp(-t / 2.2)
        env *= np.exp(-np.clip(t - (t1 - t0), 0.0, None) / 0.18)
        env *= np.clip(t / 0.006, 0.0, 1.0)
        fudu = (n.vel / 127.0) ** 1.8
        if n.voice is Voice.MELODY:
            fudu *= melody_gain
        f = _hz(n.pitch)
        sig = np.zeros_like(t)
        for k, w in enumerate(partials, start=1):
            if f * k < sr / 2:
                sig += w * np.sin(2.0 * np.pi * f * k * t)
        buf[i0:i1] += fudu * env * sig

    fengzhi = float(np.abs(buf).max())
    if fengzhi > 0:
        buf *= 0.89 / fengzhi
    pcm = (buf * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


if __name__ == "__main__":
    import os
    import tempfile
    from cantabile.audit import demo, banner, synth_wav

    perf, harm, tune = demo()
    d = tempfile.mkdtemp(prefix="cantabile-audit-")

    haode = os.path.join(d, "balanced.wav")
    synth_wav(perf, haode, melody_gain=2.2)
    banner("balance.melody_prominence on a BALANCED render (melody +7 dB)")
    print(melody_prominence(haode, perf).summary())

    huaide = os.path.join(d, "buried.wav")
    synth_wav(perf, huaide, melody_gain=0.30)
    banner("balance.melody_prominence on a BURIED melody (melody -10 dB)")
    r = melody_prominence(huaide, perf)
    print(r.summary())
    print("  ok() = %s" % r.ok())

    banner("balance.spectral_profile (buried render)")
    for k, v in spectral_profile(huaide).items():
        print("  %-22s %s" % (k, v))
    print("  %-22s %s" % ("reverb_tail_s", reverb_tail(huaide)))
