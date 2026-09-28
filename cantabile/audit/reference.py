"""和真实录音比风格。

profile 量速度、各音区音符密度、力度、亮度、混响尾巴; compare 拿去和参考范围比。
REFERENCE_ACCOMPANIMENT 是从参考钢琴伴奏录音里量出来的。要 numpy 和 librosa。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from cantabile.model import Finding, Performance, Report
from cantabile.audit.balance import load_audio, reverb_tail, spectral_profile

__all__ = ["profile", "compare", "REFERENCE_ACCOMPANIMENT", "REGISTERS"]

# 按基频分音区, 算音符密度用
REGISTERS = (
    ("bass",  0.0,     195.998),     # < G3
    ("tenor", 195.998, 329.628),     # G3 .. E4
    ("alto",  329.628, 523.251),     # E4 .. C5
    ("top",   523.251, None),        # >= C5
)

# 参考钢琴伴奏录音里量的。元组是几份录音的 (最小, 最大), 单个数是只量到一个值
REFERENCE_ACCOMPANIMENT: Dict[str, Union[float, Tuple[float, float]]] = {
    "tempo_bpm":             (89.0, 108.0),
    "notes_per_sec":         (5.8, 10.6),
    "notes_per_sec_bass":    (3.45, 4.30),
    "notes_per_sec_tenor":   (2.85, 3.23),
    "notes_per_sec_alto":    (0.80, 2.13),
    "notes_per_sec_top":     (0.40, 0.55),
    "vel_p10":               42.0,
    "vel_median":            63.0,
    "vel_p90":               88.0,
    "weight_bass":           (0.25, 0.27),
    "weight_mid":            (0.55, 0.56),
    "weight_treble":         (0.17, 0.18),
    "dynamic_spread_db":     (17.6, 20.9),
    "spectral_centroid_hz":  (734.0, 838.0),
    "rolloff90_hz":          (1734.0, 1971.0),
    "reverb_tail_s":         (3.37, 3.60),
}

_EPS = 1e-12


def _band_note_rate(y, sr, duration):
    """每个音区每秒几个起音。分频段找, 不然两只手的一个和弦只算一个。"""
    import librosa
    n_fft, hop = 2048, 512
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    pinlv = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    jieguo = {}
    zongshu = 0
    for ming, lo, hi in REGISTERS:
        mask = pinlv >= lo
        if hi is not None:
            mask &= pinlv < hi
        if not mask.any():
            jieguo[ming] = 0.0
            continue
        sub = S[mask]
        env = librosa.onset.onset_strength(
            S=librosa.amplitude_to_db(sub, ref=np.max), sr=sr, hop_length=hop)
        zhen = librosa.onset.onset_detect(onset_envelope=env, sr=sr,
                                          hop_length=hop, backtrack=False)
        n = int(len(zhen))
        zongshu += n
        jieguo[ming] = round(n / max(duration, _EPS), 3)
    jieguo["_total"] = round(zongshu / max(duration, _EPS), 3)
    return jieguo


def profile(wav_path: str, perf: Optional[Performance] = None) -> dict:
    """一段录音的风格指纹。力度分位数光靠音频量不出来, 给了 perf 就从 perf 里取。"""
    import librosa
    y, sr = load_audio(wav_path)
    shichang = len(y) / float(sr) if len(y) else 0.0
    jieguo: Dict[str, object] = {"source": wav_path, "duration_s": round(shichang, 3)}
    if not len(y):
        return jieguo

    sudu = librosa.beat.beat_track(y=y, sr=sr)[0]
    jieguo["tempo_bpm"] = round(float(np.atleast_1d(sudu)[0]), 2)

    midu = _band_note_rate(y, sr, shichang)
    jieguo["notes_per_sec"] = midu.pop("_total")
    for ming, _, _ in REGISTERS:
        jieguo["notes_per_sec_" + ming] = midu[ming]

    jieguo.update(spectral_profile(wav_path))
    jieguo["duration_s"] = round(shichang, 3)
    jieguo["reverb_tail_s"] = reverb_tail(wav_path)

    if perf is not None and perf.notes:
        lidu = np.asarray([n.vel for n in perf.notes], dtype=float)
        jieguo["vel_p10"] = round(float(np.percentile(lidu, 10)), 1)
        jieguo["vel_median"] = round(float(np.median(lidu)), 1)
        jieguo["vel_p90"] = round(float(np.percentile(lidu, 90)), 1)
        jieguo["vel_source"] = "transcription"
    else:
        jieguo["vel_source"] = "unavailable (no transcription supplied)"
    return jieguo


def _as_ranges(references):
    """变成 {key: (lo, hi)}。可以是一个字典, 也可以是一串 profile() 结果 (取最小最大)。"""
    if isinstance(references, dict):
        fanwei = {}
        for k, v in references.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                fanwei[k] = (float(v), float(v))
            elif isinstance(v, (tuple, list)) and len(v) == 2 \
                    and all(isinstance(x, (int, float)) for x in v):
                fanwei[k] = (float(min(v)), float(max(v)))
        return fanwei

    shouji = {}
    for ref in references:
        for k, v in _as_ranges(ref).items():
            shouji.setdefault(k, []).extend(v)
    return {k: (min(v), max(v)) for k, v in shouji.items()}


def compare(target: dict,
            references,
            tolerance: float = 0.25) -> Report:
    """超出范围按离最近边界的比例算: tolerance 以内算对, 以外 warn, 两倍以外 error。"""
    rep = Report()
    fanwei = _as_ranges(references)
    within = near = outside = missing = 0
    worst_key, worst_dev = None, 0.0

    for key in sorted(fanwei):
        lo, hi = fanwei[key]
        val = target.get(key)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            missing += 1
            rep.add(Finding(check="reference", beat=0.0, severity="info",
                            detail="%s: not measured (reference %s)"
                                   % (key, _fmt_range(lo, hi))))
            continue
        val = float(val)
        if lo - _EPS <= val <= hi + _EPS:
            within += 1
            continue
        edge = lo if val < lo else hi
        dev = abs(val - edge) / max(abs(edge), _EPS)
        if dev > worst_dev:
            worst_key, worst_dev = key, dev
        if dev <= tolerance:
            near += 1
            continue
        outside += 1
        rep.add(Finding(
            check="reference", beat=0.0,
            severity="error" if dev > 2 * tolerance else "warn",
            detail="%s = %s, %s reference %s (%.0f%% %s the nearest edge)"
                   % (key, _fmt(val), "below" if val < lo else "above",
                      _fmt_range(lo, hi), 100.0 * dev,
                      "below" if val < lo else "above")))

    rep.stats["reference.keys"] = len(fanwei)
    rep.stats["reference.within_range"] = within
    rep.stats["reference.within_tolerance"] = near
    rep.stats["reference.outside"] = outside
    rep.stats["reference.not_measured"] = missing
    rep.stats["reference.match_pct"] = (
        round(100.0 * (within + near) / max(1, within + near + outside), 1))
    if worst_key:
        rep.stats["reference.worst"] = "%s (%.0f%% off)" % (worst_key,
                                                            100.0 * worst_dev)
    return rep


def _fmt(v: float) -> str:
    return ("%.3f" % v).rstrip("0").rstrip(".") if abs(v) < 10 else "%.1f" % v


def _fmt_range(lo: float, hi: float) -> str:
    return _fmt(lo) if lo == hi else "%s-%s" % (_fmt(lo), _fmt(hi))


if __name__ == "__main__":
    import os
    import tempfile
    from cantabile.audit import demo, banner, synth_wav

    perf, harm, tune = demo()
    d = tempfile.mkdtemp(prefix="cantabile-audit-")
    wav = os.path.join(d, "demo.wav")
    synth_wav(perf, wav, melody_gain=1.0)

    banner("reference.REFERENCE_ACCOMPANIMENT  (shipped measurements)")
    for k, v in REFERENCE_ACCOMPANIMENT.items():
        print("  %-22s %s" % (k, v))

    banner("reference.profile of the synthetic demo render")
    p = profile(wav, perf)
    for k, v in p.items():
        print("  %-22s %s" % (k, v))

    banner("reference.compare(demo, REFERENCE_ACCOMPANIMENT)  "
           "- an 8-beat sine sketch is nothing like a real pianist")
    print(compare(p, REFERENCE_ACCOMPANIMENT).summary(limit=20))

    banner("reference.compare of a profile that sits inside every range "
           "-> must be clean")
    mid = {}
    for k, v in REFERENCE_ACCOMPANIMENT.items():
        mid[k] = float(sum(v)) / 2.0 if isinstance(v, tuple) else float(v)
    r = compare(mid, REFERENCE_ACCOMPANIMENT)
    print(r.summary())
    print("  ok() = %s" % r.ok())
