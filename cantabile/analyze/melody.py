"""从拆好的几条流里找出哪条是旋律: 统计特征加权打分。

参考: Rizo et al. ISMIR 2006; Huron 2001。
"""
from __future__ import annotations

import math
from collections import namedtuple

from ..model import Note, Voice
from .streams import Stream, separate, skyline, _as_note, _prep

__all__ = ["MELODY_FEATURES", "FEATURE_DOCS", "DEFAULT_WEIGHTS", "Feature",
           "stream_features", "identify", "melody_line", "evaluate_against"]

Feature = namedtuple("Feature", "name doc")

# 每个特征 0..1, 越大越像旋律
MELODY_FEATURES = (
    Feature("top_voice_rate",
            "Fraction of the stream's notes that are the highest pitch "
            "sounding anywhere at their onset.  The skyline baseline of "
            "Uitdenbogerd & Zobel in feature form, and Karydis et al.'s Top "
            "Voice Rule: on keyboard textures this is the strongest single "
            "predictor of the tune."),
    Feature("pitch_mean",
            "Mean pitch, scaled between the lowest and highest mean of the "
            "streams being compared.  Rizo et al.'s pitch-mean descriptor."),
    Feature("highest",
            "The stream's top note, scaled across the streams being compared."),
    Feature("occupation",
            "Proportion of the piece's beats during which this stream is "
            "sounding.  Rizo et al.'s occupation rate: the tune runs through "
            "the whole piece, a countermelody does not."),
    Feature("note_density",
            "Notes per beat over the stream's own span, scaled against the "
            "densest stream.  Rizo et al.'s note count / duration ratio."),
    Feature("note_share",
            "Share of all the notes in the texture."),
    Feature("contour_smoothness",
            "1 - (mean absolute melodic interval / an octave), clamped.  "
            "Huron's pitch-proximity principle: tunes move by step, "
            "accompaniment lines leap between chord tones."),
    Feature("register_stability",
            "1 - (pitch standard deviation / an octave), clamped.  Rizo et "
            "al.'s pitch standard deviation, inverted: a tune keeps one "
            "tessitura."),
    Feature("compactness",
            "1 - (pitch range / three octaves), clamped.  Separates a real "
            "voice from a stream that has swept the whole keyboard."),
    Feature("rhythmic_regularity",
            "1 - (standard deviation of log2 inter-onset intervals / 2), "
            "clamped.  Rizo et al.'s note-duration standard deviation, "
            "inverted: a tune has a regular tread."),
    Feature("onbeat_rate",
            "Fraction of onsets that fall on a beat.  The complement of Rizo "
            "et al.'s syncopation descriptor: accompaniment figuration is what "
            "lives off the beat."),
)

FEATURE_DOCS = {f.name: f.doc for f in MELODY_FEATURES}

_NAMES = tuple(f.name for f in MELODY_FEATURES)

# 手调的, 加起来是 1; 故意摊开, 免得一个特征说了算
DEFAULT_WEIGHTS = {
    "top_voice_rate": 0.28,
    "pitch_mean": 0.22,
    "highest": 0.04,
    "occupation": 0.12,
    "note_density": 0.03,
    "note_share": 0.03,
    "contour_smoothness": 0.10,
    "register_stability": 0.07,
    "compactness": 0.05,
    "rhythmic_regularity": 0.04,
    "onbeat_rate": 0.02,
}


class _Context:

    def __init__(self, notes, streams=()):
        self.items = _prep(notes)
        self.streams = list(streams)
        self.n_notes = len(self.items)
        self.start = min((it.onset for it in self.items), default=0.0)
        self.end = max((it.end for it in self.items), default=0.0)
        self.span = max(self.end - self.start, 1e-9)
        self._onsets = sorted({round(it.onset, 6) for it in self.items})
        self._top = {}
        for t in self._onsets:
            xiang = [it.pitch for it in self.items
                     if it.onset <= t + 1e-6 < it.end]
            if xiang:
                self._top[t] = max(xiang)

    def top_at(self, beat):
        return self._top.get(round(beat, 6))


def _context(context) -> _Context:
    if isinstance(context, _Context):
        return context
    seq = list(context)
    if seq and isinstance(seq[0], Stream):
        yin = [n for s in seq for n in s.notes]
        return _Context(yin, seq)
    return _Context(seq)


def _clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


def _scaled(value, values):
    lo, hi = min(values), max(values)
    return 0.5 if hi - lo < 1e-9 else (value - lo) / (hi - lo)


def stream_features(stream, context) -> dict:
    """算一条流的 MELODY_FEATURES。context 最好给全部的流, 不然缩放没参照。

    >>> from cantabile.model import Note
    >>> from cantabile.analyze.streams import Stream
    >>> tune = Stream([Note(0, 1, 72, 80), Note(1, 1, 74, 80)])
    >>> bass = Stream([Note(0, 1, 48, 80), Note(1, 1, 50, 80)])
    >>> f = stream_features(tune, [tune, bass])
    >>> f["top_voice_rate"], f["pitch_mean"]
    (1.0, 1.0)
    """
    ctx = _context(context)
    stream = stream if isinstance(stream, Stream) else Stream(list(stream))
    yin = sorted((_as_note(n) for n in stream.notes),
                 key=lambda n: (n.onset, n.pitch))
    f = {k: 0.0 for k in _NAMES}
    if not yin:
        return f

    tongban = ctx.streams or [stream]
    yingao = [n.pitch for n in yin]
    junzhi = sum(yingao) / len(yingao)

    tb_junzhi, tb_zuigao, tb_midu = [], [], []
    for s in tongban:
        ps = [n.pitch for n in s.notes] or [junzhi]
        tb_junzhi.append(sum(ps) / len(ps))
        tb_zuigao.append(max(ps))
        tb_midu.append(s.density() if isinstance(s, Stream)
                       else len(ps) / ctx.span)
    if stream not in tongban:
        tb_junzhi.append(junzhi)
        tb_zuigao.append(max(yingao))
        tb_midu.append(stream.density())

    zaiding = sum(1 for n in yin
                  if (ctx.top_at(n.onset) or -1) <= n.pitch)
    f["top_voice_rate"] = zaiding / len(yin)
    f["pitch_mean"] = _scaled(junzhi, tb_junzhi)
    f["highest"] = _scaled(max(yingao), tb_zuigao)

    fugai = 0.0
    last = -1e18
    for n in yin:
        a, b = max(n.onset, last), n.end
        if b > a:
            fugai += b - a
            last = b
    f["occupation"] = _clamp(fugai / ctx.span)
    f["note_density"] = _scaled(stream.density(), tb_midu) \
        if max(tb_midu) > min(tb_midu) else 0.5
    f["note_share"] = len(yin) / max(1, ctx.n_notes)

    buju = [abs(b.pitch - a.pitch) for a, b in zip(yin, yin[1:])]
    f["contour_smoothness"] = _clamp(1.0 - (sum(buju) / len(buju)) / 12.0) \
        if buju else 1.0
    var = sum((p - junzhi) ** 2 for p in yingao) / len(yingao)
    f["register_stability"] = _clamp(1.0 - math.sqrt(var) / 12.0)
    f["compactness"] = _clamp(1.0 - (max(yingao) - min(yingao)) / 36.0)

    iois = [b.onset - a.onset for a, b in zip(yin, yin[1:])
            if b.onset - a.onset > 1e-6]
    if len(iois) >= 2:
        logs = [math.log(x, 2) for x in iois]
        m = sum(logs) / len(logs)
        sd = math.sqrt(sum((x - m) ** 2 for x in logs) / len(logs))
        f["rhythmic_regularity"] = _clamp(1.0 - sd / 2.0)
    else:
        f["rhythmic_regularity"] = 1.0

    f["onbeat_rate"] = sum(1 for n in yin
                           if abs(n.onset - round(n.onset)) < 1e-3) / len(yin)
    return f


def identify(streams, *, weights=None) -> tuple:
    """挑出旋律那条流, 返回 (melody, accompaniment, scores)。"""
    ss = [s if isinstance(s, Stream) else Stream(list(s)) for s in streams]
    ss = [s for s in ss if s.notes]
    if not ss:
        return None, [], []

    w = {k: float(v) for k, v in (weights or DEFAULT_WEIGHTS).items()
         if k in _NAMES}
    total = sum(w.values()) or 1.0
    w = {k: v / total for k, v in w.items()}

    ctx = _context(ss)
    defen = []
    for s in ss:
        f = stream_features(s, ctx)
        defen.append({"stream": s, "features": f,
                      "score": sum(w.get(k, 0.0) * f[k] for k in _NAMES)})
    best = max(range(len(defen)),
               key=lambda i: (defen[i]["score"],
                              defen[i]["features"]["pitch_mean"]))
    xuanlv = ss[best]
    return xuanlv, [s for i, s in enumerate(ss) if i != best], defen


def melody_line(notes, *, method="contig", parts=4, weights=None,
                salience=None, tol=1.0 / 16, durations="span",
                tag=True) -> list:
    """分声部再挑旋律, 返回 [(beat, pitch, dur)]。parts 填写谱上的声部数。"""
    if not notes:
        return []
    sep = separate(notes, method=method, max_voices=parts,
                   weights=weights, tol=tol)
    mel, _rest, _scores = identify(sep, weights=salience)
    if mel is None:
        return []
    mel = mel.monophonic()
    if tag:
        mel = mel.tagged(Voice.MELODY)
    ns = sorted(mel.notes, key=lambda n: (n.onset, n.pitch))
    # 最后一个音拖到最后一拍结束, 跟 extract_soprano 一样
    end = math.ceil(max((_as_note(n).onset + _as_note(n).dur for n in notes),
                        default=0.0) - 1e-6)
    jieguo = []
    for i, n in enumerate(ns):
        if durations == "gate":
            dur = n.dur
        elif i + 1 < len(ns):
            dur = ns[i + 1].onset - n.onset
        else:
            dur = max(n.dur, end - n.onset)
        jieguo.append((round(n.onset, 6), n.pitch, round(max(dur, 1e-6), 6)))
    return jieguo


def _line(x):
    if isinstance(x, Stream):
        x = x.notes
    jieguo = []
    for e in x:
        if isinstance(e, Note):
            jieguo.append((float(e.onset), int(e.pitch), float(e.dur)))
        else:
            e = tuple(e)
            if len(e) == 3:
                jieguo.append((float(e[0]), int(e[1]), float(e[2])))
            elif len(e) == 4:
                jieguo.append((float(e[0]), int(e[2]), float(e[1])))
            else:
                raise TypeError(f"cannot read {e!r} as a melody note")
    jieguo.sort()
    return jieguo


def evaluate_against(tune_melody, predicted, *, onset_tol=0.12,
                     limit=12) -> dict:
    """音级别的 P/R/F1: 同音高、起点差在 onset_tol 以内就算对上。"""
    truth, pred = _line(tune_melody), _line(predicted)
    yongle = [False] * len(truth)
    pairs = []
    for pi, (pb, pp, pd) in enumerate(pred):
        best, bestd = None, None
        for ti, (tb, tp, td) in enumerate(truth):
            if yongle[ti] or tp != pp:
                continue
            d = abs(tb - pb)
            if d <= onset_tol and (bestd is None or d < bestd):
                best, bestd = ti, d
        if best is not None:
            yongle[best] = True
            pairs.append((best, pi))
    tp_n = len(pairs)
    prec = tp_n / len(pred) if pred else 0.0
    rec = tp_n / len(truth) if truth else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    dur_ok = sum(1 for ti, pi in pairs
                 if abs(truth[ti][2] - pred[pi][2]) <= 0.13)
    duishang = {pi for _ti, pi in pairs}
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp_n, "fp": len(pred) - tp_n, "fn": len(truth) - tp_n,
        "n_truth": len(truth), "n_predicted": len(pred),
        "duration_agreement": (dur_ok / tp_n) if tp_n else 0.0,
        "missing": [truth[i] for i in range(len(truth)) if not yongle[i]][:limit],
        "extra": [pred[i] for i in range(len(pred))
                  if i not in duishang][:limit],
    }


def _main():
    from .streams import _demo_four_part

    notes, truth, tune = _demo_four_part()
    want = [(float(2 * k), p, 2.0) for k, p in enumerate(tune)]
    print(f"synthetic four-part piece: {len(notes)} notes, tune = {tune}")

    sep = separate(notes, max_voices=4)
    mel, rest, scores = identify(sep)
    print(f"identify -> {mel.label}  ({len(rest)} accompaniment streams)")
    for s in scores:
        f = s["features"]
        print(f"   {s['stream'].label:10s} score={s['score']:.3f} "
              f"top={f['top_voice_rate']:.2f} pitch={f['pitch_mean']:.2f} "
              f"occ={f['occupation']:.2f} smooth={f['contour_smoothness']:.2f} "
              f"reg={f['register_stability']:.2f}")
    assert mel is sep[-1], "melody identification did not pick the top stream"

    got = melody_line(notes, parts=4)
    m = evaluate_against(want, got)
    print(f"contig  P={m['precision']:.3f} R={m['recall']:.3f} "
          f"F1={m['f1']:.3f}  dur={m['duration_agreement']:.3f}")
    assert m["f1"] == 1.0, m

    sky = skyline(notes)
    ms = evaluate_against(want, sky)
    print(f"skyline P={ms['precision']:.3f} R={ms['recall']:.3f} "
          f"F1={ms['f1']:.3f}")
    print(f"        skyline extras: {ms['extra'][:4]}")
    assert ms["f1"] < m["f1"], "skyline should not beat contig mapping here"

    assert set(FEATURE_DOCS) == set(_NAMES)
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9
    assert evaluate_against([], [])["f1"] == 0.0
    assert melody_line([]) == []
    print("features documented, weights normalised, empty input safe: OK")


if __name__ == "__main__":
    _main()
