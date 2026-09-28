"""训练一个分类器分辨真人录音和生成的渲染。

每段录音切成重叠的窗口, 算一堆声学特征。按"留一段录音"打分, 分类器没法靠背下某次演奏取胜。
准确率接近 0.5 就是分不出来; 它最依赖的特征就是还不像的地方。
"""
from __future__ import annotations
import subprocess, sys, os
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SR = 22050
WIN = 4.0          # 每个样本几秒
HOP = 2.0


def decode(path, start=0.0, dur=None):
    a = ["ffmpeg", "-v", "error"]
    if start:
        a += ["-ss", str(start)]
    if dur:
        a += ["-t", str(dur)]
    a += ["-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    return np.frombuffer(subprocess.run(a, capture_output=True).stdout, dtype="<f4")


def features(y, sr=SR) -> dict:
    import librosa
    f = {}
    if len(y) < sr:
        return f
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=256))
    pinlv = librosa.fft_frequencies(sr=sr, n_fft=2048)
    rms = librosa.feature.rms(S=S)[0]
    db = 20 * np.log10(rms + 1e-9)

    # 电平起伏
    f["rms_mean"], f["rms_sd"] = db.mean(), db.std()
    f["rms_skew"] = float(((db - db.mean()) ** 3).mean() / (db.std() ** 3 + 1e-9))
    f["rms_range"] = np.percentile(db, 95) - np.percentile(db, 5)
    d = np.diff(db)
    f["level_rise_sd"], f["level_fall_sd"] = d[d > 0].std() if (d > 0).any() else 0, \
                                             d[d < 0].std() if (d < 0).any() else 0
    f["level_ac1"] = float(np.corrcoef(db[:-1], db[1:])[0, 1])

    # 起音和间隔
    oe = librosa.onset.onset_strength(y=y, sr=sr)
    qiyin = librosa.onset.onset_detect(onset_envelope=oe, sr=sr, units="time", delta=0.05)
    f["onset_rate"] = len(qiyin) / (len(y) / sr)
    if len(qiyin) > 3:
        ioi = np.diff(qiyin)
        f["ioi_mean"], f["ioi_sd"] = ioi.mean(), ioi.std()
        f["ioi_cv"] = ioi.std() / (ioi.mean() + 1e-9)
        f["ioi_ac1"] = float(np.corrcoef(ioi[:-1], ioi[1:])[0, 1]) if len(ioi) > 2 else 0
    f["onset_strength_sd"] = oe.std()
    f["onset_strength_skew"] = float(((oe - oe.mean()) ** 3).mean() / (oe.std() ** 3 + 1e-9))

    # 频谱
    cen = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    f["centroid_mean"], f["centroid_sd"] = cen.mean(), cen.std()
    f["rolloff"] = librosa.feature.spectral_rolloff(S=S, sr=sr)[0].mean()
    f["bandwidth"] = librosa.feature.spectral_bandwidth(S=S, sr=sr)[0].mean()
    f["flatness"] = librosa.feature.spectral_flatness(S=S)[0].mean()
    f["contrast"] = librosa.feature.spectral_contrast(S=S, sr=sr).mean()
    pingjun = S.mean(axis=1); pingjun = pingjun / (pingjun.sum() + 1e-12)
    for lo, hi, nm in ((30, 120, "sub"), (120, 300, "low"), (300, 800, "lowmid"),
                       (800, 2000, "mid"), (2000, 5000, "hi"), (5000, 11000, "air")):
        m = (pinlv >= lo) & (pinlv < hi)
        f["band_" + nm] = float(pingjun[m].sum())
    # 帧和帧之间频谱变化多大
    f["flux"] = float(np.mean(np.abs(np.diff(S, axis=1)).sum(axis=0)) / (S.sum(axis=0).mean() + 1e-9))

    # 两次起音之间的衰减
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=256)
    if len(qiyin):
        juli = np.array([min(abs(x - qiyin)) for x in t])
        g, h = rms[juli > 0.2], rms[juli < 0.05]
        f["gap_ratio"] = float(g.mean() / (h.mean() + 1e-9)) if len(g) and len(h) else 0
        f["gap_share"] = float((juli > 0.2).mean())
    # 每半秒里最响那帧之后的衰减斜率
    xielv = []
    bu = int(0.5 * sr / 256)
    for i in range(0, len(db) - bu, bu):
        duan = db[i:i + bu]
        k = int(np.argmax(duan))
        if len(duan) - k > 4:
            xielv.append(np.polyfit(np.arange(len(duan) - k), duan[k:], 1)[0])
    f["decay_slope"] = float(np.mean(xielv)) if xielv else 0.0
    f["decay_slope_sd"] = float(np.std(xielv)) if xielv else 0.0

    # 谐波/打击分离: 击弦噪声 vs 音
    H, P = librosa.decompose.hpss(S)
    f["percussive_share"] = float(P.sum() / (H.sum() + P.sum() + 1e-12))
    return f


def windows(path, label, start=0.0, dur=60.0):
    y = decode(path, start, dur)
    chuangkou = []
    n = int(WIN * SR)
    bu = int(HOP * SR)
    for i in range(0, max(0, len(y) - n), bu):
        f = features(y[i:i + n])
        if f:
            chuangkou.append((f, label, str(path)))
    return chuangkou


def main():
    import warnings
    warnings.filterwarnings("ignore")
    # 参考录音不放在仓库里, 命令行传进来:
    #   python eval/discriminator.py --real a.mp3 b.mp3 --fake ours.mp3
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--real", nargs="+", required=True,
                    help="recordings of a human playing")
    ap.add_argument("--fake", nargs="+", required=True,
                    help="renders to test against them")
    ap.add_argument("--start", type=float, default=20.0)
    ap.add_argument("--dur", type=float, default=120.0)
    a = ap.parse_args()
    zhenren = [(p, a.start, a.dur) for p in a.real]
    shengcheng = [(p, a.start, a.dur) for p in a.fake]
    shuju = []
    for p, s, d in zhenren:
        shuju += windows(p, 1, s, d)
    for p, s, d in shengcheng:
        if Path(p).exists():
            shuju += windows(p, 0, s, d)
    keys = sorted(set().union(*[set(f) for f, _, _ in shuju]))
    X = np.array([[f.get(k, 0.0) for k in keys] for f, _, _ in shuju])
    y = np.array([l for _, l, _ in shuju])
    laiyuan = np.array([s for _, _, s in shuju])
    X = np.nan_to_num(X)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    print(f"{len(X)} windows ({int(y.sum())} real / {int((1-y).sum())} generated), "
          f"{len(keys)} features")

    # 留一段录音
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    zhunquelv = []
    for liuchu in sorted(set(laiyuan)):
        tr = laiyuan != liuchu
        te = ~tr
        if not te.any() or len(set(y[tr])) < 2:
            continue
        m = LogisticRegression(max_iter=2000, C=0.5).fit(Xs[tr], y[tr])
        zhunquelv.append(((m.predict(Xs[te]) == y[te]).mean(), liuchu))
    print("\nleave-one-recording-out accuracy (0.5 = indistinguishable):")
    for a, h in zhunquelv:
        print(f"   {a:5.2f}  holding out {Path(h).name}")
    print(f"   {np.mean([a for a, _ in zhunquelv]):5.2f}  mean")

    rf = RandomForestClassifier(n_estimators=400, random_state=0).fit(Xs, y)
    zhongyao = sorted(zip(keys, rf.feature_importances_), key=lambda kv: -kv[1])
    print("\nwhat the classifier uses (top 15), with each side's mean:")
    for k, v in zhongyao[:15]:
        i = keys.index(k)
        r, g = X[y == 1, i].mean(), X[y == 0, i].mean()
        print(f"   {v:.3f}  {k:22s} real {r:11.4f}   generated {g:11.4f}   "
              f"{'gen HIGHER' if g > r else 'gen lower'} x{(g/r if r else 0):.2f}")
    return keys, X, y


if __name__ == "__main__":
    main()
