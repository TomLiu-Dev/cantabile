"""最简单的 SFZ 采样器。

读 region (键和力度范围), 每个音挑一个采样, 变调, 加个简单的制音包络, 叠起来。
多力度层的 SFZ 音色会随力度变, 每键一个采样的 SoundFont 做不到。没有滤波、LFO、轮换。
"""
from __future__ import annotations
import math, re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Region:
    sample: str
    lokey: int = 0
    hikey: int = 127
    pitch_keycenter: int = 60
    lovel: int = 1
    hivel: int = 127
    volume: float = 0.0            # dB
    tune: float = 0.0              # cents
    offset: int = 0
    trigger: str = "attack"

    def matches(self, key: int, vel: int) -> bool:
        return (self.lokey <= key <= self.hikey
                and self.lovel <= vel <= self.hivel
                and self.trigger == "attack")


_NUM = {"lokey", "hikey", "pitch_keycenter", "lovel", "hivel", "offset"}
_FLT = {"volume", "tune"}
_NOTE = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def _key(v: str) -> int:
    v = v.strip()
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    m = re.fullmatch(r"([a-gA-G])([#b]?)(-?\d+)", v)
    if not m:
        raise ValueError(f"bad key {v!r}")
    p = _NOTE[m.group(1).lower()] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)
    return p + (int(m.group(3)) + 1) * 12


def parse(path) -> list:
    """读 SFZ 的 region, <group> 里的设置会继承下来。"""
    path = Path(path)
    wenben = re.sub(r"//.*", "", path.read_text(errors="ignore"))
    duan = re.split(r"<(group|region|global|control|master)>", wenben)
    quyu, zu, quanju = [], {}, {}
    for i in range(1, len(duan), 2):
        kind, body = duan[i], duan[i + 1]
        opts = dict(re.findall(r"(\w+)=([^\s]+)", body))
        if kind in ("global", "control"):
            quanju.update(opts)
        elif kind in ("group", "master"):
            zu = dict(quanju); zu.update(opts)
        elif kind == "region":
            o = dict(zu); o.update(opts)
            if "sample" not in o:
                continue
            kw = {}
            for k, v in o.items():
                if k in _NUM:
                    kw[k] = _key(v) if k in ("lokey", "hikey", "pitch_keycenter") else int(v)
                elif k in _FLT:
                    kw[k] = float(v)
                elif k in ("sample", "trigger"):
                    kw[k] = v
            quyu.append(Region(**kw))
    return quyu


@dataclass
class SFZSampler:
    sfz: str
    sr: int = 44100
    release: float = 0.28          # 松键后制音器落下的秒数
    pedal_release: float = 1.2     # 没用到, 留着兼容
    gain: float = 1.0
    regions: list = field(default_factory=list)
    _cache: dict = field(default_factory=dict)
    root: Path = None

    def __post_init__(self):
        self.root = Path(self.sfz).parent
        self.regions = parse(self.sfz)
        if not self.regions:
            raise ValueError(f"no regions in {self.sfz}")

    def layers(self) -> dict:
        lov = sorted({r.lovel for r in self.regions})
        keys = sorted({r.pitch_keycenter for r in self.regions})
        return {"regions": len(self.regions), "velocity_layers": len(lov),
                "lovel": lov, "sampled_keys": len(keys),
                "key_range": (min(keys), max(keys)) if keys else None}

    def _load(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        import soundfile as sf
        p = self.root / name.replace("\\", "/")
        y, sr = sf.read(str(p), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if sr != self.sr:
            n = int(len(y) * self.sr / sr)
            y = np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)
        self._cache[name] = y
        return y

    def pick(self, key: int, vel: int):
        houxuan = [r for r in self.regions if r.matches(key, vel)]
        if not houxuan:
            houxuan = [r for r in self.regions
                       if r.lokey <= key <= r.hikey and r.trigger == "attack"]
        if not houxuan:
            return None
        # 先找最近的采样键, 再找力度落在哪一层
        return min(houxuan, key=lambda r: (abs(r.pitch_keycenter - key),
                                         abs((r.lovel + r.hivel) / 2 - vel)))

    TARGET_GAMMA = 2.2   # 力度 -> 幅度的指数 (40 到 100 差不多 17.5 dB)

    def _level_curve(self, key: int = 60) -> "np.ndarray":
        """这个采样库自己每个力度的峰值电平 (dB)。"""
        if getattr(self, "_curve", None) is not None:
            return self._curve
        quxian = np.zeros(128)
        for v in range(1, 128):
            r = self.pick(key, v)
            if r is None:
                continue
            y = self._load(r.sample)
            lo, hi = r.lovel, max(r.hivel, r.lovel + 1)
            a = 10.0 ** (r.volume / 20.0) * (0.72 + 0.45 * (v - lo) / (hi - lo))
            quxian[v] = 20.0 * np.log10(max(1e-9, float(np.abs(y[:44100]).max()) * a))
        quxian[0] = quxian[1]
        self._curve = quxian
        return quxian

    def _level_correction(self, vel: int) -> float:
        """把采样库的力度响应校正到固定的曲线上。

        力度层只管选哪个采样, 不保证越重越响, 实际上范围很窄还不单调。
        这里只改音量, 选哪个采样 (也就是音色) 不动。
        """
        v = max(1, min(127, int(vel)))
        cur = self._level_curve()
        ref = cur[100]                      # 以 100 为基准, 这里采样库还正常
        want = ref + 20.0 * self.TARGET_GAMMA * math.log10(v / 100.0)
        return float(10.0 ** ((want - cur[v]) / 20.0))

    def note(self, key: int, vel: int, dur: float, *, pedal_hold: float = 0.0):
        r = self.pick(key, vel)
        if r is None:
            return np.zeros(1, dtype=np.float32)
        y = self._load(r.sample)
        banyin = key - r.pitch_keycenter + r.tune / 100.0
        if abs(banyin) > 1e-6:
            bili = 2.0 ** (banyin / 12.0)
            n = int(len(y) / bili)
            y = np.interp(np.arange(n) * bili, np.arange(len(y)), y).astype(np.float32)
        fudu = 10.0 ** (r.volume / 20.0)
        # 层内力度改音量, 跨层才改音色
        lo, hi = r.lovel, max(r.hivel, r.lovel + 1)
        cengnei = (vel - lo) / (hi - lo)
        fudu *= 0.72 + 0.45 * cengnei
        fudu *= self._level_correction(vel)
        zong = dur + max(self.release, pedal_hold)
        n = min(len(y), int(zong * self.sr))
        shuchu = y[:n].astype(np.float32) * fudu
        # 制音: 音符期间保持, 之后落下
        luoxia = max(self.release, pedal_hold)
        k = int(dur * self.sr)
        if k < n:
            weiba = n - k
            env = np.exp(-np.arange(weiba) / max(1.0, luoxia * self.sr * 0.35))
            shuchu[k:] *= env.astype(np.float32)
        return shuchu

    def render(self, perf, *, progress=False) -> np.ndarray:
        """渲染成单声道 float32。音一直响到松键; 松键时踏板踩着就响到踏板抬起, 然后制音。"""
        sudu = perf.tempo
        # 踏板踩下的区间 (秒): [(踩下, 抬起), ...]
        ev = sorted(perf.pedal)
        taban, caizhe, t_cai = [], False, 0.0
        for b, v in ev:
            t = sudu.seconds(b)
            if v >= 64 and not caizhe:
                caizhe, t_cai = True, t
            elif v < 64 and caizhe:
                caizhe = False
                taban.append((t_cai, t))
        if caizhe:
            taban.append((t_cai, sudu.seconds(perf.length) + 6.0))
        kaishi = [a for a, _ in taban]

        def damper_off(t):
            """t 时刻松键, 制音器真正落下的时间。"""
            import bisect
            i = bisect.bisect_right(kaishi, t) - 1
            if i >= 0 and taban[i][0] <= t < taban[i][1]:
                return taban[i][1]
            return t

        jiewei = sudu.seconds(perf.length) + 6.0
        buf = np.zeros(int(jiewei * self.sr) + self.sr, dtype=np.float32)
        for i, nt in enumerate(sorted(perf.notes, key=lambda n: n.onset)):
            t0 = sudu.seconds(nt.onset)
            songjian = max(t0 + 0.02, sudu.seconds(nt.end))
            off = damper_off(songjian)
            seg = self.note(nt.pitch, int(nt.vel), off - t0)
            a = int(t0 * self.sr)
            b = min(len(buf), a + len(seg))
            if b > a:
                buf[a:b] += seg[: b - a]
            if progress and i % 500 == 0:
                print(f"   {i}/{len(perf.notes)}", flush=True)
        m = float(np.abs(buf).max())
        if m > 0:
            buf *= (0.89 * self.gain) / m
        return buf


def measure(sampler: SFZSampler, *, pitch: int = 60,
            velocities=(30, 50, 70, 90, 110, 127)) -> dict:
    """不同力度下的频谱重心、2k 以上占比、峰值。"""
    cen, hi, amp = [], [], []
    for v in velocities:
        y = sampler.note(pitch, v, 1.0)[: int(0.5 * sampler.sr)]
        if len(y) < 256:
            continue
        S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
        f = np.fft.rfftfreq(len(y), 1 / sampler.sr)
        cen.append(float((S * f).sum() / max(S.sum(), 1e-12)))
        hi.append(float(S[f > 2000].sum() / max(S.sum(), 1e-12)))
        amp.append(float(np.abs(y).max()))
    return {"velocities": list(velocities), "centroid_hz": cen,
            "above_2k": hi, "peak": amp,
            "centroid_ratio": cen[-1] / cen[0] if cen and cen[0] else 0,
            "hf_ratio": hi[-1] / hi[0] if hi and hi[0] else 0,
            "amp_ratio": amp[-1] / amp[0] if amp and amp[0] else 0,
            "monotonic": all(b >= a * 0.95 for a, b in zip(cen, cen[1:]))}
