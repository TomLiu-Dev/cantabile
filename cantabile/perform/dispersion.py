"""力度离散: motor_noise 是手的随机误差, intentional 是有意的变化.
参考: RenCon 2025
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..model import Performance, Voice
from ..theory import Harmony


@dataclass
class Dispersion:
    motor_sd: float = 4.5
    motor_rho: float = 0.45        # 相邻音偏差的相关
    harmony_accent: float = 4.0
    contour: float = 3.0
    offbeat_give: float = 2.5
    spread: float = 1.0


def motor_noise(perf: Performance, d: Dispersion = None,
                seed: int = 0) -> Performance:
    """每个声部各自的 AR(1) 力度噪声."""
    d = d or Dispersion()
    rnd = random.Random(seed)
    out = perf.copy()
    zhuangtai = {}
    for n in sorted(out.notes, key=lambda x: x.onset):
        k = n.voice
        shangci = zhuangtai.get(k, 0.0)
        pianyi = (d.motor_rho * shangci
                  + math.sqrt(1 - d.motor_rho ** 2) * rnd.gauss(0, d.motor_sd))
        zhuangtai[k] = pianyi
        n.vel = int(max(1, min(127, round(n.vel + pianyi))))
    return out


def intentional(perf: Performance, harmony: Harmony = None,
                d: Dispersion = None, beats_per_bar: float = 4.0) -> Performance:
    d = d or Dispersion()
    out = perf.copy()
    harmony = harmony or out.meta.get("harmony")
    shang_hexian, shang_diyin = None, None
    for n in sorted(out.notes, key=lambda x: (x.onset, x.pitch)):
        jia = 0.0
        if harmony is not None:
            ch = harmony.at(n.onset)
            if ch is not shang_hexian and shang_hexian is not None:
                jia += d.harmony_accent
            shang_hexian = ch
        if n.voice is Voice.BASS:
            if shang_diyin is not None:
                jia += d.contour * (1.0 if n.pitch > shang_diyin else
                                    -0.6 if n.pitch < shang_diyin else 0.0) / 3.0
            shang_diyin = n.pitch
        weizhi = n.onset % beats_per_bar
        if abs(weizhi - round(weizhi)) > 0.1:
            jia -= d.offbeat_give
        n.vel = int(max(1, min(127, round(n.vel + jia))))
    return out


def widen(perf: Performance, factor: float = 1.6, *, pivot: str = "median",
          lo: int = 20, hi: int = 108) -> Performance:
    out = perf.copy()
    vs = sorted(n.vel for n in out.notes)
    if not vs:
        return out
    c = vs[len(vs) // 2] if pivot == "median" else sum(vs) / len(vs)
    for n in out.notes:
        n.vel = int(max(lo, min(hi, round(c + (n.vel - c) * factor))))
    return out


# 两段参考录音的力度: sd 16.9, p10 43, 中位 63, p90 89, IQR 29
REFERENCE_PROFILE = [
    26, 33, 36, 36, 36, 36, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38,
    38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40,
    40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41,
    41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
    42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43,
    43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44,
    44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45,
    45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45,
    45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46,
    46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47,
    47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48,
    48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49,
    49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50,
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51,
    51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52,
    52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53,
    53, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54,
    54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 55, 55,
    55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
    55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57,
    57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60,
    60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61,
    61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 62,
    62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
    64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65,
    65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66,
    67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 68, 68,
    68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69,
    69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 70, 70, 70, 70,
    70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71,
    71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72, 72,
    72, 72, 72, 72, 72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73,
    73, 73, 73, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 75, 75,
    75, 75, 75, 75, 75, 75, 75, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76,
    76, 76, 76, 76, 76, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77,
    77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78,
    78, 78, 78, 78, 78, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79,
    79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 80, 80, 80, 80, 80, 80, 80, 80,
    80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 81, 81, 81, 81, 81, 81, 81, 81, 81,
    81, 81, 81, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 83,
    83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 84, 84, 84, 84, 84,
    84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 85, 85, 85, 85, 85, 85,
    85, 85, 85, 85, 85, 85, 85, 85, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86,
    86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 87, 87, 87, 87, 87, 87, 87,
    87, 87, 87, 87, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88,
    88, 88, 88, 88, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89,
    89, 89, 89, 89, 90, 90, 90, 90, 90, 90, 90, 91, 91, 91, 91, 91, 91, 91, 91,
    91, 91, 91, 91, 91, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92,
    93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93,
    93, 93, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 95, 95, 95, 95, 95,
    95, 95, 95, 95, 96, 96, 96, 96, 97, 97, 97, 97, 97, 98, 99, 99, 100, 106
]
REFERENCE_VELOCITIES = REFERENCE_PROFILE


def match_distribution(perf: Performance, reference, *, strength: float = 1.0,
                       voices=None) -> Performance:
    """按分位数把力度映射到参考分布, 保持排名. strength 0~1."""
    out = perf.copy()
    yinfu = [n for n in out.notes if voices is None or n.voice in voices]
    if not yinfu or not reference:
        return out
    ref = sorted(reference)
    shunxu = sorted(range(len(yinfu)), key=lambda i: (yinfu[i].vel, i))
    m = len(yinfu)
    for paiming, i in enumerate(shunxu):
        q = (paiming + 0.5) / m
        mubiao = ref[min(len(ref) - 1, int(q * len(ref)))]
        n = yinfu[i]
        n.vel = int(max(1, min(127, round(n.vel + strength * (mubiao - n.vel)))))
    return out


def reference_velocities(*paths) -> list:
    import mido
    out = []
    for p in paths:
        m = mido.MidiFile(str(p))
        out += [x.velocity for tr in m.tracks for x in tr
                if x.type == "note_on" and x.velocity > 0]
    return out


def report(perf: Performance) -> dict:
    import statistics as st
    from collections import Counter
    v = [n.vel for n in perf.notes]
    q = sorted(v)
    n = len(q)
    return {"n": n, "sd": round(st.pstdev(v), 1),
            "p10": q[n // 10], "p90": q[9 * n // 10],
            "iqr": q[3 * n // 4] - q[n // 4],
            "distinct": len(set(v)),
            "modal_share": round(Counter(v).most_common(1)[0][1] / n, 3)}
