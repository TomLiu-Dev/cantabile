"""房间混响 (卷积)。

指数衰减的扩散声场, 有预延迟, 尾巴越衰减越暗。放在母带处理里动态和限幅之间
(MasterChain.room_wet)。默认关: 合成尾巴的中低频音色不对, 现在加了反而不像录音。
apply() 什么脉冲响应都能用, 有实测的 IR 可以直接换。
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

__all__ = ["RoomConfig", "SMALL_HALL", "apply"]


@dataclass
class RoomConfig:
    rt60: float = 1.1        # 衰减到 -60 dB 的秒数, 中等大小的厅
    pre_delay: float = 0.012 # 直达声比房间声早这么多
    wet: float = 0.28        # 房间声占多少
    damping: float = 0.25
    seed: int = 3            # 房间固定, 每次渲染一样


SMALL_HALL = RoomConfig()


def impulse(cfg: RoomConfig, sr: int) -> np.ndarray:
    """能量归一的脉冲响应。分频段各自衰减, 高频比低频死得快。"""
    rng = np.random.default_rng(cfg.seed)
    n = max(1, int(cfg.rt60 * 1.2 * sr))
    t = np.arange(n) / sr
    zaosheng = rng.standard_normal(n)
    pu = np.fft.rfft(zaosheng)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    # (下限, 上限, RT60 倍数, 电平)
    pinduan = ((0, 250, 1.15, 1.0), (250, 1000, 1.0, 0.9),
               (1000, 3000, 0.7, 0.6), (3000, sr / 2, 0.35, 0.3))
    ir = np.zeros(n)
    for lo, hi, k, g in pinduan:
        m = (f >= lo) & (f < hi)
        bufen = np.fft.irfft(np.where(m, pu, 0), n)
        ir += g * bufen * np.exp(-6.9078 * t / (cfg.rt60 * k))
    ir[:int(cfg.pre_delay * sr)] = 0.0
    e = np.sqrt((ir ** 2).sum())
    return ir / e if e else ir


def apply(y: np.ndarray, sr: int = 44100, cfg: RoomConfig = SMALL_HALL) -> np.ndarray:
    """干声和卷积后的湿声混合。湿声先调到和干声一样的 RMS, 所以 wet 跟 rt60 无关。"""
    if cfg.wet <= 0:
        return y
    ir = impulse(cfg, sr)
    # 用 FFT 卷积, 整首直接卷太慢
    m = len(y) + len(ir) - 1
    k = 1 << (m - 1).bit_length()
    shi = np.fft.irfft(np.fft.rfft(y, k) * np.fft.rfft(ir, k), k)[:len(y)]
    rms_gan = np.sqrt(np.mean(y ** 2)) if len(y) else 0.0
    rms_shi = np.sqrt(np.mean(shi ** 2)) if len(shi) else 0.0
    if rms_shi > 1e-12:
        shi *= rms_gan / rms_shi
    return (1.0 - cfg.wet) * y + cfg.wet * shi
