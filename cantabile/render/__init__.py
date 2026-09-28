"""输出: MIDI 文件、渲染音频、母带处理。"""
# 这里的 master 是函数, 要模块就 import cantabile.render.master
from .synth import to_midi, SynthConfig, render, audible_end, read_wav
from .master import (MasterChain, WARM_PIANO, NEUTRAL, master, loudness,
                     excerpt, decode, SilentOutput)

__all__ = ["to_midi", "SynthConfig", "render", "audible_end", "read_wav",
           "MasterChain", "WARM_PIANO", "NEUTRAL", "master", "loudness",
           "excerpt", "decode", "SilentOutput"]
