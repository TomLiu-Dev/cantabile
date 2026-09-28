"""曲库: 旋律 + 和弦, 给编配模式用。

仓库里不带曲子。自己的曲子放在 ~/.cantabile/corpus/ (或者环境变量
CANTABILE_CORPUS 指定的目录) 下面的 .py 文件里, 文件里写一个 TUNES 字典就行,
格式见下面的 Tune。
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Tune:
    name: str              # 曲调名
    title: str             # 歌词第一句
    author: str            # 作词
    composer: str          # 作曲
    year: str
    key: str
    meter: str             # 歌词的音节格式, 比如 "8.7.8.5.D"
    beats_per_bar: int
    bars: int
    melody: list           # [(拍, 音高, 时长)]
    harmony: dict          # {拍: (根音, 性质, 低音)}
    verses: int            # 歌词有几段, 也就是弹几遍
    form: str              # 比如 "verse+chorus"
    chorus_at: float = None   # 副歌从第几拍开始
    verified: str = "none"
    source: str = ""
    notes: str = ""
    compound: bool = False    # 6/8 这类复拍子, beats_per_bar 按四分音符算
    refrain_tempo: float = 1.0   # 副歌速度是主歌的几倍
    # 弱起放在曲子末尾, 这样每一遍都是整小节; 最后一遍不弹它
    pickup: float = 0.0
    # 谱上的力度记号和延长号。marks: {拍: 力度加减, 一直到下一个记号}
    # fermatas: [(拍, 时长)]
    marks: dict = field(default_factory=dict)
    fermatas: tuple = ()

    @property
    def length(self) -> float:
        return self.bars * self.beats_per_bar

    def span(self, a: float, b: float) -> list:
        return [(t, p, d) for t, p, d in self.melody if a <= t < b]


TUNES = {}


def _jiazai():
    mulu = Path(os.environ.get("CANTABILE_CORPUS", Path.home() / ".cantabile" / "corpus"))
    if not mulu.is_dir():
        return
    for wenjian in sorted(mulu.glob("*.py")):
        spec = importlib.util.spec_from_file_location("_cantabile_qu_" + wenjian.stem, wenjian)
        mokuai = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mokuai)
        TUNES.update(getattr(mokuai, "TUNES", {}))


def get(name: str) -> Tune:
    if not TUNES:
        _jiazai()
    key = name.upper()
    if key in TUNES:
        return TUNES[key]
    for t in TUNES.values():
        if name.lower() in t.title.lower():
            return t
    raise KeyError(f"没有这首曲子: {name!r}; 现有 {sorted(TUNES)}")


def names() -> list:
    if not TUNES:
        _jiazai()
    return sorted(TUNES)
