"""读谱: MIDI 读取和看图读谱, 两边不共用代码, sheet.cross_check 才能拿来互相核对。

sheet 要 Pillow, 所以用到时才导入。
"""
import importlib

from .midi import (load, extract_soprano, extract_harmony, quantize,
                   pedal_events, bar_lines, as_names)

_SHEET = ("render_pdf", "load_image", "find_staves", "find_noteheads",
          "find_accidentals", "pitch_at", "read_melody", "find_barlines",
          "key_signature", "cross_check")

__all__ = ["load", "extract_soprano", "extract_harmony", "quantize",
           "pedal_events", "bar_lines", "as_names", "sheet"] + list(_SHEET)


def __getattr__(name):
    if name == "sheet" or name in _SHEET:
        # 不能 from . import sheet, 会再进这个 __getattr__ 死循环
        _sheet = importlib.import_module(".sheet", __name__)
        return _sheet if name == "sheet" else getattr(_sheet, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
