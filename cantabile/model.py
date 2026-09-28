"""基本数据结构。每个音带 voice 和 tag, 检查出问题时知道是哪一步生成的。"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Iterable, Sequence


class Voice(Enum):
    MELODY = "melody"    # 旋律, 一定要听得见
    BASS   = "bass"      # 低音, 不能删
    INNER  = "inner"     # 中间的和声
    COLOUR = "colour"    # 装饰, 有冲突先删它

    @property
    def droppable(self) -> bool:
        return self in (Voice.INNER, Voice.COLOUR)


@dataclass
class Note:
    onset: float                 # 拍
    dur:   float                 # 拍
    pitch: int                   # MIDI 音高
    vel:   int                   # 1..127
    voice: Voice = Voice.INNER
    tag:   str   = ""            # 哪一步生成的

    @property
    def end(self) -> float:
        return self.onset + self.dur

    def overlap(self, other: "Note") -> float:
        return max(0.0, min(self.end, other.end) - max(self.onset, other.onset))

    def shifted(self, dt: float) -> "Note":
        return replace(self, onset=self.onset + dt)


@dataclass
class TempoMap:
    """拍 -> BPM, 分段线性, shape 可以再调。"""
    anchors: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 90.0)])
    shape: Callable[[float, float], float] | None = None   # (beat, bpm) -> bpm

    def bpm(self, beat: float) -> float:
        a = self.anchors
        if beat <= a[0][0]:
            base = a[0][1]
        elif beat >= a[-1][0]:
            base = a[-1][1]
        else:
            for (b0, v0), (b1, v1) in zip(a, a[1:]):
                if b0 <= beat <= b1:
                    t = 0.0 if b1 == b0 else (beat - b0) / (b1 - b0)
                    base = v0 + (v1 - v0) * t
                    break
        return max(20.0, self.shape(beat, base) if self.shape else base)

    def seconds(self, beat: float, step: float = 0.25) -> float:
        t, b = 0.0, 0.0
        while b + step <= beat:
            t += step * 60.0 / self.bpm(b + step / 2)
            b += step
        if beat > b:
            t += (beat - b) * 60.0 / self.bpm((b + beat) / 2)
        return t


@dataclass
class Performance:
    notes: list[Note] = field(default_factory=list)
    pedal: list[tuple[float, int]] = field(default_factory=list)   # (beat, cc64)
    tempo: TempoMap = field(default_factory=TempoMap)
    meta:  dict = field(default_factory=dict)

    def add(self, *notes: Note) -> "Performance":
        self.notes.extend(notes); return self

    def of(self, *voices: Voice) -> list[Note]:
        s = set(voices)
        return [n for n in self.notes if n.voice in s]

    def melody(self) -> list[Note]:
        return sorted(self.of(Voice.MELODY), key=lambda n: n.onset)

    def accompaniment(self) -> list[Note]:
        return [n for n in self.notes if n.voice is not Voice.MELODY]

    def sounding(self, beat: float) -> list[Note]:
        return [n for n in self.notes if n.onset <= beat < n.end]

    def melody_at(self, beat: float) -> Note | None:
        """这一拍正在响的最高旋律音 (包括八度)。"""
        c = [n for n in self.melody() if n.onset <= beat < n.end]
        return max(c, key=lambda n: n.pitch) if c else None

    @property
    def length(self) -> float:
        return max((n.end for n in self.notes), default=0.0)

    def sorted(self) -> "Performance":
        self.notes.sort(key=lambda n: (n.onset, n.pitch)); return self

    def copy(self) -> "Performance":
        return Performance([replace(n) for n in self.notes],
                           list(self.pedal), self.tempo, dict(self.meta))


@dataclass
class Finding:
    check: str
    beat: float
    detail: str
    severity: str = "warn"       # "error" | "warn" | "info"
    notes: tuple = ()

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check} @ beat {self.beat:.2f}: {self.detail}"


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, *f: Finding) -> "Report":
        self.findings.extend(f); return self

    def merge(self, other: "Report") -> "Report":
        self.findings.extend(other.findings)
        self.stats.update(other.stats)
        return self

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    def ok(self) -> bool:
        return not self.errors

    def summary(self, limit: int = 12) -> str:
        hang = []
        for k, v in self.stats.items():
            hang.append(f"  {k}: {v}")
        fenzu = {}
        for f in self.findings:
            fenzu.setdefault((f.severity, f.check), []).append(f)
        for (sev, chk), fs in sorted(fenzu.items()):
            hang.append(f"  {sev.upper():5s} {chk}: {len(fs)}")
            for f in fs[:limit]:
                hang.append(f"        beat {f.beat:8.2f}  {f.detail}")
            if len(fs) > limit:
                hang.append(f"        ... and {len(fs)-limit} more")
        return "\n".join(hang) or "  (clean)"


Pass = Callable[[Performance], Performance]


def run(perf: Performance, passes: Sequence[Pass]) -> Performance:
    for p in passes:
        perf = p(perf)
    return perf
