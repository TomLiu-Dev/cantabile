"""编配模式: 曲子 + 和声 + 曲式 (Form) -> Performance。

默认曲式: 前奏, 然后每段歌词弹一遍, 不转调。
"""
from __future__ import annotations
import zlib
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .model import Note, Performance, TempoMap, Voice
from .theory import Chord, Harmony, chord_near, name, pc_near


def _verse_shape(verses):
    """每遍的相对强弱 (单位 dyn_step): 渐强, 倒数第二遍收一下, 最后一遍最响。"""
    if verses <= 1:
        return [0.0]
    if verses == 2:
        return [0.0, 1.6]
    qiangruo = []
    for v in range(verses):
        if v == verses - 1:
            qiangruo.append(3.2)
        elif v == verses - 2:
            qiangruo.append(-0.8)
        else:
            k = v / max(1, verses - 3)
            qiangruo.append(0.4 + 1.8 * min(1.0, k))
    return qiangruo


@dataclass
class Section:
    start_bar: int                 # 从第几小节开始 (1 起)
    tune_from: float = 0.0         # 曲子里的拍
    tune_to: float = None          # 不含; None 表示到结尾
    transpose: int = 0
    dyn: int = 56                  # 伴奏的基础力度
    rh_harmony: int = 2            # 旋律下面垫几个和弦音
    rich: bool = False             # 内声部多加点
    octave: bool = False           # 旋律高八度重复
    label: str = ""

    def span(self, tune_len: float) -> tuple:
        return self.tune_from, (tune_len if self.tune_to is None else self.tune_to)


@dataclass
class Form:
    sections: list
    beats_per_bar: int = 4
    coda: Optional[Callable] = None
    compound: bool = False

    @classmethod
    def strophic(cls, tune, *, verses: int = None, intro_bars: int = 4,
                 start_dyn: int = 58, dyn_step: int = 5,
                 intro_from: float = None, beats_per_bar: int = None) -> "Form":
        """前奏 + 每段一遍, 不转调。"""
        if beats_per_bar is None:
            beats_per_bar = getattr(tune, "beats_per_bar", 4) or 4
        verses = verses or tune.verses
        qiangruo = _verse_shape(verses)
        duanluo = []
        xiaojie = 1
        if intro_bars:
            lo = tune.length - intro_bars * beats_per_bar if intro_from is None else intro_from
            duanluo.append(Section(xiaojie, lo, lo + intro_bars * beats_per_bar,
                                   dyn=max(30, start_dyn - 4), rh_harmony=2,
                                   label="intro"))
            xiaojie += intro_bars
        yibian = int(tune.length // beats_per_bar)
        for v in range(verses):
            # 三遍以上时, 最响的那遍旋律弹八度
            duanluo.append(Section(xiaojie, 0.0, tune.length,
                                   dyn=int(round(start_dyn + dyn_step * qiangruo[v])),
                                   rh_harmony=2 if v < verses // 2 else 3,
                                   rich=qiangruo[v] >= 1.0,
                                   octave=verses > 2 and qiangruo[v] >= max(qiangruo) - 0.01,
                                   label=f"verse {v+1}"))
            xiaojie += yibian
        return cls(duanluo, beats_per_bar, compound=getattr(tune, "compound", False))

    @property
    def total_bars(self) -> int:
        zuihou = self.sections[-1]
        lo, hi = zuihou.span(1e9)
        return zuihou.start_bar + int((hi - lo) // self.beats_per_bar) - 1


@dataclass
class Piece:
    tune: object                   # corpus.Tune, 有 .melody .harmony .length 就行
    form: Form
    style: object = None           # arrange.styles.Style
    tempo: TempoMap = None
    title: str = ""
    varying: bool = True

    def build(self, *, seed: int = 0, repair: bool = True) -> Performance:
        from .arrange import styles as _styles
        from .arrange.voicing import melody_harmony
        style = self.style or _styles.get("accompaniment")
        hesheng = Harmony(self.tune.harmony)
        perf = Performance(tempo=self.tempo or TempoMap([(0, 90.0)]))
        perf.meta.update(title=self.title or getattr(self.tune, "title", ""),
                         tune=getattr(self.tune, "name", ""),
                         style=style.name, sections=[], repeats=[])
        shiji = {}  # 全曲的拍 -> 实际弹的和弦

        bpb = self.form.beats_per_bar
        for sec in self.form.sections:
            lo, hi = sec.span(self.tune.length)
            t0 = (sec.start_bar - 1) * bpb - lo
            perf.meta["sections"].append((sec.label, t0 + lo, t0 + hi))
            if abs(hi - lo - self.tune.length) < 1e-6:
                perf.meta["repeats"].append(t0)
            for kaishi, jieshu, hexian in hesheng.spans():
                a = max(kaishi, lo)
                if a < min(jieshu, hi):
                    shiji[t0 + a] = (Chord((hexian.root + sec.transpose) % 12,
                                           hexian.quality,
                                           (hexian.bass + sec.transpose) % 12)
                                     if sec.transpose else hexian)
            if self.varying:
                self._lay_varying(perf, hesheng, sec, t0, lo, hi, style)
            else:
                self._lay_accompaniment(perf, hesheng, sec, t0, lo, hi, style)
            self._lay_melody(perf, hesheng, sec, t0, lo, hi, style)
        if self.form.coda:
            self.form.coda(perf)
        perf.meta["harmony"] = Harmony(shiji)
        perf.sorted()
        jihao = getattr(self.tune, "marks", None)
        if jihao:
            # 谱上的力度记号对这一段所有的音都有效, 不只是旋律
            import bisect
            keys = sorted(jihao)
            fanwei = []
            for sec in self.form.sections:
                lo, hi = sec.span(self.tune.length)
                t0 = (sec.start_bar - 1) * bpb - lo
                fanwei.append((t0 + lo, t0 + hi, t0))
            for n in perf.notes:
                for a, b, t0 in fanwei:
                    if a - 1e-6 <= n.onset < b - 1e-6:
                        i = bisect.bisect_right(keys, n.onset - t0 + 1e-6) - 1
                        if i >= 0:
                            n.vel = max(1, min(127, n.vel + jihao[keys[i]]))
                        break
        # 空档要等所有小节都排好才找得到
        from .arrange.varying import connect_gaps
        perf.meta["connected"] = connect_gaps(
            perf, perf.meta["harmony"], ceiling=style.lh_ceiling)
        perf.sorted()
        if repair:
            perf = self.repair(perf, style)
        return perf.sorted()

    @staticmethod
    def separate_voices(perf: Performance, *, overlap: float = 0.02) -> Performance:
        """伴奏里同一个音高, 前一个音在后一个音之前 overlap 拍松开。旋律不动。"""
        from .model import Voice as _V
        an_yingao = {}
        for n in perf.notes:
            if n.voice is _V.MELODY:
                continue
            an_yingao.setdefault(n.pitch, []).append(n)
        jianduan = 0
        for zu in an_yingao.values():
            zu.sort(key=lambda n: n.onset)
            for a, b in zip(zu, zu[1:]):
                if a.onset + a.dur > b.onset - overlap:
                    xin = max(0.08, b.onset - overlap - a.onset)
                    if xin < a.dur:
                        a.dur = xin
                        jianduan += 1
        perf.meta["voice_separation_trimmed"] = jianduan
        return perf

    @staticmethod
    def enforce_register(perf: Performance, style) -> Performance:
        """伴奏超过 lh_ceiling 的音往下移八度 (resolve_clashes 可能把音推上去), 低出键盘就删。"""
        shangxian = style.lh_ceiling
        liuxia, yixia, shandiao = [], 0, 0
        for n in perf.notes:
            if n.voice is Voice.MELODY or not n.tag.startswith(
                    ("quaver", "vary", "broken", "block", "pad", "walk")):
                liuxia.append(n); continue
            if n.pitch <= shangxian:
                liuxia.append(n); continue
            p = n.pitch - 12
            while p > shangxian:
                p -= 12
            if p >= 21:
                n.pitch = p; yixia += 1; liuxia.append(n)
            else:
                shandiao += 1
        perf.notes = liuxia
        perf.meta["register_folded"] = yixia
        perf.meta["register_dropped"] = shandiao
        return perf

    @staticmethod
    def repair(perf: Performance, style=None) -> Performance:
        """修: 撞音 -> 音区 -> 保护旋律 -> 分开声部。"""
        from .audit.harmony import resolve_clashes
        from .audit.melody import protect_melody
        yuanlai = len(perf.notes)
        perf = resolve_clashes(perf, perf.meta.get("harmony"))
        if style is not None:
            perf = Piece.enforce_register(perf, style)
        perf = protect_melody(perf)
        perf = Piece.separate_voices(perf)
        perf.meta["repaired"] = yuanlai - len(perf.notes)
        return perf.sorted()

    def _right_hand(self, hesheng, sec, lo, hi, style) -> list:
        """右手: [(曲子里的拍, 音高, 时长, 角色)], 旋律 + 八度 + 下面垫的和弦音。"""
        from .arrange.voicing import melody_harmony
        youshou = []
        xuanlv = list(self.tune.melody)
        ruoqi = getattr(self.tune, "pickup", 0.0) or 0.0
        if ruoqi and sec is self.form.sections[-1] and hi >= self.tune.length - 1e-6:
            # 最后一遍不弹弱起, 最后一个音延长
            cut = self.tune.length - ruoqi
            xuanlv = [(b, p, d) for b, p, d in xuanlv if b < cut - 1e-6]
            b, p, d = xuanlv[-1]
            xuanlv[-1] = (b, p, d + ruoqi)
        for pai, yingao, shichang in xuanlv:
            if not (lo <= pai < hi):
                continue
            mel = yingao + sec.transpose
            youshou.append((pai, mel, shichang, "melody"))
            if sec.octave and not self._octave_would_clash(mel, pai, shichang):
                youshou.append((pai, mel + 12, shichang, "melody.octave"))
            if shichang < 0.5 or (shichang < 0.75 and not sec.rich):
                continue
            ch = hesheng.at(pai)
            if sec.transpose:
                ch = Chord((ch.root + sec.transpose) % 12, ch.quality,
                           (ch.bass + sec.transpose) % 12)
            xiamian = melody_harmony(mel, ch, min(sec.rh_harmony, 2),
                                     gap_below_melody=5, lowest=mel - 9,
                                     expose_below=style.expose_melody_below)[
                                     :1 if shichang < 1.0 else 2]
            for i, p in enumerate(xiamian):
                if p is not None:
                    youshou.append((pai, p, shichang, f"melody.harmony.{i}"))
        return youshou

    def _lay_varying(self, perf, hesheng, sec, t0, lo, hi, style):
        """按旋律一小节一小节地排伴奏。"""
        from .arrange.varying import plan_bars, render_bar
        bpb = self.form.beats_per_bar
        nbars = int((hi - lo) // bpb)
        phr = None
        try:
            from .analyze.structure import phrases
            phr = phrases([(b, p, d) for b, p, d in self.tune.melody])
        except Exception:
            pass
        mel = [(b - lo, p, d) for b, p, d in self.tune.melody if lo <= b < hi]
        shangyige = [None]
        youshou = self._right_hand(hesheng, sec, lo, hi, style)
        jihua = plan_bars(mel, hesheng.shifted(-lo), bars=nbars,
                          beats_per_bar=bpb, phrases=phr,
                          # 用 crc32 不用 hash(), hash 每个进程不一样
                          seed=zlib.crc32(f"{sec.label}|{sec.start_bar}".encode()) & 0xffff,
                          compound=self.form.compound)
        secs = self.form.sections
        nxt = secs[secs.index(sec) + 1] if sec in secs[:-1] else None
        if jihua and (sec.label == "intro"
                      or (nxt is not None and nxt.dyn - sec.dyn >= 6)):
            # 前奏结尾, 还有下一遍大声很多时这一遍的结尾: 和弦持续,
            # 不然伴奏会盖住旋律最后那个在衰减的长音
            jihua[-1].figure, jihua[-1].second = "sustained", ()
            jihua[-1].reasons.append("end of section: hold the chord")
        for pl in jihua:
            at = lo + pl.bar * bpb
            banxiaojie = [hesheng.at(at + h * 2.0)
                          for h in range(int(bpb // 2))]
            if sec.transpose:
                banxiaojie = [Chord((c.root + sec.transpose) % 12, c.quality,
                                    (c.bass + sec.transpose) % 12) for c in banxiaojie]
            hexian = banxiaojie
            bianhua = []
            for b, _, c in hesheng.spans():
                if at <= b < at + bpb:
                    if sec.transpose:
                        c = Chord((c.root + sec.transpose) % 12, c.quality,
                                  (c.bass + sec.transpose) % 12)
                    bianhua.append((b - at, c))
            perf.add(*render_bar(banxiaojie, pl, t0 + at, dyn=sec.dyn,
                                 beats_per_bar=bpb, ceiling=style.lh_ceiling,
                                 melody=youshou, mel_offset=t0,
                                 changes=bianhua, compound=self.form.compound))
            # 和弦变了才换踏板, 不是每小节都换
            if not perf.pedal or hesheng.at(at) is not shangyige[0]:
                perf.pedal.append((t0 + at, 1))
                shangyige[0] = hesheng.at(at)
            mid = at + bpb / 2
            if hesheng.at(mid) is not shangyige[0]:
                perf.pedal.append((t0 + mid, 1))
                shangyige[0] = hesheng.at(mid)
        perf.meta.setdefault("bar_plans", []).extend(jihua)

    def _lay_accompaniment(self, perf, hesheng, sec, t0, lo, hi, style):
        """每个和弦一个伴奏型, 每个和弦换一次踏板。"""
        bpb = self.form.beats_per_bar
        for kaishi, jieshu, hexian in hesheng.spans():
            a, b = max(kaishi, lo), min(jieshu, hi)
            if a >= b:
                continue
            if sec.transpose:
                hexian = Chord((hexian.root + sec.transpose) % 12, hexian.quality,
                               (hexian.bass + sec.transpose) % 12)
            perf.add(*style.render(hexian, t0 + a, b - a, sec.dyn,
                                   bar_origin=t0, rich=sec.rich))
            perf.pedal.append((t0 + a, 1))

    def _octave_would_clash(self, mel: int, beat: float, dur: float) -> bool:
        from .theory import is_clash
        top = mel + 12
        for b, p, d in self.tune.melody:
            if b >= beat + dur * 1.2 or b + d * 1.2 <= beat:
                continue
            if p != mel and is_clash(p, top):
                return True
        return False

    def _lay_melody(self, perf, hesheng, sec, t0, lo, hi, style):
        """把右手的音放进 perf。这里只给长音加一点力度, 旋律的强弱后面演奏时再处理。"""
        an_pai = {}
        for pai, p, shichang, juese in self._right_hand(hesheng, sec, lo, hi, style):
            an_pai.setdefault((pai, shichang), []).append((p, juese))
        for (pai, shichang), yinfu in sorted(an_pai.items()):
            t = t0 + pai
            # 长音弹重一点, 才撑得住
            lidu = int(sec.dyn + min(9.0, 3.4 * max(0.0, shichang - 1.0)))
            mel = next((p for p, r in yinfu if r == "melody"), None)
            # 旋律音稍微连到下一个音 (legato), 下一个音是半音时不连
            over = min(1.12, 1.0 + 0.12 / shichang)
            if mel is not None:
                for b2, p2, _ in self.tune.melody:
                    if pai < b2 < pai + shichang * 1.2 and \
                            abs(p2 + sec.transpose - mel) in (1, 11, 13):
                        over = min(over, max(0.9, (b2 - pai) / shichang))
                        break
            for p, juese in yinfu:
                if juese == "melody":
                    perf.add(Note(t, shichang * over, p, lidu, Voice.MELODY, "melody"))
                elif juese == "melody.octave":
                    # 八度早点放, 不要响到下一个音里去
                    perf.add(Note(t, shichang * 0.94, p, lidu - 9, Voice.MELODY,
                                  "melody.octave"))
                else:
                    i = int(juese.rsplit(".", 1)[1])
                    perf.add(Note(t, shichang * .94, p, lidu - 4 - i * 3,
                                  Voice.INNER, "melody.harmony"))


def default_tempo(bpm: float, length: float, *, phrase: float = 16.0,
                  ease: float = 0.055, lift: float = 0.034,
                  final_rit: float = 0.30, beats_per_bar: float = None,
                  factor=None) -> TempoMap:
    """速度曲线: 每个乐句中间快一点 (lift), 收尾慢一点 (ease), 最后十拍渐慢。

    factor 是每拍再乘的系数 (副歌速度, 延长号)。录音里 10%-90% 分位差不多差 12-13 bpm。
    beats_per_bar 给了的话一句按四小节算, 不然按 phrase 拍。
    """
    import math
    if beats_per_bar:
        juchang = 4.0 * beats_per_bar
    else:
        juchang = phrase

    def shape(beat, base):
        ph = (beat % juchang) / juchang
        # 一句一个周期, 大概三分之一处最快
        arch = math.cos(2.0 * math.pi * (ph - 0.30))
        t = base * (1.0 + lift * arch)
        # 进终止式再慢一点
        t *= 1.0 - ease * max(0.0, (ph - 0.72) / 0.28) ** 2
        if factor is not None:
            t *= factor(beat)
        weiba = length - 10.0
        if beat >= weiba:
            t *= 1 - final_rit * min(1.0, (beat - weiba) / 10.0) ** 1.2
        return t

    return TempoMap([(0.0, bpm)], shape=shape)
