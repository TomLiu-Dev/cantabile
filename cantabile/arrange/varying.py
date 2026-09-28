# -*- coding: utf-8 -*-
"""左手伴奏每小节有变化: 按乐句位置/和声/旋律疏密选音型, 音型会保持几小节再换。"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..model import Note, Voice
from ..theory import chord_near
from .voicing import open_left_hand


# (格内偏移, 声部编号, 力度差); 声部: 0 低音 1 五度 2 八度 3/4 色彩音 5 低八度
# 元组表示一起弹。二拍子一格半小节, 三拍子和复拍子一格一小节。
FIGURES_DUPLE = {
    "block":    [(0.0, (0, 5), 0), (1.0, (1, 2), -3), (1.5, 3, -4)],
    "plain":      [(0.0, 0, 0), (0.5, (2, 3), -4), (1.0, (1, 2), -3)],
    "open":       [(0.0, (0, 5, 2), 0), (1.0, (1, 3), -3)],
    "rocking":    [(0.0, 0, 0), (1.0, (1, 2), -3), (1.5, 3, -4)],
    "walking":    [(0.0, 0, 0), (0.5, 3, -4), (1.0, (1, 2), -3), (1.5, 1, -4)],
    "sustained":  [(0.0, (0, 5, 2), 0), (0.5, 3, -4), (1.0, (1, 2), -4)],
    "arpeggio":   [(0.0, 0, 0), (0.5, 1, -3), (1.0, (2, 3), -2), (1.5, 1, -4)],
    "cadential":  [(0.0, (0, 5, 2), 0), (1.0, (1, 2), -3), (1.5, 3, -4)],
    "thin":       [(0.0, 0, 0), (0.5, 1, -4), (1.0, (1, 2), -3)],
    "afterbeat":  [(0.0, 0, 0), (0.5, (1, 2), -2), (1.5, (2, 3), -4)],
    "double":     [(0.0, (0, 5, 1), 0), (1.0, (2, 3), -2)],
    "syncopated": [(0.0, 0, 0), (0.75, (2, 3), -2), (1.5, (1, 2), -3)],
    "driving":    [(0.0, 0, 0), (0.5, (2, 3), -3), (1.0, (1, 2), -2), (1.5, 3, -4)],
    "sparse":     [(0.0, (0, 5, 2), 0), (1.5, (1, 3), -4)],
    "lilting":    [(0.0, 0, 0), (0.75, (2, 3), -3), (1.5, (1, 2), -4)],
    "answering":  [(0.0, 0, 0), (0.25, 1, -4), (1.0, (2, 3), -2), (1.75, 1, -4)],
    "drone":      [(0.0, (0, 1), 0), (0.5, (2, 3), -3), (1.5, (2, 3), -4)],
    "settled":    [(0.0, (0, 5, 2), 0), (0.75, (1, 3), -3), (1.5, 2, -4)],
    "reaching":   [(0.0, 0, 0), (0.5, 1, -3), (0.75, (2, 3), -2), (1.5, (1, 2), -4)],
}

FIGURES_TRIPLE = {
    "block":    [(0.0, (0, 5), 0), (1.0, (1, 2), -3), (1.5, 3, -4),
                   (2.0, (1, 2), -4)],
    "waltz":      [(0.0, 0, 0), (1.0, (2, 3), -3), (1.75, 1, -4), (2.0, (1, 2), -4)],
    "open":       [(0.0, 0, 0), (0.75, 1, -4), (1.0, (1, 2), -3), (2.0, (2, 3), -4)],
    "rocking":    [(0.0, 0, 0), (1.0, (2, 3), -2), (1.5, 1, -4), (2.0, (1, 2), -4)],
    "walking":    [(0.0, 0, 0), (1.0, (1, 2), -3), (1.5, 3, -4), (2.0, 1, -3),
                   (2.5, 2, -4)],
    "sustained":  [(0.0, (0, 5, 2), 0), (1.0, (1, 3), -3), (2.0, (1, 2), -4)],
    "arpeggio":   [(0.0, 0, 0), (0.75, 1, -3), (1.25, 2, -2), (1.75, 3, -3),
                   (2.25, (1, 2), -4)],
    "cadential":  [(0.0, (0, 5, 2), 0), (1.0, 3, -4), (1.5, (1, 2), -3)],
    "thin":       [(0.0, 0, 0), (1.0, 1, -4), (1.5, (1, 2), -3)],
    "lilting":    [(0.0, (0, 5), 0), (0.75, (2, 3), -3), (2.0, (1, 2), -4)],
    "answering":  [(0.0, 0, 0), (1.0, 1, -4), (1.25, (2, 3), -2), (2.0, 1, -4),
                   (2.5, 2, -4)],
    "sparse":     [(0.0, (0, 5), 0), (1.25, 2, -3), (2.0, (1, 3), -4)],
    "drone":      [(0.0, (0, 5, 1), 0), (1.0, (2, 3), -3), (2.0, (2, 3), -4)],
    "afterbeat":  [(0.0, 0, 0), (0.5, (1, 2), -3), (1.5, (2, 3), -3), (2.5, 1, -4)],
}

# 6/8
FIGURES_COMPOUND = {
    "plain":      [(0.0, 0, 0), (0.5, (2, 3), -4), (1.0, (2, 3), -5),
                   (1.5, 1, -2), (2.0, (2, 3), -4), (2.5, (2, 3), -5)],
    "lilting":    [(0.0, (0, 5), 0), (0.5, (2, 3), -4), (1.0, (1, 2), -5),
                   (1.5, 1, -2), (2.0, (2, 3), -4), (2.5, (1, 2), -5)],
    "rocking":    [(0.0, 0, 0), (1.0, (2, 3), -3), (1.5, 1, -2), (2.5, (2, 3), -4)],
    "sustained":  [(0.0, (0, 5, 2), 0), (1.5, (1, 3), -3)],
    "cadential":  [(0.0, (0, 5, 2), 0), (1.5, (1, 2, 3), -3)],
    "settled":    [(0.0, (0, 5), 0), (0.5, (2, 3), -4), (1.5, (1, 2), -3)],
    "sparse":     [(0.0, (0, 5), 0), (1.0, (2, 3), -4), (1.5, 1, -3)],
    "thin":       [(0.0, 0, 0), (1.0, (2, 3), -4), (1.5, 1, -3), (2.5, 2, -5)],
    "arpeggio":   [(0.0, 0, 0), (0.5, 1, -4), (1.0, 2, -3), (1.5, 3, -3),
                   (2.0, 2, -4), (2.5, 1, -5)],
    "walking":    [(0.0, 0, 0), (0.5, 2, -4), (1.0, 3, -4), (1.5, 1, -2),
                   (2.0, (2, 3), -4)],
    "answering":  [(0.0, 0, 0), (1.0, (2, 3), -3), (2.0, (1, 2), -3)],
    "open":       [(0.0, (0, 2), 0), (1.5, (1, 3), -3), (2.5, 2, -5)],
    "block":    [(0.0, (0, 5), 0), (0.5, (1, 2), -4), (1.5, (1, 2, 3), -3)],
    "double":     [(0.0, (0, 5, 1), 0), (1.0, (2, 3), -3), (1.5, 1, -3),
                   (2.0, (2, 3), -4)],
    "afterbeat":  [(0.0, 0, 0), (0.5, (1, 2), -3), (1.5, 1, -2), (2.0, (2, 3), -3)],
    "reaching":   [(0.0, 0, 0), (0.5, 1, -4), (1.0, (2, 3), -3), (1.5, 1, -3),
                   (2.5, (2, 3), -4)],
}


def figures_for(beats_per_bar: float, compound: bool = False) -> dict:
    if compound:
        return FIGURES_COMPOUND
    return FIGURES_TRIPLE if abs(beats_per_bar % 2) > 1e-6 else FIGURES_DUPLE


FIGURES = FIGURES_DUPLE


def inflect_cell(fig: list, seed: int, cell: float, step: float = 0.25) -> list:
    """音型最多改一处: 挪一个音 / 加经过音 / 去掉最后一个音。seed=0 不改。"""
    if not seed:
        return fig
    rnd = random.Random(seed)
    jieguo = [tuple(e) for e in fig]
    zhuti = [i for i, (off, _, _) in enumerate(jieguo) if off > 1e-6]
    if not zhuti:
        return jieguo
    # 复拍子格子粗, 多变一点
    roll = rnd.random() * (0.82 if step >= 0.5 else 1.0)
    if roll < 0.34:
        i = rnd.choice(zhuti)
        off, m, dv = jieguo[i]
        pianyi = rnd.choice((-2, -1, 1, 1, 2)) * step
        xin = off + pianyi
        if 0.25 <= xin <= cell - 0.25 and all(abs(xin - o) > 0.2
                                              for j, (o, _, _) in enumerate(jieguo)
                                              if j != i):
            jieguo[i] = (round(xin, 3), m, dv)
    elif roll < 0.62:
        i = rnd.choice(zhuti)
        off, m, dv = jieguo[i]
        at = off - rnd.choice((1, 2)) * step
        shengbu = 1 if not isinstance(m, tuple) else rnd.choice(m)
        if at > 0.2 and all(abs(at - o) > 0.2 for o, _, _ in jieguo):
            jieguo.append((round(at, 3), shengbu, dv - 2))
    elif roll < 0.80 and len(zhuti) > 1:
        jieguo.pop(max(zhuti))
    return sorted(jieguo, key=lambda e: e[0])


@dataclass
class BarPlan:
    bar: int
    figure: str
    density: float          # melody notes per beat in this bar
    phrase_pos: float       # 0..1 within its phrase
    harmony_changed: bool
    cadence: bool
    reasons: list = field(default_factory=list)
    second: tuple = ()      # figure for the bar's second half, if different
    inflect: int = 0        # seed for `inflect_cell`


def plan_bars(melody, harmony, *, bars: int, beats_per_bar: float = 4.0,
              phrases=None, seed: int = 0, compound: bool = False) -> list:
    """每小节选一个音型。melody 是 (拍, 音高, 时值), phrases 是 (起, 止)。"""
    ku = figures_for(beats_per_bar, compound)
    rnd = random.Random(seed)
    plans, shangci = [], None
    use = 0
    lianxu = 0
    jianguo = {}
    for b in range(bars):
        t0, t1 = b * beats_per_bar, (b + 1) * beats_per_bar
        zheli = [n for n in melody if t0 <= n[0] < t1]
        midu = len(zheli) / beats_per_bar
        chang = max((n[2] for n in zheli), default=0.0)
        weizhi = 0.0
        if phrases:
            for a, z in phrases:
                if a <= t0 < z and z > a:
                    weizhi = (t0 - a) / (z - a)
                    break
        bianle = True
        if harmony is not None:
            bianle = harmony.at(t0) is not harmony.at(max(0.0, t0 - beats_per_bar))
        # 旋律长音不算终止, 左手要继续动
        chiyin = chang >= 2.0
        cadence = weizhi > 0.78 and not chiyin

        houxuan, liyou = [], []
        if chiyin:
            houxuan = ["arpeggio", "walking", "rocking", "lilting", "answering",
                       "reaching", "afterbeat", "double"]
            liyou.append(f"tune holds {chang:.1f} beats: keep moving underneath it")
        elif cadence:
            houxuan = ["cadential", "sustained", "thin", "settled", "sparse"]
            liyou.append("phrase is closing: settle")
        elif midu >= 1.4:
            houxuan = ["thin", "open", "plain", "sparse", "block", "settled"]
            liyou.append(f"tune is busy ({midu:.2f} notes/beat): stay out of the way")
        else:
            houxuan = ["plain", "rocking", "open", "walking", "afterbeat", "double",
                       "block", "waltz", "lilting", "drone", "reaching",
                       "answering", "syncopated", "driving"]
            liyou.append("ordinary bar: ordinary figure")
        if bianle:
            liyou.append("harmony just changed")
            if "sustained" not in houxuan and not cadence:
                houxuan = houxuan + ["sustained"]
        houxuan = [c for c in houxuan if c in ku] or sorted(ku)
        key = (round(weizhi, 1), cadence, midu >= 1.4, chang >= 2.0)
        baochi = rnd.choice((2, 3, 3, 4)) if beats_per_bar < 4 else rnd.choice((1, 2))
        if shangci in houxuan and lianxu < baochi:
            fig = shangci
            lianxu += 1
            liyou.append(f"stays on {fig} ({lianxu} bars)")
        elif key in jianguo and jianguo[key] in houxuan and rnd.random() < 0.3:
            fig = jianguo[key]
            lianxu = 1
            liyou.append("same shape as the matching bar of an earlier phrase")
        else:
            chizi = [c for c in houxuan if c != shangci] or houxuan
            use = (use + 1 + rnd.randrange(len(chizi))) % len(chizi)
            fig = chizi[use]
            lianxu = 1
        jianguo.setdefault(key, fig)
        houban = ()
        if not cadence and beats_per_bar >= 4 and rnd.random() < 0.5:
            qita = [c for c in houxuan if c != fig]
            if qita:
                houban = (rnd.choice(qita),)
                liyou.append(f"second half varies to {houban[0]}")
        shangci = fig
        plans.append(BarPlan(b, fig, midu, weizhi, bianle, cadence, liyou, houban,
                             inflect=rnd.randrange(1, 1 << 30)))
    return plans


def _clashes_with_melody(pitch, at, melody, span):
    """小二度/大七度/小九度算冲突。"""
    for entry in melody:
        b, p, d = entry[0], entry[1], entry[2]
        if b >= at + span or b + d <= at:
            continue
        if abs(p - pitch) in (1, 11, 13):
            return True
    return False


def _lh_members(chord, target, ceiling):
    h = open_left_hand(chord, target=target, lo=target - 5,
                       hi=target + 9, ceiling=ceiling)
    m = [h.get("bass"), h.get("fifth"), h.get("octave"),
         h.get("colour"), h.get("colour2")]
    return [x if x is not None else h.get("bass") for x in m] + [h.get("deep")]


def render_bar(chord, plan: BarPlan, start: float, *, dyn: int,
               beats_per_bar: float = 4.0, ceiling: int = 59,
               target: int = 40, melody=None, mel_offset: float = 0.0,
               changes=(), compound: bool = False) -> list:
    """把一小节的 plan 弹出来。changes 是小节内换和弦 [(偏移, Chord)]。"""
    hexian_lie = list(chord) if isinstance(chord, (list, tuple)) else [chord]
    changes = sorted((c, x) for c, x in (changes or ())
                     if 0.0 <= c < beats_per_bar)

    def chord_at(off):
        if not changes:
            return None
        ch = None
        for c, x in changes:
            if c <= off + 1e-9:
                ch = x
        return ch

    shou = [_lh_members(ch, target, ceiling) for ch in hexian_lie]
    ku = figures_for(beats_per_bar, compound)
    cell = 2.0 if ku is FIGURES_DUPLE else 3.0
    mingzi = [plan.figure] + list(plan.second or [])
    mingzi = [n if n in ku else sorted(ku)[0] for n in mingzi]
    figs = [ku[f] for f in mingzi]
    figs = [inflect_cell(f, plan.inflect if (i == 0 or plan.inflect & 1)
                         else (plan.inflect * 2654435761) & 0x7fffffff, cell,
                         0.5 if compound else 0.25)
            for i, f in enumerate(figs)]
    # 每个音到同一声部下一次出现的距离
    jiange = []
    for fig in figs:
        nxt = {}
        for i, (off, member, _) in enumerate(fig):
            ms = member if isinstance(member, tuple) else (member,)
            houmian = [o for o, m, _ in fig
                       for mm in (m if isinstance(m, tuple) else (m,))
                       if mm in ms and o > off + 1e-6]
            nxt[i] = (min(houmian) - off) if houmian else (cell - off)
        jiange.append(nxt)

    jieguo = []
    for half in range(max(1, int(round(beats_per_bar / cell)))):
        base = start + half * cell
        fi = half % len(figs)
        base_members = shou[half % len(shou)]
        base_chord = hexian_lie[half % len(hexian_lie)]
        for idx, (off, member_spec, dv) in enumerate(figs[fi]):
            dangqian = chord_at(base + off - start)
            if dangqian is not None and dangqian is not base_chord:
                ch_here = dangqian
                members = _lh_members(dangqian, target, ceiling)
            else:
                members, ch_here = base_members, base_chord
            for member in (member_spec if isinstance(member_spec, tuple)
                           else (member_spec,)):
                p = members[member]
                if p is None:
                    continue
                zhengpai = (abs(off % 1.5) < 1e-6 if compound
                            else abs(off - round(off)) < 1e-6)
                voice = (Voice.BASS
                         if (member in (0, 5) or (member == 1 and zhengpai))
                         else Voice.INNER)
                at = base + off
                baochi = 0.92 if zhengpai else 0.68
                shichang = max(0.12, min(jiange[fi][idx] * baochi, 1.6))
                if len(hexian_lie) > 1:
                    shichang = min(shichang, cell - off)
                for c, _ in changes:
                    if at - start < c:
                        shichang = min(shichang, (start + c) - at)
                        break
                if melody is not None and member not in (0, 5):
                    rel = at - mel_offset
                    if _clashes_with_melody(p, rel, melody, shichang):
                        tihuan = None
                        for cand in ch_here.notes_in(max(21, p - 10),
                                                     min(ceiling, p + 10)):
                            if cand == p or _clashes_with_melody(cand, rel,
                                                                 melody, shichang):
                                continue
                            if tihuan is None or abs(cand - p) < abs(tihuan - p):
                                tihuan = cand
                        if tihuan is None:
                            continue
                        p = tihuan
                if member not in (0, 5) and p > ceiling:
                    while p > ceiling and p - 12 >= 21:
                        p -= 12
                jieguo.append(Note(at, shichang, p, max(1, min(127, dyn + dv)), voice,
                                   f"vary.{plan.figure}.{member}"))
    return jieguo


def pattern_entropy(onsets, *, bar: float = 4.0, grid: float = 0.25) -> dict:
    """每小节节奏型的熵 (bit), 看变化够不够。"""
    xiaojie = {}
    for t in onsets:
        xiaojie.setdefault(int(t // bar), set()).add(round((t % bar) / grid))
    pats = [tuple(sorted(v)) for v in xiaojie.values()]
    if not pats:
        return {"bars": 0, "distinct": 0, "modal_share": 0.0, "entropy": 0.0}
    c = Counter(pats)
    n = len(pats)
    H = -sum((v / n) * math.log2(v / n) for v in c.values())
    return {"bars": n, "distinct": len(c),
            "modal_share": round(c.most_common(1)[0][1] / n, 3),
            "entropy": round(H, 2)}


REFERENCE_ENTROPY = (4.5, 5.2)


def connect_gaps(perf, harmony, *, max_gap: float = 1.25, ceiling: int = 55,
                 floor: int = 33) -> int:
    """超过 max_gap 拍没有新音的地方补一个轻的和弦音, 返回补了几个。"""
    if harmony is None:
        return 0
    added = 0
    while True:
        ons = sorted({round(n.onset, 3) for n in perf.notes})
        kong = None
        for a, b in zip(ons, ons[1:]):
            if b - a > max_gap and (kong is None or b - a > kong[1] - kong[0]):
                kong = (a, b)
        if kong is None:
            break
        at = round((kong[0] + kong[1]) / 2 / 0.25) * 0.25
        if not (kong[0] + 0.2 < at < kong[1] - 0.2):
            at = (kong[0] + kong[1]) / 2
        hexian = harmony.at(at)
        if hexian is None:
            break
        zaixiang = [n for n in perf.notes if n.onset <= at < n.end]
        fujin = [n.pitch for n in zaixiang]
        ding = min([p for p in fujin if p <= ceiling] + [ceiling])
        p = chord_near(hexian, min(ceiling, ding + 2), floor, ceiling)
        for cand in sorted(hexian.notes_in(floor, ceiling),
                           key=lambda c: abs(c - (ding + 2))):
            if all(abs(cand - q) not in (1, 11, 13) for q in fujin):
                p = cand
                break
        lidu = int(np.mean([n.vel for n in zaixiang])) if zaixiang else 50
        perf.add(Note(at, min(max_gap, kong[1] - at) * 0.9, p,
                      max(1, lidu - 6), Voice.INNER, "vary.connect"))
        added += 1
        if added > 400:
            break
    return added
