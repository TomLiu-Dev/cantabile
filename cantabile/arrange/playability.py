# -*- coding: utf-8 -*-
"""两只手能不能弹: 手的跨度, 每只手几个音, 两手分界线要稳定不能来回跳。

参考: Parncutt et al. 1997 (跨度 MaxComf 13 / MaxPrac 15), Nakamura et al.
"""
from __future__ import annotations

from dataclasses import dataclass

from cantabile.model import Finding, Note, Performance, Report, Voice

__all__ = ["HandSpan", "reachable", "split_point", "hand_cost", "assign_hands",
           "check", "slices", "MIDDLE_C"]

EPS = 1e-6

MIDDLE_C = 60

# 不可能的手型, 比所有舒适度罚分加起来都大
HARD = 40.0

MAX_CROSS = 5.0


def _pitch(n):
    return int(n.pitch) if hasattr(n, "pitch") else int(n[2])


def _onset(n):
    return float(n.onset) if hasattr(n, "onset") else float(n[0])


def _dur(n):
    return float(n.dur) if hasattr(n, "dur") else float(n[1])


def _end(n):
    return _onset(n) + _dur(n)


def _is_melody(n):
    return getattr(n, "voice", None) is Voice.MELODY


def _pitches(ns):
    return sorted({_pitch(n) for n in ns}) if ns else []


def _centre(ps):
    # 取两端中点, 不是平均值
    return (ps[0] + ps[-1]) / 2.0 if ps else 0.0


@dataclass
class HandSpan:
    """一只手的极限。小手用 scaled(0.9)。"""
    max_span: int = 14
    comfortable: int = 9
    max_notes: int = 5

    def scaled(self, factor: float) -> "HandSpan":
        return HandSpan(int(round(self.max_span * factor)),
                        int(round(self.comfortable * factor)),
                        self.max_notes)


def reachable(pitches, span: HandSpan = HandSpan()) -> bool:
    """一只手能不能同时按住这些音。"""
    ps = sorted({p if isinstance(p, int) else _pitch(p) for p in pitches}) \
        if pitches else []
    if not ps:
        return True
    if len(ps) > span.max_notes:
        return False
    return ps[-1] - ps[0] <= span.max_span


def _cross_cost(ps, prev):
    """穿指的代价: 新位置和上一个位置重叠就要穿指。"""
    if not ps or not prev:
        return 0.0
    chongdie = min(ps[-1], prev[-1]) - max(ps[0], prev[0])
    if chongdie < 0:
        return 0.0
    if abs(_centre(ps) - _centre(prev)) < 1.0:
        return 0.0
    return 0.6 + 0.5 * max(0.0, chongdie - MAX_CROSS)


def hand_cost(notes_for_hand, prev_notes, span: HandSpan = HandSpan()) -> float:
    """一只手弹 notes_for_hand 的代价: 跨度 + 音数 + 跳动 + 穿指。"""
    ps = sorted({p if isinstance(p, int) else _pitch(p) for p in notes_for_hand}) \
        if notes_for_hand else []
    if not ps:
        return 0.0
    cost = 0.0

    kuandu = ps[-1] - ps[0]
    if kuandu > span.max_span:
        cost += HARD + 6.0 * (kuandu - span.max_span)
    elif kuandu > span.comfortable:
        cost += 1.2 * (kuandu - span.comfortable)

    n = len(ps)
    if n > span.max_notes:
        cost += HARD * (n - span.max_notes)
    elif n > 3:
        cost += 0.5 * (n - 3)

    prev = sorted({p if isinstance(p, int) else _pitch(p) for p in prev_notes}) \
        if prev_notes else []
    if prev:
        tiao = abs(_centre(ps) - _centre(prev))
        if tiao > span.comfortable:
            cost += 0.35 * (tiao - span.comfortable)
        if tiao > 2 * span.max_span:
            cost += 0.5 * HARD
        cost += _cross_cost(ps, prev)
    return cost


def slices(notes, tol: float = 0.03, release: float = 0.05,
           release_frac: float = 0.03) -> list:
    """每个起音点上正在响的音 [(拍, 音)], 包括还按着的。快放掉的不算。"""
    if not notes:
        return []
    qiyin = sorted({round(_onset(n), 6) for n in notes})
    hebing = [qiyin[0]]
    for t in qiyin[1:]:
        if t - hebing[-1] > tol:
            hebing.append(t)
    jieguo = []
    for t in hebing:
        zaixiang = [n for n in notes
                    if _onset(n) <= t + tol
                    and _end(n) - t > max(release, release_frac * _dur(n))]
        if zaixiang:
            jieguo.append((t, zaixiang))
    return jieguo


def _score_split(ps, k, prev_l, prev_r, span, anchor, melody):
    zuo, you = ps[:k], ps[k:]
    cost = hand_cost(zuo, prev_l, span) + hand_cost(you, prev_r, span)
    cost += 0.25 * abs(len(zuo) - len(you))
    fenjie = _boundary(ps, k)
    cost += 0.02 * abs(fenjie - anchor)
    # 旋律跑到左手、右手却在弹伴奏, 罚
    if melody:
        for p in zuo:
            if p in melody and any(q not in melody for q in you):
                cost += 1.5
    return cost


def _boundary(ps, k):
    if not ps:
        return float(MIDDLE_C)
    if k <= 0:
        return ps[0] - 0.5
    if k >= len(ps):
        return ps[-1] + 0.5
    return (ps[k - 1] + ps[k]) / 2.0


def _instant_split(live, prev_l, prev_r, span, anchor, melody):
    ps = _pitches(live)
    if not ps:
        return float(anchor)
    best, best_cost = float(anchor), float("inf")
    for k in range(len(ps) + 1):
        c = _score_split(ps, k, prev_l, prev_r, span, anchor, melody)
        if c < best_cost - 1e-9:
            best_cost, best = c, _boundary(ps, k)
    return best


def _trajectory(notes, span, hysteresis, window):
    """先逐个切片找分界, 再平滑, 最后分手。返回 ([(拍, 分界)], {下标: 'L'/'R'})。"""
    sl = slices(notes)
    if not sl:
        return [], {}
    xuanlv = {_pitch(n) for n in notes if _is_melody(n)}

    yuanshi = []
    prev_l, prev_r = [], []
    maodian = float(MIDDLE_C)
    for beat, live in sl:
        s = _instant_split(live, prev_l, prev_r, span, maodian, xuanlv)
        yuanshi.append((beat, s))
        prev_l = [p for p in _pitches(live) if p < s]
        prev_r = [p for p in _pitches(live) if p >= s]
        maodian = s

    pinghua = []
    dangqian = yuanshi[0][1]
    for i, (beat, s) in enumerate(yuanshi):
        fujin = [v for b, v in yuanshi if beat - window <= b <= beat + window]
        zhongwei = sorted(fujin)[len(fujin) // 2] if fujin else s
        if abs(zhongwei - dangqian) > hysteresis:
            dangqian = zhongwei
        if abs(s - dangqian) > max(hysteresis, span.max_span / 2.0):
            dangqian = s
        ps = _pitches(sl[i][1])
        if ps and not _splits_ok(ps, dangqian, span) and _splits_ok(ps, s, span):
            dangqian = s
        pinghua.append(dangqian)

    # 按下标存, 不用 id(): 结果会缓存, id 会被复用
    xiabiao = {}
    for i, n in enumerate(notes):
        xiabiao.setdefault(id(n), i)
    hands = {}
    by_index = {}
    jieguo = []
    prev = {"L": [], "R": []}
    for i, (beat, live) in enumerate(sl):
        want = pinghua[i]
        xin = [n for n in live if id(n) not in hands]
        liu = {"L": [n for n in live if hands.get(id(n)) == "L"],
               "R": [n for n in live if hands.get(id(n)) == "R"]}
        ps = _pitches(live)
        houxuan = [float(want)] + [_boundary(ps, k) for k in range(len(ps) + 1)]
        best, best_cost = float(want), None
        for c in houxuan:
            zuo = liu["L"] + [n for n in xin if _pitch(n) < c]
            you = liu["R"] + [n for n in xin if _pitch(n) >= c]
            cost = (hand_cost(zuo, prev["L"], span)
                    + hand_cost(you, prev["R"], span)
                    + 0.08 * abs(c - want))
            if not (reachable(_pitches(zuo), span)
                    and reachable(_pitches(you), span)):
                cost += HARD
            if best_cost is None or cost < best_cost - 1e-9:
                best_cost, best = cost, c
        for n in xin:
            hands[id(n)] = "L" if _pitch(n) < best else "R"
            by_index[xiabiao[id(n)]] = hands[id(n)]
        prev = {"L": _pitches([n for n in live if hands.get(id(n)) == "L"]),
                "R": _pitches([n for n in live if hands.get(id(n)) == "R"])}
        jieguo.append((beat, want))
    return jieguo, by_index


def _splits_ok(ps, split, span):
    return (reachable([p for p in ps if p < split], span)
            and reachable([p for p in ps if p >= split], span))


_TRAJ_CACHE = {}
_TRAJ_CACHE_MAX = 8


def _cached_trajectory(notes, span, hysteresis, window):
    key = (hysteresis, window, span.max_span, span.comfortable, span.max_notes,
           tuple((round(_onset(n), 6), round(_dur(n), 6), _pitch(n),
                  _is_melody(n)) for n in notes))
    hit = _TRAJ_CACHE.get(key)
    if hit is None:
        if len(_TRAJ_CACHE) >= _TRAJ_CACHE_MAX:
            _TRAJ_CACHE.clear()
        hit = _trajectory(notes, span, hysteresis, window)
        _TRAJ_CACHE[key] = hit
    return hit


def split_point(notes, beat, *, hysteresis: float = 2.0,
                span: HandSpan = HandSpan(), window: float = 8.0) -> int:
    """beat 处两手的分界音高, 低于它归左手。"""
    traj, _ = _cached_trajectory(list(notes), span, float(hysteresis),
                                 float(window))
    if not traj:
        return MIDDLE_C
    best = traj[0][1]
    for b, s in traj:
        if b <= beat + EPS:
            best = s
        else:
            break
    return int(round(best))


def assign_hands(notes, *, span: HandSpan = HandSpan(), smooth: bool = True) -> dict:
    """{id(note): 'L' 或 'R'}。key 是 id, 用的时候 notes 不能被回收。"""
    ns = list(notes)
    if not ns:
        return {}
    traj, by_index = _cached_trajectory(ns, span, 2.0 if smooth else 0.0,
                                        8.0 if smooth else 0.0)
    if not traj:
        return {}
    jieguo = {id(ns[i]): v for i, v in by_index.items()}
    # 很短的音进不了切片, 按起音处的分界分
    if len(jieguo) < len(ns):
        beats = [b for b, _ in traj]
        fenjie = [s for _, s in traj]
        i = 0
        for n in sorted(ns, key=_onset):
            if id(n) in jieguo:
                continue
            t = _onset(n)
            while i + 1 < len(beats) and beats[i + 1] <= t + EPS:
                i += 1
            jieguo[id(n)] = "L" if _pitch(n) < fenjie[i] else "R"
    return jieguo


def check(perf, span: HandSpan = HandSpan()) -> Report:
    """找出两只手弹不了的地方 (span / fingers / unreachable), 超舒适跨度记 warn。"""
    notes = perf.notes if isinstance(perf, Performance) else list(perf)
    rep = Report()
    if not notes:
        rep.stats["playability.slices"] = 0
        return rep

    hands = assign_hands(notes, span=span)
    sl = slices(notes)
    zuida_l = zuida_r = 0
    lashen = chuanzhi = 0
    prev = {"L": [], "R": []}
    bianhua = 0
    shangci = None
    yidong, zuida_yidong = [], 0.0
    # 手的位置 = 最近 chuangkou 拍里这只手弹过的范围, 同一手型里的分解不算移动
    zuijin = {"L": [], "R": []}
    weizhi = {"L": None, "R": None}
    chuangkou = 1.0

    for beat, live in sl:
        split = split_point(notes, beat, span=span)
        if shangci is not None and split != shangci:
            bianhua += 1
        shangci = split

        by = {"L": [], "R": []}
        for n in live:
            by[hands.get(id(n), "L" if _pitch(n) < split else "R")].append(n)

        allp = _pitches(live)
        if allp and not _coverable(allp, span):
            rep.add(Finding(check="playability.unreachable", beat=beat,
                            severity="error", notes=tuple(live),
                            detail="%d notes spanning %d semitones (%d..%d) "
                                   "cannot be covered by two hands"
                                   % (len(allp), allp[-1] - allp[0],
                                      allp[0], allp[-1])))

        for h in ("L", "R"):
            ps = _pitches(by[h])
            if not ps:
                continue
            kuandu = ps[-1] - ps[0]
            if h == "L":
                zuida_l = max(zuida_l, kuandu)
            else:
                zuida_r = max(zuida_r, kuandu)
            if kuandu > span.max_span:
                rep.add(Finding(check="playability.span", beat=beat,
                                severity="error", notes=tuple(by[h]),
                                detail="%s hand spans %d semitones (%d..%d), "
                                       "max %d" % (h, kuandu, ps[0], ps[-1],
                                                   span.max_span)))
            elif kuandu > span.comfortable:
                lashen += 1
                rep.add(Finding(check="playability.stretch", beat=beat,
                                severity="warn", notes=tuple(by[h]),
                                detail="%s hand stretches %d semitones "
                                       "(comfortable %d)"
                                       % (h, kuandu, span.comfortable)))
            if len(ps) > span.max_notes:
                rep.add(Finding(check="playability.fingers", beat=beat,
                                severity="error", notes=tuple(by[h]),
                                detail="%s hand asked for %d notes, has %d "
                                       "fingers" % (h, len(ps), span.max_notes)))
            if _cross_cost(ps, prev[h]) > 0.0:
                chuanzhi += 1
            zuijin[h] = [(b, p) for b, p in zuijin[h] if b > beat - chuangkou]
            zuijin[h].extend((beat, p) for p in ps)
            zheli = _centre(sorted(p for _, p in zuijin[h]))
            if weizhi[h] is not None:
                d = abs(zheli - weizhi[h])
                yidong.append(d)
                zuida_yidong = max(zuida_yidong, d)
            weizhi[h] = zheli
            prev[h] = ps

    rep.stats["playability.slices"] = len(sl)
    rep.stats["playability.split_changes"] = bianhua
    rep.stats["playability.widest_left"] = zuida_l
    rep.stats["playability.widest_right"] = zuida_r
    rep.stats["playability.stretches"] = lashen
    rep.stats["playability.thumb_crossings"] = chuanzhi
    rep.stats["playability.mean_hand_move"] = round(
        sum(yidong) / len(yidong), 2) if yidong else 0.0
    rep.stats["playability.max_hand_move"] = round(zuida_yidong, 2)
    return rep


def _coverable(ps, span):
    for k in range(len(ps) + 1):
        if reachable(ps[:k], span) and reachable(ps[k:], span):
            return True
    return False


if __name__ == "__main__":
    from cantabile.theory import name

    def banner(s):
        print("\n" + s + "\n" + "-" * max(62, len(s)))

    banner("HandSpan")
    hs = HandSpan()
    print("  max_span    %2d" % hs.max_span)
    print("  comfortable %2d" % hs.comfortable)
    print("  max_notes   %2d" % hs.max_notes)
    print("  small hand (x0.9):", hs.scaled(0.9))

    banner("reachable()")
    for ps in ([60, 64, 67], [48, 60, 62], [48, 65], [48, 64], [48, 65, 69],
               [60, 62, 64, 65, 67, 69]):
        print("  %-28s span %2d  n=%d  -> %s"
              % ([name(p) for p in ps], max(ps) - min(ps), len(ps),
                 reachable(ps, hs)))
    assert reachable([48, 62], hs) and not reachable([48, 65], hs)
    assert not reachable([60, 61, 62, 63, 64, 65], hs)

    banner("hand_cost()")
    now = [55, 59, 62]
    for prev, why in (([], "first chord"),
                      ([55, 59, 62], "already there"),
                      ([53, 57, 60], "a step below"),
                      ([31, 35, 38], "two octaves below")):
        print("  prev=%-18s %-18s cost %6.2f"
              % ([name(p) for p in prev] or "-", why,
                 hand_cost(now, prev, hs)))
    print("  wide  [48,65]:           cost %6.2f" % hand_cost([48, 65], [], hs))
    assert hand_cost([48, 65], [], hs) > HARD
    print("  cross [57,60] after [60,64]: cost %6.2f"
          % hand_cost([57, 60], [60, 64], hs))
    assert _cross_cost([57, 60], [60, 64]) > 0.0

    banner("split_point()")
    demo = []
    for bar in range(6):
        t = bar * 4.0
        for i in range(4):
            demo.append(Note(t + i, 1.0, 43 + (i % 2) * 7, 60, Voice.BASS))
        top = [74, 72, 71, 69] if bar != 3 else [62, 60, 59, 57]
        for i, p in enumerate(top):
            demo.append(Note(t + i, 1.0, p, 80, Voice.MELODY))
    for b in range(0, 24, 2):
        print("  beat %4.1f  split %s" % (b, name(split_point(demo, b))))
    steady = [split_point(demo, b) for b in range(0, 12)]
    assert len(set(steady)) == 1

    banner("assign_hands()")
    ha = assign_hands(demo)
    first = sorted([n for n in demo if n.onset == 0.0], key=lambda n: -n.pitch)
    print("  beat 0:", [(name(n.pitch), ha[id(n)]) for n in first])
    assert all(ha[id(n)] == ("R" if n.voice is Voice.MELODY else "L")
               for n in first)
    steady_on = [split_point(demo, b, hysteresis=2.0) for b in range(24)]
    steady_off = [split_point(demo, b, hysteresis=0.0, window=0.0)
                  for b in range(24)]
    print("  split with hysteresis:    %d distinct values, %d moves"
          % (len(set(steady_on)),
             sum(1 for a, b in zip(steady_on, steady_on[1:]) if a != b)))
    print("  split without:            %d distinct values, %d moves"
          % (len(set(steady_off)),
             sum(1 for a, b in zip(steady_off, steady_off[1:]) if a != b)))
    assert len(set(steady_on)) <= len(set(steady_off))

    banner("held notes keep their hand")
    sus = [Note(0.0, 4.0, 72, 80, Voice.MELODY),
           Note(0.0, 0.5, 40, 60, Voice.BASS),
           Note(1.0, 0.5, 55, 60, Voice.INNER),
           Note(2.0, 0.5, 59, 60, Voice.INNER),
           Note(3.0, 0.5, 62, 60, Voice.INNER)]
    hb = assign_hands(sus)
    for n in sorted(sus, key=lambda x: (x.onset, x.pitch)):
        print("  %5.1f %-4s %s" % (n.onset, name(n.pitch), hb[id(n)]))
    assert hb[id(sus[0])] == "R"
    assert check(Performance(list(sus))).ok()

    banner("check(): playable texture")
    r = check(Performance(list(demo)))
    print(r.summary(limit=4))
    print("  ok() =", r.ok())

    banner("check(): unplayable chord")
    bad = Performance([Note(0.0, 2.0, 40, 70, Voice.BASS, "bad"),
                       Note(0.0, 2.0, 57, 70, Voice.INNER, "bad"),
                       Note(0.0, 2.0, 74, 80, Voice.MELODY, "bad")])
    rb = check(bad)
    print(rb.summary())
    assert not rb.ok()
    assert any(f.check == "playability.span" for f in rb.errors)
    assert any(f.check == "playability.unreachable" for f in rb.errors)

    banner("check(): a 17th split between the hands")
    fine = Performance([Note(0.0, 2.0, 48, 70, Voice.BASS, "ok"),
                        Note(0.0, 2.0, 65, 80, Voice.MELODY, "ok")])
    rf = check(fine)
    print("  ok() =", rf.ok())
    assert rf.ok()

    banner("check(): too many notes")
    six = Performance([Note(0.0, 1.0, p, 70, Voice.INNER, "ok")
                       for p in (55, 56, 57, 58, 59, 60)])
    print("  six notes in a ninth  -> ok() =", check(six).ok())
    assert check(six).ok()
    cluster = Performance([Note(0.0, 1.0, p, 70, Voice.INNER, "bad")
                           for p in range(55, 66)])
    rc = check(cluster)
    print(rc.summary())
    assert any(f.check == "playability.fingers" for f in rc.errors)
