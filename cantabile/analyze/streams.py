"""分声部: 把复调音符拆成几条单声部的流。

参考: Chew & Wu 2005 contig mapping; Guiomard-Kagan 2016; Karydis 2007 (VISA); skyline 作对照。
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace

from ..model import Note, Voice

__all__ = [
    "Stream", "Separation", "contigs", "separate", "skyline", "streaming_cost",
    "evaluate", "CONNECTION_FEATURES", "WEIGHTS_CW", "WEIGHTS_IMS",
    "WEIGHTS_GA1", "WEIGHTS_DEFAULT", "METHODS",
]


@dataclass
class Stream:
    notes: list = field(default_factory=list)
    label: str = ""

    def pitch_range(self) -> tuple:
        if not self.notes:
            return (None, None)
        ps = [n.pitch for n in self.notes]
        return (min(ps), max(ps))

    def density(self, tempo=None) -> float:
        """每拍几个音; 给了 tempo 就是每秒。"""
        if not self.notes:
            return 0.0
        span = self.span
        if span <= 0:
            return float(len(self.notes))
        if tempo is None:
            return len(self.notes) / span
        miao = tempo.seconds(self.end) - tempo.seconds(self.start)
        return len(self.notes) / miao if miao > 0 else float(len(self.notes))

    @property
    def start(self) -> float:
        return min((n.onset for n in self.notes), default=0.0)

    @property
    def end(self) -> float:
        return max((n.end for n in self.notes), default=0.0)

    @property
    def span(self) -> float:
        return self.end - self.start

    def sort(self) -> "Stream":
        self.notes.sort(key=lambda n: (n.onset, n.pitch))
        return self

    def monophonic(self, tol: float = 1e-6, prefer: str = "high") -> "Stream":
        """同时起的音只留最高(或最低)的, 重叠的截断。

        >>> from cantabile.model import Note
        >>> s = Stream([Note(0, 4, 60, 80), Note(0, 4, 64, 80), Note(2, 1, 67, 80)])
        >>> [(n.onset, n.pitch, n.dur) for n in s.monophonic()]
        [(0, 64, 2), (2, 67, 1)]
        """
        ns = sorted(self.notes, key=lambda n: (n.onset, n.pitch))
        liu = []
        for n in ns:
            if liu and abs(liu[-1].onset - n.onset) <= tol:
                genhao = n.pitch > liu[-1].pitch if prefer == "high" \
                    else n.pitch < liu[-1].pitch
                if genhao:
                    liu[-1] = n
            else:
                liu.append(n)
        jieguo = []
        for i, n in enumerate(liu):
            if i + 1 < len(liu) and n.end > liu[i + 1].onset + tol:
                n = replace(n, dur=max(liu[i + 1].onset - n.onset, tol))
            jieguo.append(n)
        return Stream(jieguo, self.label)

    def is_monophonic(self, tol: float = 1e-6) -> bool:
        ns = sorted(self.notes, key=lambda n: (n.onset, n.pitch))
        return all(a.end <= b.onset + tol for a, b in zip(ns, ns[1:]))

    def tagged(self, voice: Voice) -> "Stream":
        return Stream([replace(n, voice=voice, tag=self.label or n.tag)
                       for n in self.notes], self.label)

    def __len__(self) -> int:
        return len(self.notes)

    def __iter__(self):
        return iter(self.notes)

    def __repr__(self) -> str:
        lo, hi = self.pitch_range()
        return (f"<Stream {self.label or '?'} {len(self.notes)}n "
                f"{lo}..{hi} {self.start:.2f}-{self.end:.2f}>")


class Separation(list):
    """separate() 的结果, 另外带着 fragments / connections / contigs / method, 给 evaluate 用。"""

    def __init__(self, streams, fragments=(), connections=(), contigs=(),
                 method=""):
        super().__init__(streams)
        self.fragments = [list(f) for f in fragments]
        self.connections = [tuple(c) for c in connections]
        self.contigs = list(contigs)
        self.method = method


class _N:
    # Note 不能 hash, 所以按下标记
    __slots__ = ("i", "onset", "end", "dur", "pitch", "note")

    def __init__(self, i, note):
        self.i = i
        self.note = note
        self.onset = float(note.onset)
        self.dur = float(note.dur)
        self.end = self.onset + self.dur
        self.pitch = int(note.pitch)


def _as_note(x) -> Note:
    if isinstance(x, Note):
        return x
    if isinstance(x, (tuple, list)):
        if len(x) == 4:                       # (onset, dur, pitch, vel)
            return Note(float(x[0]), float(x[1]), int(x[2]), int(x[3]))
        if len(x) == 3:                       # (beat, pitch, dur)
            return Note(float(x[0]), float(x[2]), int(x[1]), 80)
    raise TypeError(f"cannot read {x!r} as a Note")


def _prep(notes, tol=1e-6):
    items = [_N(i, _as_note(n)) for i, n in enumerate(notes)]
    items = [it for it in items if it.dur > 1e-9]
    items.sort(key=lambda it: (it.onset, it.pitch, it.i))
    for k, it in enumerate(items):
        it.i = k
    return items


class _Slice:
    __slots__ = ("a", "b", "slots")

    def __init__(self, a, b, slots):
        self.a, self.b, self.slots = a, b, slots


class _Contig:
    __slots__ = ("slices", "k")

    def __init__(self, slices):
        self.slices = slices
        self.k = len(slices[0].slots)

    @property
    def a(self):
        return self.slices[0].a

    @property
    def b(self):
        return self.slices[-1].b

    def notes(self):
        yijian, jieguo = set(), []
        for s in self.slices:
            for slot in s.slots:
                for it in slot:
                    if it.i not in yijian:
                        yijian.add(it.i)
                        jieguo.append(it)
        jieguo.sort(key=lambda it: (it.onset, it.pitch))
        return jieguo

    def fragment(self, j):
        lian, yijian = [], set()
        for s in self.slices:
            for it in s.slots[j]:
                if it.i not in yijian:
                    yijian.add(it.i)
                    lian.append(it)
        return lian


def _slices(items, tol, max_voices):
    if not items:
        return []
    ts = sorted(set([round(it.onset / tol) for it in items]
                    + [round(it.end / tol) for it in items]))
    ts = [t * tol for t in ts]
    jieguo = []
    for a, b in zip(ts, ts[1:]):
        if b - a <= tol:
            continue
        mid = 0.5 * (a + b)
        xiang = [it for it in items if it.onset <= mid < it.end]
        if not xiang:
            continue
        xiang.sort(key=lambda it: (it.pitch, it.i))
        if max_voices and len(xiang) > max_voices:
            # 超过声部数: 把最低的几个并成一格, 上面的声部不动
            duo = len(xiang) - max_voices
            ge = [xiang[:duo + 1]] + [[it] for it in xiang[duo + 1:]]
        else:
            ge = [[it] for it in xiang]
        jieguo.append(_Slice(a, b, ge))
    return jieguo


def _contigs(items, tol, max_voices):
    jieguo, run = [], []
    for s in _slices(items, tol, max_voices):
        if run and len(run[-1].slots) == len(s.slots) \
                and abs(run[-1].b - s.a) <= tol:
            run.append(s)
        else:
            if run:
                jieguo.append(_Contig(run))
            run = [s]
    if run:
        jieguo.append(_Contig(run))
    return jieguo


def contigs(notes, *, tol=1e-6) -> list:
    """Chew & Wu 的 contig 切分: ``[(start, end, [Note, ...]), ...]``。

    >>> len(contigs([]))
    0
    """
    return [(c.a, c.b, [it.note for it in c.notes()])
            for c in _contigs(_prep(notes, tol), tol, None)]


CONNECTION_FEATURES = (
    "increase", "increase_one", "increase_equal",
    "decrease", "decrease_one", "decrease_equal",
    "difference_nb_voices", "maximal_voices", "minimal_voices",
    "maximal_sim_notes", "crossed_voices", "no_crossed_voices",
    "extreme_pitch", "avg_pitch_right", "avg_pitch_left", "avg_pitch",
    "extreme_dur", "avg_dur_right", "avg_dur_left", "avg_dur",
    "top_voice",
)

# ISMIR 2016 Table 1
WEIGHTS_CW = {"maximal_voices": 0.50, "no_crossed_voices": 0.25,
              "extreme_pitch": 0.25}

WEIGHTS_IMS = {"increase_equal": 0.25, "minimal_voices": 0.25,
               "no_crossed_voices": 0.25, "extreme_pitch": 0.25}

WEIGHTS_GA1 = {
    "increase": 0.004, "increase_one": 0.004, "increase_equal": 0.137,
    "decrease": 0.013, "decrease_one": 0.019, "decrease_equal": 0.112,
    "difference_nb_voices": 0.009, "maximal_voices": 0.026,
    "minimal_voices": 0.007, "maximal_sim_notes": 0.007,
    "crossed_voices": 0.009, "no_crossed_voices": 0.248,
    "extreme_pitch": 0.090, "avg_pitch_right": 0.117, "avg_pitch_left": 0.023,
    "avg_pitch": 0.041, "extreme_dur": 0.007, "avg_dur_right": 0.048,
    "avg_dur_left": 0.006, "avg_dur": 0.073,
}

# top_voice 给权重的话, 旋律一休止就被内声部抢走
WEIGHTS_DEFAULT = dict(WEIGHTS_GA1, top_voice=0.0)

_NU = 60.0
_LAM = 6.0


def _logdur(d):
    # 64分音符 -> 0, 全音符 -> 6
    return min(_LAM, max(0.0, math.log(max(d, 1e-6) * 16.0, 2.0)))


class _Frag:
    __slots__ = ("notes", "fid", "first_pitch", "last_pitch", "avg_pitch",
                 "first_dur", "last_dur", "avg_dur", "rank")

    def __init__(self, notes, rank, fid=None):
        self.notes = notes
        self.rank = rank
        self.fid = fid
        ps = [it.pitch for it in notes]
        ds = [_logdur(it.dur) for it in notes]
        self.first_pitch, self.last_pitch = ps[0], ps[-1]
        self.avg_pitch = sum(ps) / len(ps)
        self.first_dur, self.last_dur = ds[0], ds[-1]
        self.avg_dur = sum(ds) / len(ds)


def _features(nl, nr, n, zuo, you, pairs):
    m = max(1, len(pairs))
    nu, lam = _NU * m, _LAM * m
    f = {}
    f["increase"] = 1.0 if nl < nr else 0.0
    f["increase_one"] = 1.0 if nl + 1 == nr else 0.0
    f["increase_equal"] = 1.0 if nl <= nr else 0.0
    f["decrease"] = 1.0 if nl > nr else 0.0
    f["decrease_one"] = 1.0 if nl - 1 == nr else 0.0
    f["decrease_equal"] = 1.0 if nl >= nr else 0.0
    f["difference_nb_voices"] = 1.0 - abs(nl - nr) / max(1.0, n - 1.0)
    f["maximal_voices"] = max(nl, nr) / float(n)
    f["minimal_voices"] = (n + 1 - min(nl, nr)) / float(n)

    gongxiang = sum(1 for li, ri in pairs
                    if zuo[li].notes[-1].i == you[ri].notes[0].i)
    f["maximal_sim_notes"] = gongxiang / float(m)

    jiaocha = any(zuo[a].rank < zuo[b].rank and you[c].rank > you[d].rank
                  for (a, c), (b, d) in itertools.combinations(pairs, 2))
    f["crossed_voices"] = 1.0 if jiaocha else 0.0
    f["no_crossed_voices"] = 0.0 if jiaocha else 1.0

    ep = al = ar = ap = 0.0
    ed = dl = dr = dd = 0.0
    for li, ri in pairs:
        L, R = zuo[li], you[ri]
        ep += abs(L.last_pitch - R.first_pitch)
        ar += abs(L.last_pitch - R.avg_pitch)
        al += abs(L.avg_pitch - R.first_pitch)
        ap += abs(L.avg_pitch - R.avg_pitch)
        ed += abs(L.last_dur - R.first_dur)
        dr += abs(L.last_dur - R.avg_dur)
        dl += abs(L.avg_dur - R.first_dur)
        dd += abs(L.avg_dur - R.avg_dur)
    f["extreme_pitch"] = max(0.0, 1.0 - ep / nu)
    f["avg_pitch_right"] = max(0.0, 1.0 - ar / nu)
    f["avg_pitch_left"] = max(0.0, 1.0 - al / nu)
    f["avg_pitch"] = max(0.0, 1.0 - ap / nu)
    f["extreme_dur"] = max(0.0, 1.0 - ed / lam)
    f["avg_dur_right"] = max(0.0, 1.0 - dr / lam)
    f["avg_dur_left"] = max(0.0, 1.0 - dl / lam)
    f["avg_dur"] = max(0.0, 1.0 - dd / lam)

    f["top_voice"] = 1.0 if any(li == len(zuo) - 1 and ri == len(you) - 1
                                for li, ri in pairs) else 0.0
    return f


def _score(f, quanzhong):
    return sum(w * f.get(k, 0.0) for k, w in quanzhong.items())


class _Super:
    # 已经连起来的一串 contig; slots[j] 是第 j 声部的 (fragment_id, notes) 列表
    __slots__ = ("lo", "hi", "slots")

    def __init__(self, index, contig, fids):
        self.lo = self.hi = index
        self.slots = [[(fids[j], contig.fragment(j))] for j in range(contig.k)]

    @property
    def k(self):
        return len(self.slots)

    def edge(self, right_side):
        jieguo = []
        for rank, pd in enumerate(self.slots):
            fid, frag = pd[-1] if right_side else pd[0]
            jieguo.append(_Frag(frag, rank, fid))
        return jieguo


def _crossing_free(p, q):
    m = min(p, q)
    if p <= q:
        for combo in itertools.combinations(range(q), m):
            yield list(zip(range(m), combo))
    else:
        for combo in itertools.combinations(range(p), m):
            yield list(zip(combo, range(m)))


def _best_connection(a, b, n, quanzhong):
    zuo, you = a.edge(True), b.edge(False)
    qiangzhi = {}
    for li, L in enumerate(zuo):
        for ri, R in enumerate(you):
            if L.notes[-1].i == R.notes[0].i:      # 同一个音被边界切开
                qiangzhi[li] = ri
    best, best_s, best_f = None, float("-inf"), None
    for pairs in _crossing_free(a.k, b.k):
        if qiangzhi:
            d = dict(pairs)
            if any(d.get(li) != ri for li, ri in qiangzhi.items()):
                continue
        f = _features(a.k, b.k, n, zuo, you, pairs)
        s = _score(f, quanzhong)
        if s > best_s:
            best, best_s, best_f = pairs, s, f
    if best is None:
        for pairs in _crossing_free(a.k, b.k):
            f = _features(a.k, b.k, n, zuo, you, pairs)
            s = _score(f, quanzhong)
            if s > best_s:
                best, best_s, best_f = pairs, s, f
    return best, best_s, best_f


def _merge(a, b, pairs):
    if a.k <= b.k:
        ge = [list(f) for f in b.slots]
        for li, ri in pairs:
            ge[ri] = a.slots[li] + ge[ri]
    else:
        ge = [list(f) for f in a.slots]
        for li, ri in pairs:
            ge[li] = ge[li] + b.slots[ri]
    a.slots = ge
    a.hi = b.hi
    return a


def _separate_contig(items, tol, max_voices, quanzhong, order):
    cs = _contigs(items, tol, max_voices)
    if not cs:
        return Separation([], method="contig")

    n = max(c.k for c in cs)
    # 从声部最多的 contig 往外连; juli[i] 是到最近那个的距离
    juli = [0 if c.k == n else 10 ** 6 for c in cs]
    for i in range(1, len(cs)):
        juli[i] = min(juli[i], juli[i - 1] + 1)
    for i in range(len(cs) - 2, -1, -1):
        juli[i] = min(juli[i], juli[i + 1] + 1)

    pianduan, fids, fspan = [], [], []
    for c in cs:
        ids = []
        for j in range(c.k):
            ids.append(len(pianduan))
            pianduan.append(c.fragment(j))
            fspan.append((c.a, c.b))
        fids.append(ids)

    supers = [_Super(i, c, fids[i]) for i, c in enumerate(cs)]
    cache = {}

    def key(idx):
        a, b = supers[idx], supers[idx + 1]
        if idx not in cache:
            cache[idx] = _best_connection(a, b, n, quanzhong)
        pairs, s, _ = cache[idx]
        if order == "score":
            return (-s, idx)
        return (min(juli[a.hi], juli[b.lo]), -s, idx)

    lianjie = []
    while len(supers) > 1:
        idx = min(range(len(supers) - 1), key=key)
        a, b = supers[idx], supers[idx + 1]
        pairs, _s, _f = cache[idx]
        zuo, you = a.edge(True), b.edge(False)
        for li, ri in pairs:
            lianjie.append((zuo[li].fid, you[ri].fid))
        _merge(a, b, pairs)
        supers.pop(idx + 1)
        cache.pop(idx, None)
        cache.pop(idx - 1, None)
        cache = {k: v for k, v in cache.items() if k < idx}

    # 跨边界的音会出现在两个片段里, 归到它开始的那个声部
    root = supers[0]
    guishu, beiyong = {}, {}
    for j, pd in enumerate(root.slots):
        for fid, frag in pd:
            a, _b = fspan[fid]
            for it in frag:
                if it.onset >= a - tol:
                    guishu[it.i] = j
                else:
                    beiyong.setdefault(it.i, j)
    for i, j in beiyong.items():
        guishu.setdefault(i, j)

    liu = []
    for j in range(len(root.slots)):
        yin = [it.note for it in items if guishu.get(it.i) == j]
        yin.sort(key=lambda x: (x.onset, x.pitch))
        liu.append(Stream(yin, f"contig v{j}"))
    fangle = set(guishu)

    # 比 tol 还短的音不会进任何 slice, 放到音区最近的流里
    guer = [it for it in items if it.i not in fangle]
    if guer and liu:
        junzhi = [sum(x.pitch for x in s.notes) / len(s.notes) if s.notes else 0.0
                  for s in liu]
        for it in guer:
            k = min(range(len(liu)), key=lambda j: abs(junzhi[j] - it.pitch))
            liu[k].notes.append(it.note)
        for s in liu:
            s.sort()
    return Separation(liu, fragments=[[it.note for it in f]
                                      for f in pianduan],
                      connections=lianjie,
                      contigs=[(c.a, c.b, [it.note for it in c.notes()])
                               for c in cs],
                      method="contig")


def _tail(x):
    if isinstance(x, Stream):
        return max(x.notes, key=lambda n: (n.onset, n.pitch)) if x.notes else None
    if isinstance(x, _N):
        return x.note
    return _as_note(x)


def _head(x):
    if isinstance(x, Stream):
        return min(x.notes, key=lambda n: (n.onset, n.pitch)) if x.notes else None
    if isinstance(x, _N):
        return x.note
    return _as_note(x)


def streaming_cost(a, b, *, w_pitch=1.0, w_gap=1.0, w_overlap=2.0) -> float:
    """把 b 听成 a 的延续的代价: 音程 + 空隙 + 重叠 (Karydis 2007)。

    >>> from cantabile.model import Note
    >>> streaming_cost(Note(0, 1, 60, 80), Note(1, 1, 62, 80))
    2.0
    >>> streaming_cost(Note(0, 1, 60, 80), Note(2, 1, 60, 80))
    1.0
    """
    x, y = _tail(a), _head(b)
    if x is None or y is None:
        return float("inf")
    chongdie = max(0.0, min(x.end, y.end) - max(x.onset, y.onset))
    kongxi = max(0.0, y.onset - x.end)
    daijia = w_pitch * abs(x.pitch - y.pitch) + w_gap * kongxi
    if chongdie > 0:
        if w_overlap == float("inf"):
            return float("inf")
        daijia += w_overlap * chongdie
    return daijia


def _cluster_vertically(sls, chuangkou, w_size, yuzhi, tol):
    # 窗口里同时起、等长的音对比例超过阈值, 才把它们并成一团
    same = tot = 0
    for p, q in itertools.combinations(chuangkou, 2):
        if abs(p.onset - q.onset) <= tol:
            tot += 1
            if abs(p.dur - q.dur) <= max(tol, 0.05 * max(p.dur, q.dur)):
                same += 1
    r = (same / tot) if tot else 0.0
    if r <= yuzhi:
        return [[it] for it in sls]
    zu = {}
    for it in sls:
        zu.setdefault(round(it.dur / max(tol, 1e-3)), []).append(it)
    jieguo = [sorted(g, key=lambda x: x.pitch) for g in zu.values()]
    jieguo.sort(key=lambda g: g[0].pitch)
    return jieguo


def _match_crossing_free(shengbu, tuan, daijia, top_penalty):
    p, q = len(shengbu), len(tuan)
    if p == 0 or q == 0:
        return []
    best, best_c = None, float("inf")
    for pairs in _crossing_free(p, q):
        c = 0.0
        ok = True
        for vi, ci in pairs:
            w = daijia[vi][ci]
            if w == float("inf"):
                ok = False
                break
            c += w
        if not ok:
            continue
        if not any(vi == p - 1 and ci == q - 1 for vi, ci in pairs):
            c += top_penalty
        if c < best_c:
            best, best_c = pairs, c
    return best or []


def _separate_streaming(items, tol, max_voices, w_size, yuzhi,
                        top_penalty, quanzhong):
    w_pitch = (quanzhong or {}).get("w_pitch", 1.0)
    w_gap = (quanzhong or {}).get("w_gap", 1.0)
    shengbu = []
    i, nitems = 0, len(items)
    while i < nitems:
        j = i
        while j < nitems and abs(items[j].onset - items[i].onset) <= tol:
            j += 1
        sls = items[i:j]
        lo, hi = max(0, i - w_size // 2), min(nitems, j + w_size // 2)
        tuan = _cluster_vertically(sls, items[lo:hi], w_size, yuzhi, tol)
        # 只有一个和弦时把最高音拆出来, 好给最高声部
        if len(tuan) == 1 and len(tuan[0]) > 1 and (max_voices or 9) > 1:
            top = tuan[0]
            tuan = [top[:-1], [top[-1]]]
        daijia = []
        for v in shengbu:
            hang = []
            last = v[-1]
            for c in tuan:
                if any(it.onset < last.end - tol for it in c):
                    hang.append(float("inf"))
                else:
                    d = min(abs(it.pitch - last.pitch) for it in c)
                    g = max(0.0, min(it.onset for it in c) - last.end)
                    hang.append(w_pitch * d + w_gap * g)
            daijia.append(hang)
        pairs = _match_crossing_free(shengbu, tuan, daijia, top_penalty)
        yongle = set()
        for vi, ci in pairs:
            shengbu[vi].extend(tuan[ci])
            yongle.add(ci)
        for ci, c in enumerate(tuan):
            if ci in yongle:
                continue
            if len(shengbu) < (max_voices or 99):
                shengbu.append(list(c))
            else:
                k = min(range(len(shengbu)),
                        key=lambda v: abs(shengbu[v][-1].pitch - c[0].pitch))
                shengbu[k].extend(c)
        # 按平均音高排, 比按最后一个音稳
        shengbu.sort(key=lambda v: sum(it.pitch for it in v) / len(v))
        i = j
    liu = []
    for k, v in enumerate(shengbu):
        yin = [it.note for it in sorted(v, key=lambda x: (x.onset, x.pitch))]
        liu.append(Stream(yin, f"visa v{k}"))
    return Separation(liu, method="streaming")


def skyline(notes) -> Stream:
    """每个时刻取最高音, 被更高的音盖住就截断。

    >>> from cantabile.model import Note
    >>> [n.pitch for n in skyline([Note(0, 2, 60, 80), Note(1, 1, 67, 80)])]
    [60, 67]
    """
    items = _prep(notes)
    if not items:
        return Stream([], "skyline")
    duan = []
    for s in _slices(items, 1e-6, None):
        top = s.slots[-1][-1]
        if duan and duan[-1][0].i == top.i and abs(duan[-1][2] - s.a) <= 1e-6:
            duan[-1][2] = s.b
        else:
            duan.append([top, s.a, s.b])
    yin = []
    for it, a, b in duan:
        yin.append(replace(it.note, onset=a, dur=max(b - a, 1e-6),
                           voice=Voice.MELODY, tag="skyline"))
    return Stream(yin, "skyline")


METHODS = ("contig", "streaming", "skyline")


def separate(notes, *, method="contig", max_voices=6, weights=None,
             tol=1.0 / 16) -> list:
    """把 notes 拆成几条 Stream, method 是 contig / streaming / skyline。"""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; have {METHODS}")
    tol = max(float(tol), 1e-9)
    items = _prep(notes, tol)
    if not items:
        return Separation([], method=method)

    if method == "skyline":
        sky = skyline(notes)
        liuxia = {(round(n.pitch), round(n.onset, 6)) for n in sky.notes}
        shengxia = [it.note for it in items
                    if (round(it.pitch), round(it.onset, 6)) not in liuxia]
        return Separation([Stream(shengxia, "skyline rest"), sky], method="skyline")

    if method == "streaming":
        w = dict(weights or {})
        return _separate_streaming(items, tol, max_voices,
                                   int(w.get("window", 12)),
                                   float(w.get("threshold", 0.8)),
                                   float(w.get("top_penalty", 3.0)), w)

    w = dict(weights or WEIGHTS_DEFAULT)
    order = w.pop("order", "outward")
    return _separate_contig(items, tol, max_voices, w, order)


def _truth_map(truth):
    if isinstance(truth, dict):
        jieguo = {}
        for k, v in truth.items():
            if isinstance(k, Note):
                jieguo[(k.pitch, round(k.onset, 6))] = v
            else:
                jieguo[k] = v
        return jieguo
    jieguo = {}
    for label, group in enumerate(truth):
        mingzi = getattr(group, "label", "") or f"truth v{label}"
        for n in (group.notes if isinstance(group, Stream) else group):
            n = _as_note(n)
            jieguo[(n.pitch, round(n.onset, 6))] = mingzi
    return jieguo


def _dominant(notes, tmap):
    jishu = {}
    for n in notes:
        lab = tmap.get((n.pitch, round(n.onset, 6)))
        if lab is not None:
            jishu[lab] = jishu.get(lab, 0) + 1
    if not jishu:
        return None, 0, 0
    lab = max(jishu, key=lambda k: (jishu[k], str(k)))
    return lab, jishu[lab], sum(jishu.values())


def evaluate(predicted, truth) -> dict:
    """跟标准答案比: AVC, AFC, 连接正确率, NPR, TR。按 (pitch, onset) 对音。"""
    tmap = _truth_map(truth)
    liu = [s if isinstance(s, Stream) else Stream(list(s))
           for s in predicted]

    yizhi, npr_ok, npr_all = [], 0, 0
    fenpei = {}
    for k, s in enumerate(liu):
        lab, hit, tot = _dominant(s.notes, tmap)
        if tot:
            yizhi.append(hit / tot)
            npr_ok += hit
            npr_all += tot
        for n in s.notes:
            fenpei[(n.pitch, round(n.onset, 6))] = k
    avc = sum(yizhi) / len(yizhi) if yizhi else 0.0

    def _trans(groups):
        jieguo = set()
        for g in groups:
            ns = sorted(g, key=lambda n: (n.onset, n.pitch))
            for a, b in zip(ns, ns[1:]):
                jieguo.add(((a.pitch, round(a.onset, 6)),
                            (b.pitch, round(b.onset, 6))))
        return jieguo

    tgroups = {}
    for key, lab in tmap.items():
        tgroups.setdefault(lab, []).append(Note(key[1], 0.0, key[0], 80))
    tr_truth = _trans(tgroups.values())
    tr_pred = _trans([s.notes for s in liu])
    tr = len(tr_truth & tr_pred) / len(tr_truth) if tr_truth else 0.0

    jieguo = {
        "voices": len(liu),
        "voice_consistency": avc,
        "note_precision": (npr_ok / npr_all) if npr_all else 0.0,
        "transition": tr,
        "fragment_consistency": None,
        "correct_connections": None,
        "fragments": 0,
        "connections": 0,
    }

    frags = getattr(predicted, "fragments", None)
    if frags:
        vals, doms = [], []
        for f in frags:
            lab, hit, tot = _dominant(f, tmap)
            doms.append(lab)
            if tot:
                vals.append(hit / tot)
        jieguo["fragment_consistency"] = sum(vals) / len(vals) if vals else 0.0
        jieguo["fragments"] = len(frags)
        conns = getattr(predicted, "connections", [])
        good = [1.0 for li, ri in conns
                if doms[li] is not None and doms[li] == doms[ri]]
        jieguo["connections"] = len(conns)
        jieguo["correct_connections"] = (len(good) / len(conns)) if conns else None
    return jieguo


def _demo_four_part():
    """合成的四声部例子 (SATB), 返回 (notes, truth_streams, tune_pitches)。"""
    from ..theory import Chord, chord_near

    jinxing = [("C", ""), ("A", "m"), ("F", ""), ("G", "7"),
               ("C", ""), ("F", ""), ("G", ""), ("C", "")]
    tune = [72, 74, 76, 74, 72, 77, 76, 72]
    # 下面几个声部小节中间重敲, 声部数会变
    neisheng = [(0, 1, 2), (0,), (1, 2), (0, 1, 2), (0,), (0, 1), (0, 1, 2), ()]
    truth = [[], [], [], []]                       # B T A S
    mubiao = [48, 60, 65, None]
    for k, ((root, qual), mel) in enumerate(zip(jinxing, tune)):
        ch = Chord.parse(root, qual)
        beat = float(k * 2)
        pitches = [chord_near(ch, t) for t in mubiao[:3]] + [mel]
        for v, p in enumerate(pitches):
            if v == 3:
                # 旋律提前放键
                truth[v].append(Note(beat, 1.75, p, 84))
                continue
            if v in neisheng[k]:
                truth[v].append(Note(beat, 1.0, p, 74))
                truth[v].append(Note(beat + 1.0, 1.0, p, 70))
            else:
                truth[v].append(Note(beat, 2.0, p, 74))
        mubiao = [pitches[0], pitches[1], pitches[2], None]
    notes = [n for g in truth for n in g]
    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes, [Stream(g, f"truth v{i}") for i, g in enumerate(truth)], tune


def _main():
    notes, truth, tune = _demo_four_part()
    print(f"synthetic four-part piece: {len(notes)} notes, 4 known voices")

    cs = contigs(notes)
    ks = [c.k for c in _contigs(_prep(notes), 1e-6, None)]
    print(f"contigs: {len(cs)}  voice counts {ks[:14]}...")

    for label, kw in (("contig/CW", dict(method="contig", weights=WEIGHTS_CW)),
                      ("contig/GA1", dict(method="contig")),
                      ("streaming/VISA", dict(method="streaming")),
                      ("skyline", dict(method="skyline"))):
        got = separate(notes, max_voices=4, **kw)
        m = evaluate(got, truth)
        fc = m["fragment_consistency"]
        cc = m["correct_connections"]
        print(f"  {label:16s} voices={m['voices']} "
              f"AVC={m['voice_consistency']:6.2%} "
              f"NPR={m['note_precision']:6.2%} TR={m['transition']:6.2%} "
              f"AFC={'n/a' if fc is None else format(fc, '6.2%')} "
              f"CC={'n/a' if cc is None else format(cc, '6.2%')}")

    sky = skyline(notes)
    print(f"  skyline stream: {len(sky)} notes, monophonic={sky.is_monophonic()}"
          f", range={sky.pitch_range()}, density={sky.density():.2f}/beat")

    got = separate(notes, max_voices=4)
    assert len(got) == 4, got
    assert all(s.is_monophonic() for s in got), "a separated stream overlaps itself"
    top = got[-1]
    assert [n.pitch for n in top] == tune, [n.pitch for n in top]
    print("  top stream recovers the synthetic tune exactly: OK")

    assert streaming_cost(Note(0, 1, 60, 80), Note(1, 1, 62, 80)) == 2.0
    assert streaming_cost(Note(0, 1, 60, 80), Note(0, 1, 62, 80)) == 2.0 + 2.0
    print("  streaming_cost: OK")


if __name__ == "__main__":
    _main()
