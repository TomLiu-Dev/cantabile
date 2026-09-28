"""切分踏板: 换和弦前抬, 换后再踩. pedal 存 (beat, cc64), 0 抬 >=64 踩."""
from __future__ import annotations

from typing import List, Optional, Tuple

from ..model import Performance
from ..theory import Harmony

__all__ = ["harmonic_pedal", "pedal_events", "clear_at"]

UP = 0


def _tidy(shijian):
    jieguo = []
    shangci = None
    for beat, val in sorted((max(0.0, round(float(b), 6)), int(v))
                            for b, v in shijian):
        if jieguo and abs(jieguo[-1][0] - beat) < 1e-6:
            # 同一时刻取后一个
            jieguo[-1] = (beat, val)
            shangci = val
            continue
        if val == shangci:
            continue
        jieguo.append((beat, val))
        shangci = val
    return jieguo


def harmonic_pedal(perf: Performance, harmony: Harmony, lift: float = 0.025,
                   catch: float = 0.06, depth: int = 110) -> Performance:
    """lift: 换和弦前多少拍抬, catch: 换后多少拍踩."""
    out = perf.copy()
    if not out.notes:
        out.pedal = []
        return out
    diyige = min(n.onset for n in out.notes)
    end = out.length
    taban_shijian = [(diyige + catch, depth)]

    qishi = sorted(n.onset for n in out.notes)
    for start, _stop, _chord in harmony.spans():
        if start <= diyige + 1e-9 or start > end:
            continue
        # 琶音/提前的音在拍前, 要在最早那个之前抬
        tiqian = [b for b in qishi if start - 0.25 <= b < start]
        up = min(start, min(tiqian) if tiqian else start) - lift
        taban_shijian.append((up, UP))
        taban_shijian.append((start + catch, depth))

    taban_shijian.append((end, UP))
    out.pedal = _tidy(taban_shijian)
    return out


def pedal_events(perf: Performance) -> List[Tuple[float, int]]:
    return _tidy(perf.pedal)


def clear_at(perf: Performance, beat: float, catch: float = 0.06,
             depth: Optional[int] = None) -> Performance:
    """在 beat 处强制换踏板, 踏板本来是抬的就不动."""
    out = perf.copy()
    taban_shijian = _tidy(out.pedal)
    zhuangtai, dianping = UP, depth or 110
    for b, v in taban_shijian:
        if b <= beat + 1e-9:
            zhuangtai = v
            if v:
                dianping = v
    if not zhuangtai:
        out.pedal = taban_shijian
        return out
    taban_shijian = [(b, v) for b, v in taban_shijian
                     if not (beat - 1e-9 <= b <= beat + catch + 1e-9)]
    taban_shijian.append((beat, UP))
    taban_shijian.append((beat + catch, depth or dianping))
    out.pedal = _tidy(taban_shijian)
    return out


if __name__ == "__main__":
    from . import demo_performance

    perf, harmony = demo_performance()
    p = harmonic_pedal(perf, harmony)
    ev = pedal_events(p)
    print("%d pedal events over %.0f beats" % (len(ev), p.length))
    for b, v in ev[:8]:
        print("   beat %7.3f  cc64 %3d" % (b, v))

    assert ev == sorted(ev), "events out of order"
    assert all(b >= 0 for b, _ in ev)
    assert all(a[1] != b[1] for a, b in zip(ev, ev[1:])), "redundant events"
    assert ev[-1][1] == UP, "pedal left down at the end"
    changes = [k for k, _, _ in harmony.spans() if 0 < k <= p.length]
    assert len(ev) == 2 * len(changes) + 2, ev
    for k in changes:
        assert any(abs(b - (k - 0.025)) < 1e-6 and v == UP for b, v in ev), k
        assert any(abs(b - (k + 0.06)) < 1e-6 and v == 110 for b, v in ev), k
    assert all(ev[i][0] < ev[i + 1][0] for i in range(len(ev) - 1))

    from .timing import NATURAL, humanize

    hp = harmonic_pedal(humanize(perf, NATURAL, seed=5), harmony)
    hev = pedal_events(hp)
    for k, _, _ in harmony.spans():
        if k <= 0 or k > hp.length:
            continue
        ups = [b for b, v in hev if v == UP and b < k + 0.06]
        early = [n.onset for n in hp.notes if k - 0.25 <= n.onset < k]
        if early:
            assert max(ups) <= min(early) + 1e-9, (k, max(ups), min(early))
    print("humanised: %d events, all lifts ahead of the new harmony" % len(hev))

    c = clear_at(p, 6.0)
    got = [e for e in pedal_events(c) if 5.9 <= e[0] <= 6.2]
    print("clear_at(6.0):", got)
    assert got == [(6.0, 0), (6.06, 110)]
    assert clear_at(p, p.length + 4.0).pedal == pedal_events(p), \
        "clear_at with the pedal up should be a no-op"
    assert perf.pedal == [], "input was mutated"
    print("pedal ok")
