"""命令行: 编配一首曲子, 检查, 渲染。"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from .corpus import TUNES, get, names
from .piece import Form, Piece, default_tempo
from .arrange import styles
from .model import Performance


def make(tune_name, *, style="accompaniment", bpm=92, seed=0, verses=None,
         rules="accompaniment") -> Performance:
    """编配 + 演奏 + 人性化, 返回 Performance。"""
    from .perform import (voice_balance, phrase_arc, accents, clamp,
                          duck_under_sustain, harmonic_pedal, humanize)
    from .perform.timing import NATURAL
    qu = tune_name if hasattr(tune_name, "melody") else get(tune_name)
    st = styles.get(style)
    form = Form.strophic(qu, verses=verses)
    bpb = form.beats_per_bar
    # 副歌变速和延长号是曲子本身的, 直接做进速度里
    bili = getattr(qu, "refrain_tempo", 1.0) or 1.0
    yanchang = tuple(getattr(qu, "fermatas", ()) or ())
    factor = None
    if (bili != 1.0 and qu.chorus_at is not None) or yanchang:
        fanwei = []
        for sec in form.sections:
            lo, hi = sec.span(qu.length)
            t0 = (sec.start_bar - 1) * bpb - lo
            fanwei.append((t0 + lo, t0 + hi, t0))

        def factor(beat):
            for a, b, t0 in fanwei:
                if a <= beat < b:
                    u = beat - t0                        # 曲子里的拍
                    f = 1.0
                    if bili != 1.0 and qu.chorus_at is not None:
                        x = u - qu.chorus_at             # 进副歌第几拍
                        if x >= -1.0:
                            f = 1.0 + (bili - 1.0) * min(1.0, (x + 1.0) / 2.0)
                    for fb, fl in yanchang:
                        # 前一拍慢慢放慢, 然后差不多拉长一倍
                        if fb - 0.75 <= u < fb:
                            f *= 1.0 - 0.3 * (u - fb + 0.75) / 0.75
                        elif fb <= u < fb + fl:
                            f *= 0.5
                    return f
            return 1.0
    perf = Piece(qu, form, st, default_tempo(bpm, form.total_bars * bpb,
                             beats_per_bar=bpb, factor=factor),
                 qu.title).build(seed=seed)
    # 顺序不能乱: 先力度, 衰减看最终力度, 时间要用网格上的拍所以最后做
    perf = clamp(accents(phrase_arc(voice_balance(perf, melody_gain=5,
                                                  inner_cut=2, colour_cut=4)),
                         meter=bpb, compound=form.compound))
    # 表情之前存一份, 检查音高时用谱上的位置
    perf.meta["content"] = perf.copy()
    if rules and rules != "none":
        # KTH 演奏规则, 它不分旋律和伴奏, 所以 voice_balance 要先做
        from .perform.rules import PALETTES
        perf = PALETTES[rules].apply(perf)
    # 旋律长音下面伴奏放轻, 不去重弹旋律
    perf = clamp(duck_under_sustain(perf, grace=0.3, max_cut=26, rate=16.0,
                                    thin_after=1.0, thin_above=50))
    perf = harmonic_pedal(perf, perf.meta.get("harmony"))
    from .perform.dispersion import (motor_noise, intentional,
                                     match_distribution, REFERENCE_PROFILE)
    perf = intentional(motor_noise(perf, seed=seed), perf.meta.get("harmony"),
                       beats_per_bar=bpb)
    if REFERENCE_PROFILE:
        # 整首一起映射到真人力度分布, 不分声部; 只用一半强度, 保留原来的大起伏
        perf = match_distribution(perf, REFERENCE_PROFILE, strength=0.5)
    perf = humanize(perf, NATURAL, seed=seed)
    return perf


def _on_grid(perf, grid=0.25):
    q = perf.copy()
    for n in q.notes:
        n.onset = round(n.onset / grid) * grid
    return q.sorted()


def _coalesce_melody(perf, gap=0.06):
    """把 refresh 重弹的旋律音并回原来那个音, 真正的同音反复不动。"""
    q = perf.copy()
    xuanlv = sorted([n for n in q.notes if n.voice.name == "MELODY"],
                    key=lambda n: (n.pitch, n.onset))
    shan, i = set(), 0
    while i < len(xuanlv):
        j = i
        while (j + 1 < len(xuanlv) and xuanlv[j + 1].pitch == xuanlv[i].pitch
               and "refresh" in xuanlv[j + 1].tag
               and abs(xuanlv[j + 1].onset - xuanlv[j].end) <= gap):
            j += 1
        if j > i:
            xuanlv[i].dur = xuanlv[j].end - xuanlv[i].onset
            for k in range(i + 1, j + 1):
                shan.add(id(xuanlv[k]))
        i = j + 1
    q.notes = [n for n in q.notes if id(n) not in shan]
    return q.sorted()


def audit(perf, tune_name):
    """和弦音、旋律完整用表情前的那份检查; 撞音、遮盖、同度、延音用最终的。"""
    from .audit.harmony import check_chord_tones, check_clashes
    from .audit.melody import check_tune, check_never_masked, check_unisons
    from .perform.decay import sustain_report
    t = get(tune_name)
    neirong = perf.meta.get("content", perf)
    neirong.meta.setdefault("harmony", perf.meta.get("harmony"))
    neirong.meta.setdefault("repeats", perf.meta.get("repeats", ()))
    r = check_chord_tones(_on_grid(neirong), neirong.meta["harmony"])
    for f in (check_clashes(perf), check_never_masked(perf), check_unisons(perf),
              check_tune(_coalesce_melody(_on_grid(neirong)), t, tuple(neirong.meta["repeats"])),
              sustain_report(perf)):
        r.merge(f)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cantabile")
    ap.add_argument("tune", nargs="?", help=f"one of: {', '.join(names())}")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--style", default="accompaniment")
    ap.add_argument("--bpm", type=float, default=92)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules", default="accompaniment",
                    help="KTH palette: deadpan | nominal | romantic | "
                         "accompaniment | none")
    ap.add_argument("--expressive", action="store_true",
                    help="restore velocity-dependent timbre (band-split render)")
    ap.add_argument("--sfz", default=None,
                    help="render with a multi-velocity-layer SFZ library "
                         "instead of a SoundFont (the only way to get real "
                         "velocity-dependent timbre)")
    # gain 大了整首的响度起伏会被抹平 (录音 2.37 dB, 0.14 时只剩 1.92)
    ap.add_argument("--resonance", type=float, default=0.08, metavar="GAIN",
                    help="add sympathetic and damper-pedal resonance "
                         "(0.10-0.35; a sampler has none at all)")
    # 默认关: 合成的房间声会染色, 反而更容易听出不是真录音
    ap.add_argument("--room", type=float, default=0.0, metavar="WET",
                    help="share of the finished signal that is room sound "
                         "(0 disables it); see render/room.py")
    ap.add_argument("--soundfont", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    a = ap.parse_args(argv)

    if a.list or not a.tune:
        for n in names():
            t = TUNES[n]
            print(f"  {n:14s} {t.title:34s} {t.key:3s} {t.verses} verses  "
                  f"verified={t.verified}")
        return 0

    perf = make(a.tune, style=a.style, bpm=a.bpm, seed=a.seed, rules=a.rules)
    baogao = audit(perf, a.tune)
    miao = perf.tempo.seconds(perf.length)
    print(f"{perf.meta['title']}  {len(perf.notes)} notes  "
          f"{int(miao//60)}m{miao%60:04.1f}s  {len(perf.notes)/miao:.1f} notes/sec")
    print(baogao.summary(limit=4))
    if not baogao.ok():
        print(f"\n{len(baogao.errors)} error(s)")
    if a.audit_only:
        return 0 if baogao.ok() else 1

    shuchu = Path(a.out or f"{a.tune.lower()}.mp3")
    if a.sfz:
        from .render.sfz import SFZSampler
        from .render import master, audible_end
        import numpy as np, wave
        wav = shuchu.with_suffix(".wav")
        y = SFZSampler(a.sfz).render(perf)
        if a.resonance > 0:
            from .render.resonance import render as add_res, ResonanceConfig, events_from
            y = add_res(y.astype(np.float64), events_from(perf), 44100,
                        ResonanceConfig(gain=a.resonance, max_voices=260))
            y = y / max(1e-9, np.abs(y).max()) * 0.9
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
            w.writeframes((y * 32767).astype("<i2").tobytes())
        # 房间声在母带处理里加
        import dataclasses
        from .render.master import WARM_PIANO
        chain = dataclasses.replace(WARM_PIANO, room_wet=a.room)
        master(wav, shuchu, chain=chain, trim=(0.0, audible_end(wav) + 2))
        print(f"wrote {shuchu}")
        return 0 if baogao.ok() else 1
    if a.soundfont:
        from .render import SynthConfig, render, master, audible_end
        wav = shuchu.with_suffix(".wav")
        cfg = SynthConfig(a.soundfont)
        if a.expressive:
            from .render.timbre import render_expressive
            render_expressive(perf, cfg, wav)
        else:
            render(perf, cfg, wav)
        if a.resonance > 0:
            from .render.resonance import render as add_res, ResonanceConfig, events_from
            import numpy as np, subprocess, wave
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav),
                                  "-f", "f32le", "-ac", "1", "-ar", "44100", "-"],
                                 capture_output=True).stdout
            y = np.frombuffer(raw, dtype="<f4").copy()
            y = add_res(y, events_from(perf), 44100,
                        ResonanceConfig(gain=a.resonance, max_voices=260))
            y = y / max(1e-9, np.abs(y).max()) * 0.95
            with wave.open(str(wav), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100)
                w.writeframes((y * 32767).astype("<i2").tobytes())
        jiewei = audible_end(wav) + 2
        master(wav, shuchu, trim=(0.0, jiewei))
        print(f"wrote {shuchu}")
    else:
        from .render import to_midi
        to_midi(perf, shuchu.with_suffix(".mid"))
        print(f"wrote {shuchu.with_suffix('.mid')} (no --soundfont, MIDI only)")
    return 0 if baogao.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
