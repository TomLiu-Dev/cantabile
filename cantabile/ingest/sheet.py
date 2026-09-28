"""看图读旋律, 跟 ingest.midi 互相核对用。尺寸都按线间距算, 250 dpi 以上都能读。"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from ..model import Finding, Report
from ..theory import name

__all__ = ["render_pdf", "load_image", "find_staves", "find_noteheads",
           "find_accidentals", "pitch_at", "read_melody", "find_barlines",
           "key_signature", "cross_check"]

_DIATONIC = (0, 2, 4, 5, 7, 9, 11)
_SHARP_ORDER = (3, 0, 4, 1, 5, 2, 6)          # F C G D A E B
_FLAT_ORDER = (6, 2, 5, 1, 4, 0, 3)           # B E A D G C F
# 各谱号最下面一条线的音名序号和八度
_BOTTOM_LINE = {"treble": (2, 4),             # E4
                "bass": (4, 2),               # G2
                "alto": (3, 3),               # F3
                "tenor": (1, 3)}              # D3


def render_pdf(pdf_path, dpi=400, out_dir=None):
    """用 pdftoppm 转成 PNG, 返回每页路径。低于 250 dpi 符头和谱线就分不开了。"""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    if shutil.which("pdftoppm") is None:
        raise RuntimeError("pdftoppm (poppler) is not on PATH")

    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="cantabile-score-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    qianzhui = out_dir / pdf_path.stem

    subprocess.run(["pdftoppm", "-r", str(int(dpi)), "-png",
                    "-aa", "yes", "-aaVector", "yes",
                    str(pdf_path), str(qianzhui)],
                   check=True, capture_output=True)

    yemian = sorted(out_dir.glob(f"{pdf_path.stem}-*.png"),
                    key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    if not yemian:
        raise RuntimeError(f"pdftoppm produced no pages for {pdf_path}")
    return yemian


def load_image(src, threshold=0.62):
    """转成墨迹的布尔图。threshold 是相对页面白色的比例, 灰的扫描件也能用。"""
    if isinstance(src, np.ndarray):
        arr = src
        if arr.dtype == bool:
            return arr
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
    else:
        if not isinstance(src, Image.Image):
            src = Image.open(str(src))
        arr = np.asarray(src.convert("L"))
    arr = arr.astype(np.float32)
    baise = float(np.percentile(arr, 95)) or 255.0
    return arr < threshold * baise


def find_staves(img, min_width=0.35, tol=0.30):
    """找五线谱, 返回 [{'lines', 'spacing', 'x0', 'x1'}], 从上到下。"""
    bw = load_image(img)
    h, w = bw.shape
    jishu = bw.sum(axis=1)
    hang = np.flatnonzero(jishu > min_width * w)
    if hang.size == 0:
        return []

    # 相邻的行并起来, 一条谱线有两三个像素粗
    xiandai, start = [], hang[0]
    prev = hang[0]
    for y in hang[1:]:
        if y - prev > 2:
            xiandai.append((start, prev))
            start = y
        prev = y
    xiandai.append((start, prev))
    zhongxin = [0.5 * (a + b) for a, b in xiandai]

    wuxianpu = []
    i = 0
    while i + 4 < len(zhongxin):
        wutiao = zhongxin[i:i + 5]
        jianju = [wutiao[j + 1] - wutiao[j] for j in range(4)]
        mean = sum(jianju) / 4.0
        if mean > 2 and all(abs(g - mean) <= tol * mean for g in jianju):
            xs = np.flatnonzero(bw[int(round(wutiao[0])):int(round(wutiao[4])) + 1].any(axis=0))
            wuxianpu.append({"lines": [float(y) for y in wutiao],
                             "spacing": float(mean),
                             "x0": int(xs[0]) if xs.size else 0,
                             "x1": int(xs[-1]) if xs.size else w - 1})
            i += 5
        else:
            i += 1
    return wuxianpu


def _staff_band(bw, staff, above=2.0, below=2.0):
    # 上下多留一点给加线
    sp = staff["spacing"]
    y0 = max(0, int(round(staff["lines"][0] - above * sp)))
    y1 = min(bw.shape[0], int(round(staff["lines"][4] + below * sp)) + 1)
    return bw[y0:y1], y0


def find_noteheads(img, staff, above=2.0, below=2.0, min_fill=0.55,
                   snap=0.33):
    """找一行谱上的符头。先填小洞 (空心符头), 再腐蚀掉谱线和符干。"""
    bw = load_image(img)
    band, y_off = _staff_band(bw, staff, above, below)
    sp = staff["spacing"]

    k = max(3, int(round(sp * 0.34)) | 1)           # 奇数, 400 dpi 大约 7 像素
    struct = np.ones((k, k), bool)

    # 只填符头大小的洞; 小节线和符干围出来的空白要宽得多
    quantian = ndi.binary_fill_holes(band)
    dong = quantian & ~band
    lab, n = ndi.label(dong)
    shixin = band.copy()
    liuxia = []
    for idx, sl in enumerate(ndi.find_objects(lab), start=1):
        ys, xs = sl
        if (xs.stop - xs.start) <= 1.5 * sp and (ys.stop - ys.start) <= 1.0 * sp:
            liuxia.append(idx)
    if liuxia:
        shixin |= np.isin(lab, liuxia)

    hexin = ndi.binary_erosion(shixin, structure=struct)
    lab, n = ndi.label(hexin)
    if not n:
        return []

    yintou = []
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs, start=1):
        ys, xs = sl
        h = ys.stop - ys.start
        w = xs.stop - xs.start
        area = int((lab[sl] == i).sum())
        # 符头宽比高大, 能排除大部分文字; 1.85 是给全音符留的
        if not (0.45 * sp <= h <= 1.10 * sp and 0.55 * sp <= w <= 1.85 * sp):
            continue
        if w < 0.95 * h:
            continue
        if area < 0.10 * sp * sp:
            continue
        if area < min_fill * h * w:
            continue
        cy, cx = ndi.center_of_mass(lab[sl] == i)
        y = ys.start + cy + y_off
        x = xs.start + cx

        # 不在线上也不在间里的是文字
        steps = (staff["lines"][4] - y) / (sp / 2.0)
        if abs(steps - round(steps)) > snap:
            continue

        # 实心空心看填洞之前的墨
        y0 = max(0, int(y - y_off - 0.45 * sp))
        y1 = int(y - y_off + 0.45 * sp) + 1
        x0 = max(0, int(x - 0.45 * sp))
        x1 = int(x + 0.45 * sp) + 1
        patch = band[y0:y1, x0:x1]
        density = float(patch.mean()) if patch.size else 1.0
        yintou.append({"x": float(x), "y": float(y), "filled": density > 0.80,
                       "w": int(w), "h": int(h), "area": area,
                       "density": round(density, 3)})

    yintou.sort(key=lambda d: d["x"])
    return yintou


def pitch_at(y, staff, clef="treble", sharps=0, flats=0):
    """y 处符头的 MIDI 音高, 从最下面一条线每半个间距一个音级。"""
    if clef not in _BOTTOM_LINE:
        raise ValueError(f"unsupported clef {clef!r}")
    sp = staff["spacing"]
    steps = int(round((staff["lines"][4] - y) / (sp / 2.0)))
    letter0, octave0 = _BOTTOM_LINE[clef]
    idx = letter0 + steps
    letter = idx % 7
    octave = octave0 + idx // 7
    pitch = 12 * (octave + 1) + _DIATONIC[letter]
    if sharps and letter in _SHARP_ORDER[:sharps]:
        pitch += 1
    if flats and letter in _FLAT_ORDER[:flats]:
        pitch -= 1
    return pitch


def key_signature(img, staff, clef="treble"):
    """读调号, 返回 (sharps, flats)。降号的墨偏下, 升号居中。"""
    bw = load_image(img)
    band, y_off = _staff_band(bw, staff, 1.2, 1.2)
    sp = staff["spacing"]
    qudiao = _strip_rules(band, sp)

    lab, n = ndi.label(qudiao)
    fanwei = staff["x0"] + 7.0 * sp
    fuhao = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        if not (1.3 * sp <= h <= 2.9 * sp and 0.30 * sp <= w <= 1.15 * sp):
            continue
        if not (staff["x0"] + 0.5 * sp <= xs.start <= fanwei):
            continue
        mask = lab[sl] == i
        cy = float(ndi.center_of_mass(mask)[0]) / max(h - 1, 1)
        fuhao.append((xs.start, cy))
    if not fuhao:
        return 0, 0
    fuhao.sort()
    # 空出两个间距以上就是调号结束了
    run = [fuhao[0]]
    for g in fuhao[1:]:
        if g[0] - run[-1][0] > 2.2 * sp:
            break
        run.append(g)
    low = sum(1 for _, cy in run if cy > 0.56)
    if low > len(run) / 2.0:
        return 0, len(run)
    return len(run), 0


def _strip_rules(band, sp):
    # 去掉谱线, 再把符号上被切开的小缝补上
    puxian = ndi.binary_opening(band, structure=np.ones((1, int(round(3 * sp))), bool))
    qiao = max(3, int(round(0.30 * sp)) | 1)
    return ndi.binary_closing(band & ~puxian, structure=np.ones((qiao, 1), bool))


def find_accidentals(img, staff, above=2.0, below=2.0):
    """找变音记号 [{'x', 'x0', 'y', 'kind', 'step', 'off'}], x 是右边缘。

    降号墨偏下, 升号对称, 还原号左竖比右竖低。
    """
    bw = load_image(img)
    band, y_off = _staff_band(bw, staff, above, below)
    sp = staff["spacing"]
    qudiao = _strip_rules(band, sp)
    lab, n = ndi.label(qudiao)

    jieguo = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        ys, xs = sl
        h, w = ys.stop - ys.start, xs.stop - xs.start
        # 升号可能比 3 个间距略高
        if not (1.15 * sp <= h <= 3.4 * sp and 0.28 * sp <= w <= 1.20 * sp):
            continue
        mask = lab[sl] == i
        if mask.mean() > 0.70:                      # 实心块, 不是符号
            continue
        cy = float(ndi.center_of_mass(mask)[0]) / max(h - 1, 1)
        q = max(1, int(round(w * 0.30)))
        left, right = mask[:, :q], mask[:, -q:]
        if not (left.any() and right.any()):
            continue
        lcy = float(ndi.center_of_mass(left)[0]) / max(h - 1, 1)
        rcy = float(ndi.center_of_mass(right)[0]) / max(h - 1, 1)

        if cy > 0.58:
            kind = "flat"
            bowl = mask[int(0.45 * h):]
            y = ys.start + 0.45 * h + float(ndi.center_of_mass(bowl)[0]) + y_off
        elif lcy - rcy > 0.15:
            kind = "natural"
            y = ys.start + float(ndi.center_of_mass(mask)[0]) + y_off
        else:
            kind = "sharp"
            y = ys.start + float(ndi.center_of_mass(mask)[0]) + y_off

        steps = (staff["lines"][4] - y) / (sp / 2.0)
        jieguo.append({"x": float(xs.stop), "x0": float(xs.start), "y": float(y),
                       "kind": kind, "step": int(round(steps)),
                       "off": abs(steps - round(steps))})
    jieguo.sort(key=lambda d: d["x"])
    return jieguo


_ACCIDENTAL_SHIFT = {"sharp": 1, "flat": -1, "natural": 0}


def find_barlines(img, staff, cover=0.90, merge=None):
    """小节线的 x 位置。符干不是从第一线到第五线, 靠这个排除; 双小节线并成一条。"""
    bw = load_image(img)
    sp = staff["spacing"]
    y0 = int(round(staff["lines"][0]))
    y1 = int(round(staff["lines"][4])) + 1
    band = bw[y0:y1]
    need = cover * band.shape[0]
    cols = np.flatnonzero(band.sum(axis=0) >= need)
    if cols.size == 0:
        return []

    reach = max(3, int(round(0.30 * sp)))
    lo = max(0, y0 - int(round(1.5 * sp)))
    hi = min(bw.shape[0], y1 + int(round(1.5 * sp)))
    houxuan = []
    for x in cols:
        col = bw[lo:hi, x]
        mid = (y0 + y1) // 2 - lo
        a = mid
        while a > 0 and col[a - 1]:
            a -= 1
        b = mid
        while b < col.size - 1 and col[b + 1]:
            b += 1
        top, bot = lo + a, lo + b
        if not (y0 - reach <= top <= y0 + reach and y1 - reach <= bot <= y1 + reach):
            continue
        houxuan.append(x)
    cols = np.array(houxuan)
    if cols.size == 0:
        return []
    gap = merge if merge is not None else 0.7 * sp
    jieguo, zu = [], [cols[0]]
    for x in cols[1:]:
        if x - zu[-1] <= gap:
            zu.append(x)
        else:
            jieguo.append(float(sum(zu)) / len(zu))
            zu = [x]
    jieguo.append(float(sum(zu)) / len(zu))
    return jieguo


def read_melody(path, page=1, clef="treble", dpi=400, sharps=None, flats=None,
                out_dir=None, detail=False):
    """从谱图或 PDF 某页读旋律 [(x, pitch, filled)]; 不读节奏。detail=True 返回带小节号等的 dict。"""
    src = Path(path)
    if src.suffix.lower() == ".pdf":
        yemian = render_pdf(src, dpi=dpi, out_dir=out_dir)
        if not 1 <= page <= len(yemian):
            raise ValueError(f"page {page} outside 1..{len(yemian)}")
        src = yemian[page - 1]
    bw = load_image(src)

    wuxianpu = find_staves(bw)
    if not wuxianpu:
        raise RuntimeError(f"no staves found in {src}")

    jieguo = []
    bar = 1
    for si, staff in enumerate(wuxianpu):
        ks, kf = key_signature(bw, staff, clef)
        sh = ks if sharps is None else sharps
        fl = kf if flats is None else flats
        xiaojiexian = find_barlines(bw, staff)
        # 跳过行首的谱号、调号、拍号
        sp = staff["spacing"]
        music_x0 = staff["x0"] + 3.0 * sp
        bianyin = find_accidentals(bw, staff)
        yongle = set()
        # 临时记号管到本小节结束
        youxiao = {}
        prev_bar = None

        for head in find_noteheads(bw, staff):
            if head["x"] < music_x0:
                continue
            steps = int(round((staff["lines"][4] - head["y"]) / (sp / 2.0)))
            n_before = sum(1 for b in xiaojiexian if b < head["x"] - 0.2 * sp)
            this_bar = bar + n_before
            if this_bar != prev_bar:
                youxiao = {}
                prev_bar = this_bar

            for k, a in enumerate(bianyin):
                if k in yongle or a["step"] != steps:
                    continue
                if 0.25 * sp <= head["x"] - a["x"] <= 3.2 * sp:
                    youxiao[steps] = a["kind"]
                    yongle.add(k)
                    break

            pitch = pitch_at(head["y"], staff, clef, sharps=sh, flats=fl)
            if steps in youxiao:
                natural = pitch_at(head["y"], staff, clef, sharps=0, flats=0)
                pitch = natural + _ACCIDENTAL_SHIFT[youxiao[steps]]

            rec = {"x": head["x"], "pitch": pitch, "filled": head["filled"],
                   "staff": si, "bar": this_bar, "name": name(pitch),
                   "y": head["y"], "accidental": youxiao.get(steps)}
            jieguo.append(rec)
        zhehang = sum(1 for b in xiaojiexian if b > music_x0)
        bar += max(1, zhehang)
    if detail:
        return jieguo
    return [(r["x"], r["pitch"], r["filled"]) for r in jieguo]


def _pitches(melody):
    jieguo = []
    for ev in melody:
        if isinstance(ev, dict):
            jieguo.append(int(ev["pitch"]))
        elif len(ev) == 3:
            a, b, c = ev
            # midi 是 (beat, pitch, dur), sheet 是 (x, pitch, filled)
            jieguo.append(int(b))
        else:
            raise ValueError(f"cannot read a pitch from {ev!r}")
    return jieguo


def _align(a, b, gap=-1.6, octave=-0.4):
    # Needleman-Wunsch; 差八度算半对, 免得报一串错音
    n, m = len(a), len(b)
    defen = np.zeros((n + 1, m + 1))
    back = np.zeros((n + 1, m + 1), np.int8)
    defen[1:, 0] = np.arange(1, n + 1) * gap
    defen[0, 1:] = np.arange(1, m + 1) * gap
    back[1:, 0] = 1
    back[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                s = 1.0
            elif (a[i - 1] - b[j - 1]) % 12 == 0:
                s = octave
            else:
                s = -1.0 - min(abs(a[i - 1] - b[j - 1]), 12) * 0.05
            best = (defen[i - 1, j - 1] + s, 0)
            if defen[i - 1, j] + gap > best[0]:
                best = (defen[i - 1, j] + gap, 1)
            if defen[i, j - 1] + gap > best[0]:
                best = (defen[i, j - 1] + gap, 2)
            defen[i, j], back[i, j] = best

    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        d = back[i, j]
        if d == 0 and i > 0 and j > 0:
            pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif d == 1 and i > 0:
            pairs.append((i - 1, None))
            i -= 1
        else:
            pairs.append((None, j - 1))
            j -= 1
    pairs.reverse()
    return pairs


def cross_check(midi_melody, sheet_melody, label_midi="midi",
                label_sheet="sheet") -> Report:
    """把 MIDI 读出来的和看谱读出来的对齐, 报告不一样的地方。"""
    a, b = _pitches(midi_melody), _pitches(sheet_melody)
    beats = [ev["beat"] if isinstance(ev, dict) else ev[0] for ev in midi_melody]
    rep = Report()

    if not a or not b:
        rep.add(Finding("cross_check", 0.0,
                        f"{label_midi} has {len(a)} notes, {label_sheet} has {len(b)}"
                        " - nothing to align", "error"))
        rep.stats.update({"midi_notes": len(a), "sheet_notes": len(b),
                          "agreement": 0.0})
        return rep

    pairs = _align(a, b)
    same = subs = octs = only_a = only_b = 0
    for i, j in pairs:
        beat = float(beats[i]) if i is not None else (
            float(beats[min(len(beats) - 1, max(0, (j or 0)))]))
        if i is not None and j is not None:
            if a[i] == b[j]:
                same += 1
            elif (a[i] - b[j]) % 12 == 0:
                octs += 1
                rep.add(Finding("cross_check", beat,
                                f"octave: {label_midi} {name(a[i])} vs "
                                f"{label_sheet} {name(b[j])}", "warn"))
            else:
                subs += 1
                rep.add(Finding("cross_check", beat,
                                f"pitch: {label_midi} {name(a[i])} vs "
                                f"{label_sheet} {name(b[j])}", "error"))
        elif i is not None:
            only_a += 1
            rep.add(Finding("cross_check", beat,
                            f"{label_sheet} has no note for {label_midi} "
                            f"{name(a[i])}", "error"))
        else:
            only_b += 1
            rep.add(Finding("cross_check", beat,
                            f"{label_midi} has no note for {label_sheet} "
                            f"{name(b[j])} (sheet note {j})", "error"))

    total = max(len(a), len(b))
    rep.stats.update({
        "midi_notes": len(a), "sheet_notes": len(b),
        "matched": same, "pitch_disagreements": subs,
        "octave_disagreements": octs,
        f"only_in_{label_midi}": only_a, f"only_in_{label_sheet}": only_b,
        "agreement": round(same / float(total), 4) if total else 0.0,
    })
    return rep
