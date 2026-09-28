"""下载钢琴采样 (Salamander Grand Piano V3, Alexander Holm, CC-BY 3.0)。"""
import sys
import tarfile
import urllib.request
from pathlib import Path

DIZHI = ("https://freepats.zenvoid.org/Piano/SalamanderGrandPiano/"
         "SalamanderGrandPianoV3+20161209_44khz16bit.tar.xz")
MULU = Path.home() / ".cantabile" / "piano"


def zhao_sfz(mulu=MULU):
    """找已经下载好的 .sfz, 没有就返回 None。"""
    mulu = Path(mulu)
    if not mulu.is_dir():
        return None
    houxuan = sorted(mulu.rglob("*.sfz"), key=lambda p: ("Retuned" in p.name, len(str(p))))
    return houxuan[0] if houxuan else None


def xiazai(dizhi=DIZHI, mulu=MULU):
    mulu = Path(mulu)
    yiyou = zhao_sfz(mulu)
    if yiyou:
        print(f"已经有了: {yiyou}")
        return yiyou
    mulu.mkdir(parents=True, exist_ok=True)
    bao = mulu / "piano.tar.xz"
    print(f"下载钢琴采样 (大约 400 MB) -> {mulu}")

    def jindu(kuai, kuaidaxiao, zong):
        if zong > 0:
            bai = min(100, kuai * kuaidaxiao * 100 // zong)
            sys.stdout.write(f"\r  {bai}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(dizhi, bao, jindu)
    print("\n解压 ...")
    with tarfile.open(bao, "r:*") as tar:
        for m in tar.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts:
                raise RuntimeError(f"压缩包里有奇怪的路径: {m.name}")
        tar.extractall(mulu)
    bao.unlink()
    sfz = zhao_sfz(mulu)
    if not sfz:
        raise RuntimeError("解压完了但是没找到 .sfz 文件")
    print(f"好了: {sfz}")
    return sfz
