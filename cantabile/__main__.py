import argparse
import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "download-piano":
        from .xiazai import xiazai
        xiazai()
        return 0
    if not argv or argv[0] != "play":
        from .cli import main as bianpei
        return bianpei(argv)

    ap = argparse.ArgumentParser(prog="cantabile play",
                                 description="把 MIDI 弹得像真人")
    ap.add_argument("midi")
    ap.add_argument("-o", "--out", help="输出文件, 默认和 MIDI 同名的 .mp3")
    ap.add_argument("--sfz", help="钢琴采样 .sfz, 也可以设 CANTABILE_SFZ")
    ap.add_argument("--bpm", type=float, help="速度, 默认用 MIDI 里的")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv[1:])

    from .play import play
    shuchu = play(a.midi, a.out, sfz=a.sfz, bpm=a.bpm, seed=a.seed)
    print(shuchu)
    return 0


if __name__ == "__main__":
    sys.exit(main())
