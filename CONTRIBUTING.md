# 参与

欢迎提 issue 和 PR。这个项目有个麻烦的地方: 改坏了往往什么都不报错, 生成的文件
照样能播放, 调也对, 长度也对, 只是听起来有点不对。下面几条规矩都是从这种 bug
里来的, 具体经过见 [docs/why-it-sounds-wrong.md](docs/why-it-sounds-wrong.md)。

## 环境

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]"
```

最低支持 Python 3.9: 模块开头保留 `from __future__ import annotations`, 不要用
`match`, 运行时求值的地方不要写 `X | Y`。渲染需要 ffmpeg 和 SFZ 钢琴, 见 README。

## 测试

不依赖测试框架, 每个文件自己跑, 有失败就返回非零:

```bash
for t in tests/test_*.py; do python3 $t; done
```

测试不需要音频工具和采样。覆盖 `theory` 的不变量、所有和弦所有转位的配声、
`tests/lianxiqu.py` 里三首练习曲的完整生成和检查。

## 改了就要有测量

修一个"听起来不对"的问题, 要同时加上 (或指出) 能证明问题存在、也能证明已经
修好的东西: 一个 audit 检查、`Report.stats` 里的一个数、一个测试, 或者和
[docs/research.md](docs/research.md) 里参照数字的前后对比。PR 里说明是哪个。

原因是耳朵会漂。旋律太弱, 往上推, 结果推到力度 p90 94 (参照是 88), 听起来就
"一直在砸"。两边都要有界, 这样修过头也能测出来。

如果是听的人发现的问题, 在 docs/why-it-sounds-wrong.md 里按同样的格式加一条:
现象、原因、怎么检测、怎么修, 写实测数字。

## 代码规矩

- 不按固定音程放音。不写 `bass + 7`、`root + 4`。G7/B 上 bass + 7 是 F#, 和和弦
  里的 F 差半音。所有音高都从 `theory.chord_near` 或 `theory.stack` 取。故意的
  和弦外音要打 tag, 让 audit 能区分。
- 按音级找音的窗口不能小于 12 个半音, 否则会漏一个音级。`theory.pc_near` 会
  检查结果, 类似的函数也要这样做。
- 旋律不让。冲突时先挪或删 `COLOUR`, 再是 `INNER`。会顺带把 `MELODY` 音变短、
  变弱或删掉的代码就是错的。
- 结果要确定: 同一首曲子、同样的参数和种子, 在两个不同进程里跑出来的音要完全
  一样。不要用 `hash()` 生成种子 (每个进程加盐不同), 用 `zlib.crc32` 或整数。
- 新的测量方法先在已知答案的输入上验证, 再相信它。
- audit 检查只测量和报告, 不改 `Performance`。把数写进 `report.stats`;
  severity `error` 只用在错音和没声音上, 因为它会让 `Report.ok()` 失败。

## 报 bug

"听起来不对"就是合格的 bug 报告。请写上: 你听到了什么 (用自己的话), 能指出的话
是第几秒, 用的 MIDI 或曲子和完整命令 (包括 `--seed`), 编配模式的话附上
`--audit-only` 的输出, 以及 Python、ffmpeg 和钢琴采样的版本。如果 audit 什么都
没报但你还是听得出问题, 这种报告最有用: 说明缺了一个测量。

## PR

一个 PR 只做一件事, 方便二分定位回归。如果加了处理步骤之间的先后依赖, 在旁边
写注释说明为什么。提交的代码按 MIT 许可发布。
