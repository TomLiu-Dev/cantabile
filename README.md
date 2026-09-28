# cantabile

cantabile takes a MIDI file and renders it on a sampled piano so that it
sounds played by a person. Every written note is kept; what it adds is the
performance: the melody brought out, dynamics, pedalling, tempo that breathes,
velocity-dependent key timing (softer notes sound slightly later), damper and
string resonance.

**Listen first** (same piano, same mix; first 20 s is the MIDI played as written, then the same 20 s played by cantabile):
[Mozart, Rondo alla Turca](https://github.com/TomLiu-Dev/cantabile/releases/download/v0.1.0/turkish_march_AB_compare.mp3) ·
[Vivaldi, Spring](https://github.com/TomLiu-Dev/cantabile/releases/download/v0.1.0/spring_AB_compare.mp3)

```bash
pip install -e .
python -m cantabile download-piano        # piano samples, ~400 MB, once
python -m cantabile play examples/spring.mid -o spring.mp3
```

Needs Python 3.9+ and ffmpeg. The rest of this page is in Chinese.

把 MIDI 用钢琴采样弹出来, 听起来像人在弹。谱子上的音一个都不改, 只加演奏:
旋律突出、力度起伏、踏板、速度有呼吸、按键时间差 (弱的音稍晚一点响)、制音器和
琴弦共鸣。

另外还有一个编配模式: 给一条旋律加和弦进行, 生成钢琴伴奏。

## 先听听

同一台钢琴、同一套混音。每段前 20 秒是 MIDI 原样播放 (力度都一样、速度像节拍器、不踩踏板), 后 20 秒是同一段用 cantabile 弹的:

- 莫扎特《土耳其进行曲》: [对比](https://github.com/TomLiu-Dev/cantabile/releases/download/v0.1.0/turkish_march_AB_compare.mp3) · [完整版](https://github.com/TomLiu-Dev/cantabile/releases/download/v0.1.0/turkish_march_B_cantabile.mp3)
- 维瓦尔第《四季·春》第一乐章 (弦乐合奏谱直接放到钢琴上弹): [对比](https://github.com/TomLiu-Dev/cantabile/releases/download/v0.1.0/spring_AB_compare.mp3) · [完整版](https://github.com/TomLiu-Dev/cantabile/releases/download/v0.1.0/spring_B_cantabile.mp3)

## 安装

需要 Python 3.9 或更新。

```bash
git clone https://github.com/TomLiu-Dev/cantabile.git
cd cantabile
pip install -e .
```

系统里要有 `ffmpeg` (macOS: `brew install ffmpeg`, Debian/Ubuntu:
`apt install ffmpeg`)。

钢琴采样不在仓库里, 第一次用先下载 (约 400 MB, 放在 `~/.cantabile/piano/`):

```bash
python -m cantabile download-piano
```

下载的是 Alexander Holm 的 Salamander Grand Piano V3 (SFZ 格式, CC-BY 3.0),
来源 FreePats, 也可以手动下载:
<https://freepats.zenvoid.org/Piano/acoustic-grand-piano.html>
选 `SalamanderGrandPianoV3+20161209_44khz16bit.tar.xz` (约 394 MB)。别的 SFZ
钢琴应该也能用, 但下面的数字都是用这个测的。

## 用法

```bash
python -m cantabile play 曲子.mid -o 曲子.mp3
```

- `-o`: 输出文件, 默认是和 MIDI 同名的 `.mp3`; 写 `.wav` 就输出 wav。
- `--sfz`: 钢琴采样。不写的话先看环境变量 `CANTABILE_SFZ`, 再找 `download-piano` 下载的那个。
  两个都没有的话只输出一个处理过的 `.mid`。
- `--bpm`: 速度, 默认用 MIDI 里写的。MIDI 里是 60 (很多导出软件不写速度时的
  默认值) 的话会先用 100, 最好自己指定。
- `--seed`: 随机种子, 默认 0。同样的种子结果一样。

## 例子

`examples/` 里有两首, 都来自 Mutopia Project, 出处和许可见
[examples/README.md](examples/README.md)。

```bash
export CANTABILE_SFZ=path/to/SalamanderGrandPianoV3.sfz
python -m cantabile play examples/turkish_march.mid -o turkish_march.mp3 --bpm 120
python -m cantabile play examples/spring.mid -o spring.mp3
```

- `turkish_march.mid`: 莫扎特 K.331 第三乐章 (土耳其进行曲), 钢琴原作。
  这个文件里的速度写的是 60, 所以上面加了 `--bpm`。
- `spring.mid`: 维瓦尔第《四季》"春" 第一乐章, 原来是独奏小提琴加弦乐,
  这里全部放到钢琴上弹, `solo` 音轨当旋律。

## 编配模式

```bash
python -m cantabile 曲名 --sfz path/to/piano.sfz -o 曲名.mp3
python -m cantabile 曲名 --audit-only     # 只检查, 不渲染
python -m cantabile --list                # 列出能用的曲子
```

仓库里不带曲子。自己的曲子写在 `~/.cantabile/corpus/*.py` 里 (或者环境变量
`CANTABILE_CORPUS` 指定的目录), 每个文件定义一个 `TUNES` 字典, 值是
`cantabile.corpus.Tune`: 旋律 `[(拍, 音高, 时长)]`, 和弦 `{拍: (根音, 性质, 低音)}`,
拍号、小节数、遍数等。格式可以参考 `tests/lianxiqu.py` 里的三首小练习曲
(4/4、3/4、6/8), 详细字段见 [docs/adding-a-tune.md](docs/adding-a-tune.md)。
其他参数 (`--style`、`--rules`、`--resonance`、`--room` 等) 看
`python -m cantabile --help`。

## 原理

1. 读 MIDI, 分出旋律、低音、内声部, 推断和声 (只用来决定什么时候换踏板)。
2. 力度: 旋律比伴奏响一些, 乐句有起伏, 重拍有重音, 再加上 KTH 演奏规则。
3. 旋律长音下面伴奏自动让一让, 不删音。
4. 按和声换踏板 (和声换之前一点抬起, 换了之后马上再踩)。
5. 力度的随机起伏, 并往实测的真人力度分布上靠一半。
6. 时间: 按乐句的速度弧线, 加上按键时间差 (Goebl 2001), 最后才做。
7. SFZ 采样渲染 (修正过的力度曲线、制音器), 加共鸣, ffmpeg 做母带。

各层的说明见 [docs/architecture.md](docs/architecture.md)。

## 效果

判断"像不像真人"不能靠自己听, 听多了就听不出来了。所以用一个分类器
(`eval/discriminator.py`): 30 个声学特征, 4 秒一个窗口, 逻辑回归, 学着区分
真人录音和渲染, 按录音留一交叉验证。准确率 0.5 表示分不出来, 1.0 表示总能分出来。

对照是两段真人钢琴录音 (一段 3/4, 一段 4/4)。一路改下来, 准确率从 0.93
降到 0.51; 渲染和第一段录音同一首曲子时是 0.29。每一步改了什么见
[docs/research.md](docs/research.md)。

这个数字的意思只是这个分类器用这些特征分不出来, 不是说人耳一定听不出来。
参照只有两段录音, 也没有做过听众测试。
要自己跑的话: `pip install -e ".[eval]"`, 然后
`python eval/discriminator.py --real a.mp3 b.mp3 --fake ours.mp3`。

## 局限

- 只有钢琴。弦乐、合奏的 MIDI 也会全部放到钢琴上弹。
- 旋律是自动找的: 音轨名里有 `solo` / `melody` 就用那个音轨, 否则取最高那个
  音轨每个时刻的最高音。找错了整首的平衡就不对。
- 参数是按两段录音校准的, 风格比较单一, 不是每种曲子都合适。
- 出过的问题和怎么查出来的, 记在
  [docs/why-it-sounds-wrong.md](docs/why-it-sounds-wrong.md)。

## 许可

代码是 MIT, 见 [LICENSE](LICENSE)。`examples/spring.mid` 例外, 它是
CC BY-SA 3.0, 保持原许可, 见 [examples/README.md](examples/README.md)。
钢琴采样和参照录音都不在仓库里。

## 参考

- A. Friberg, R. Bresin, J. Sundberg, "Overview of the KTH rule system for
  musical performance", Advances in Cognitive Psychology 2(2-3), 2006.
  (`cantabile/perform/kth.py`)
- W. Goebl, "Melody lead in piano performance: expressive device or
  artifact?", JASA 110(1), 2001. (`cantabile/perform/timing.py`)
- RenCon 2025: Revival of the Expressive Performance Rendering Competition,
  arXiv:2605.02059.
- E. Chew, X. Wu, "Separating voices in polyphonic music: a contig mapping
  approach", CMMR 2004, LNCS 3310, 2005. (`cantabile/analyze/streams.py`)
- Salamander Grand Piano V3, Alexander Holm, CC-BY 3.0, FreePats.
- Mutopia Project, <https://www.mutopiaproject.org/>, 例子里的两个 MIDI。
