# 给编配模式加曲子

编配模式 (`python -m cantabile 曲名`) 用的曲子不在仓库里, 放在
`~/.cantabile/corpus/` 下面的 `.py` 文件里, 或者环境变量 `CANTABILE_CORPUS`
指定的目录。每个文件定义一个 `TUNES` 字典, 键是曲名 (大写), 值是
`cantabile.corpus.Tune`。第一次用到时这些文件会被执行并合并。

`tests/lianxiqu.py` 是现成的例子: 三首 16 小节的练习曲, 4/4、3/4 和带弱起的
6/8。它里面的字典叫 `LIANXIQU`, 拿来当曲库用的话加一行 `TUNES = LIANXIQU`。

## Tune 的字段

```python
from cantabile.corpus import Tune

WODEQU = Tune(
    name="WODEQU", title="我的曲子", author="-", composer="...", year="2026",
    key="C", meter="4/4", beats_per_bar=4, bars=16,
    melody=[(0.0, 64, 1.0), (1.0, 67, 1.0), ...],   # (拍, MIDI 音高, 时长)
    harmony={0.0: ("C", "", "C"), 4.0: ("G", "7", "B"), ...},  # 拍: (根音, 性质, 低音)
    verses=3, form="strophic")

TUNES = {WODEQU.name: WODEQU}
```

- 拍都按四分音符算, 从 0 开始。
- `harmony` 里每个和弦一直持续到下一个。第三项是低音, 转位要写出来
  (`("G", "7", "B")` 就是 G7/B)。和弦性质是 `cantabile/theory.py` 里
  `QUALITIES` 的键 (`""`、`"m"`、`"7"`、`"m7"`、`"maj7"`、`"dim"`、`"sus4"` 等)。
- `verses`: 整首弹几遍。前面有一小段前奏, 然后每遍弹一次旋律, 不转调。
- `beats_per_bar` 和 `compound`: 6/8 写成 `beats_per_bar=3, compound=True`,
  重拍在 0 和 1.5。
- `pickup`: 弱起的长度。弱起放在曲子最后一小节的末尾, 这样每一遍都是整数个
  小节; 最后一遍会去掉它。见 `LIUBA`。
- `chorus_at`: 副歌从第几拍开始; `refrain_tempo`: 副歌相对主歌的速度倍数。
- `marks`: 谱上的力度记号, `{拍: 力度增减}`, 一直保持到下一个记号。
- `fermatas`: `((拍, 时值), ...)`, 延长记号, 那里会放慢。
- `verified`、`source`、`notes`: 旋律核对过没有、从哪来、有什么看起来像错但
  其实是故意的地方 (比如经过音)。自己的曲子可以随便写, 但写清楚对以后查问题
  有帮助。

## 从四声部 MIDI 导入

如果手上有一份四声部 MIDI (每个声部一个音轨, 或者上面两个声部在一个音轨),
可以用 `cantabile.ingest.satb` 提取旋律和和弦:

```python
from cantabile.ingest import satb

tracks = satb.voices("qu.mid")             # 每个音轨一个列表, 从高到低
LENGTH = 16 * 4                            # 小节数 x 每小节拍数
one = satb.one_pass(tracks, LENGTH, start=0.0)   # 截出一遍
melody = satb.soprano(one[0])              # 上面两个声部同一音轨时加 split_top=True
harmony = satb.harmony_from_voices(one, melody, length=LENGTH, key_pc=0)
print(satb.chart(harmony, 4, 16))          # 逐小节的和弦表, 拿去和谱子对
```

导出来的东西一定要和谱子对一遍:

- 旋律逐个音核对。相邻的同音要分开 (见 why-it-sounds-wrong.md 第 7 条)。用了
  `split_top=True` 的话特别注意长音下面的内声部有没有混进旋律 (第 18 条)。
- 和弦表: 经过音可能被当成换和弦, 有歧义时会偏向 `key_pc` 调的主要三和弦。
  和谱子不一样的地方手动改。

## 检查

```bash
python -m cantabile WODEQU --audit-only
```

会打印音符数、时长、密度和检查结果; 有 `error` (错音或没声音) 时退出码是 1。
然后先单独听旋律, 再听整个编配。
