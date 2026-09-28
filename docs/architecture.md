# 结构

cantabile 是几层处理, 共用一个很小的数据模型 (`cantabile/model.py`)。每层吃进和
吐出的都是这些类型, 所以每层都能单独换掉或单独检查。每个生成的音都记着它属于
哪个声部、是哪段代码生成的; 生成音乐出错不会抛异常, audit 找到错音时只能靠这个
追到源头。

## 两条流水线

演奏模式 (`python -m cantabile play`, `cantabile/play.py`):

```
读 MIDI -> 分出旋律/低音/内声部 -> 推断和声 -> 力度 -> KTH 规则 -> 长音下伴奏让位
        -> 踏板 -> 力度起伏 -> 时间 -> SFZ 渲染 -> 共鸣 -> ffmpeg 母带
```

编配模式 (`python -m cantabile 曲名`, `cantabile/cli.py`):

```
曲库 (旋律 + 和弦) -> 编配 (左手织体, 右手和弦, 前奏 + 若干遍) -> 同样的演奏处理
                   -> 渲染; audit 检查音符和音频
```

## 包

| 包 | 做什么 |
|---|---|
| `theory` | 音级、`Chord`、`Harmony`、`chord_near`、`pc_near`、`stack`、`is_clash` |
| `model` | `Voice`、`Note`、`TempoMap`、`Performance`、`Finding`、`Report`、`run` |
| `play` | 演奏模式: 读 MIDI、找旋律、跑演奏处理、渲染 |
| `corpus` | `Tune` 和曲库加载 (`~/.cantabile/corpus/` 或 `CANTABILE_CORPUS`) |
| `ingest` | `satb`: 从四声部 MIDI 导入; `midi`: 读钢琴 MIDI; `sheet`: 从印刷谱读旋律 (需要 Pillow 和 poppler) |
| `analyze` | 和声推断、声部分离 (Chew & Wu 的 contig mapping 等)、判断哪条是旋律 |
| `arrange` | 配声、织体、风格、逐小节的左手规划 (`varying.py`)、可弹性 |
| `piece` | `Section`、`Form`、`Piece`、`default_tempo`: 把一首曲子铺成整个作品 |
| `perform` | 力度、衰减补偿、踏板、KTH 规则、力度分布、时间 |
| `render` | MIDI 输出、SFZ 采样器、共鸣、房间、母带、fluidsynth |
| `audit` | `harmony`、`melody`: 音符层面的检查; `balance`、`reference`: 音频测量 |

`theory`、`model`、音符层面的 audit 和编配只需要 numpy、scipy、mido; SFZ 采样器
用 soundfile 读采样。音频测量需要 librosa (`analysis` 可选依赖), 用到时才导入。

## 数据模型

`Voice` 说明一个音的作用: `MELODY`、`BASS`、`INNER`、`COLOUR`。`INNER` 和
`COLOUR` 是 `droppable` 的: 两个音冲突时 (同音碰撞、伴奏挤着旋律长音) 靠它决定
谁让。旋律永远不让。

`Note(onset, dur, pitch, vel, voice, tag)`: 起点和时长用拍, 真实时间尽量晚才由
`TempoMap` 算。`tag` 是生成它的代码的名字 (比如 `fill.fifth`), audit 报错时会
带上。

`TempoMap` 存 `(拍, bpm)` 锚点和一个可选的乐句速度函数; `seconds(beat)` 积分
得到秒数, 所有音频测量都靠它对准位置。

`Performance` 存 `notes`、`pedal` (`(拍, cc64)` 列表)、`tempo` 和 `meta`
(各步之间传东西, 比如和声)。

`Report` 收集 `Finding` (severity 是 `error`/`warn`/`info`) 和一个 `stats`
字典。`ok()` 在没有 `error` 时为真。stats 和 findings 一样重要: findings 抓
错误, stats 抓修过头。

## 处理顺序

一步就是 `Performance -> Performance` 的函数。每步只做一件事, 不动 `voice` 和
`tag`。顺序有讲究:

1. `voice_balance`、`phrase_arc`、`accents`、`clamp`: 旋律和伴奏的层次、乐句
   起伏、按拍号的重音。
2. KTH 规则 (`perform/rules.py`, `perform/kth.py`): 乐句弧线、旋律和和声张力、
   时值对比、句读、结尾渐慢。
3. `duck_under_sustain`: 旋律长音下面伴奏变弱。要用最终力度, 所以放在力度之后。
   演奏模式里它不删音。
4. `harmonic_pedal`: 在和声变化处换踏板。
5. `motor_noise`、`intentional`、`match_distribution`: 音与音之间的力度起伏,
   然后整体往实测的真人力度分布上靠一半。
6. `humanize`: 最后做时间, 因为前面每一步都假设音在拍点上。按键时间差
   (`perform/timing.py::key_travel_ms`) 在这里加。

编配模式在第 1 步之后存一份快照 (`perf.meta["content"]`), "这个音在不在和弦里"、
"旋律全不全"这类问题问的是快照, 不是加了时间偏移之后的结果。

## 渲染

`render.sfz.SFZSampler` 读 SFZ, 按音高和力度层选采样, 并把采样库自己的
力度-音量曲线校正成固定的规律, 让力度真正控制音量 (不改音色的选择)。音一直
响到松键, 踏板踩着的话响到踏板抬起。`render.resonance` 加上采样器没有的
共鸣。`render.master` 跑 ffmpeg (EQ、压缩、可选房间、限幅), 输出没声音时抛
`SilentOutput`, 不会默默写出一个静音文件。

## 扩展

- 新织体: 所有音高从 `theory.chord_near` 或 `theory.stack` 取, 音级查找窗口
  至少一个八度, `voice` 如实设置 (可以删的用 `COLOUR`), 设 `tag`。用带转位和
  七和弦的和声测, 原位三和弦会掩盖按音程放音的 bug。
- 新风格: `arrange/styles.py` 里加一个预设, 数字最好按真人录音定。
- 新 audit 检查: 返回 `Report`, 不改演奏, 往 `stats` 里写数, `error` 只用在
  错音和没声音上。先在故意做错的例子和正常例子上都跑一遍再相信它。
