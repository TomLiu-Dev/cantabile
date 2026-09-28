# Changelog

## 0.1.0 - 2026-09-28

第一次公开发布。

- `python -m cantabile play`: 把 MIDI 用 SFZ 钢琴采样弹出来, 保留所有音, 加上
  旋律突出、力度、踏板、速度弧线、按键时间差、制音器和共鸣。
- 编配模式 `python -m cantabile 曲名`: 旋律 + 和弦生成钢琴伴奏, 曲子放在
  `~/.cantabile/corpus/` 或 `CANTABILE_CORPUS`。
- 检查 (audit): 和弦音、冲突、旋律完整、旋律被盖住等。
- 评估脚本: 真人/渲染分类器 (`eval/discriminator.py`) 和旋律识别测试
  (`eval/heldout_melody.py`)。
- 两个例子: 莫扎特土耳其进行曲、维瓦尔第"春"第一乐章 (Mutopia Project)。
