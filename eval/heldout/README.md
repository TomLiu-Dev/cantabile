# Held-out 旋律测试

`eval/heldout_melody.py` 用来测"从多声部 MIDI 里找出旋律"找得准不准。这里的
文件不随仓库发布 (版权不是我的), 要自己放:

- `名字.mid`: 四声部 MIDI, 程序只看这个;
- `名字.pdf`: 同一首的印刷谱, 用另一条代码路径 (`cantabile.ingest.sheet`)
  从谱面上读出标准答案。

任何有配套 MIDI 和 PDF 谱的四声部曲子都可以。读 PDF 需要 Pillow 和 poppler 的
`pdftoppm`。

```bash
pip install -e ".[sheet]"
python eval/heldout_melody.py
```
