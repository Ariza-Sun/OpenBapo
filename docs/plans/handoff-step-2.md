# 交接：第 2 步进行中（2026-11-04）

> 给接手的 agent 看的。读完本文、`CLAUDE.md`、`docs/plans/step-2.md`，再动手。分支 `claude/gracious-cannon-i2saip`，已全部推送；工作树干净；未开 PR。

## 1. 负责人的工作规则（必须遵守）

- 先写计划再写代码；每个决定**一次只讨论一个**：问题、选项、利弊、推荐与理由；负责人拍板后记入 `step-2.md` 第 8 节，再谈下一个。
- 实现完成后**先问负责人是否开 PR**，不要自己开；PR 描述逐条对照路线图当前步骤的验收标准。
- 验收数字必须由仓库里的脚本产出，脚本与结果文件一起提交；数据集的音频 / 标注、负责人的歌谱、渲染出的音频一律不进仓库。
- 负责人不是乐理专家，偏好机器学习的语言；汇报要主动、及时，不要等他来问；解释软件工程术语（PR、CI 等）。
- 子代理一律用 Opus 5.5（`model="claude-opus-5-5"`；Agent 工具用 `model="opus"`、`effort="max"`）。
- 负责人不持有、也不会提供任何歌曲录音（版权风险）；评测只用开放数据集和谱子渲染的合成音频。
- 中文写文档与沟通；代码、标识符、提交信息用英文。保持 `ruff format --check . && ruff check . && mypy && pytest -q` 全绿后再推送（本分支现在 322 个测试通过）。

## 2. 第 2 步的现状

验收口径已改为负责人曲库（决定 11）：和弦 majmin ≥ 80%、调性 ≥ 85% 在**负责人提供的谱子渲染成的样本集**上算；GuitarSet 与 AAM 只作公开数据集参考。

| 项 | 状态 |
|---|---|
| 和弦 + 调性流水线（BTC + lv-chordia + 色度模板 + 低音 + 调性规则，拍网格上融合） | 完成，`FusionParams` 默认值即调参结果 |
| 公开数据集参考数字（`docs/benchmarks/step-2-chords.md`） | GuitarSet 伴奏 majmin 0.775、调性 0.789；AAM 0.947、单一调性 0.931；耗时 6.1 s |
| 融合层 / 规则层优化 | 四轮，合计 ±0.002，已停（`step-2.md` 8.9）；剩余差距是两模型对木吉他 Funk / 爵士伴奏的能力 |
| 样本生成器：谱子文本格式、渲染器（按谱上指法扫弦 + 钢琴旋律）、`--sheets` 评测接入 | 完成 |
| 样本生成器：识谱（PDF / 图片 → 谱子文本）一条命令 | 完成；2026-11-04 改为按行读图、模型分小节（决定 15），因按小节切图在曲库的多数版式上失败。《曾经的你》的 59/60 基准是旧协议的数字，新协议待重测 |
| 样本集 manifest；评测按 manifest 读样本集（`step-2.md` 10.4 第 4 项） | 完成（决定 12）：`sheet_ocr.py manifest DIR`，评测只认清单，报表逐首列出 |
| 负责人批量提供谱子 → 生成样本集 → 报表"曲库合成"一组出数字（10.4 第 5 项） | 进行中：负责人已提供 238 首的图片（本地 `sheets-input/raw_data/`），按决定 13 先整理、按版式试点，再批量 |
| 扫弦节奏型 | 固定型（下 下上 上下上），未读谱上的六线谱；负责人知情，暂不做 |

## 3. 接下来按顺序做什么

1. ~~manifest 与评测接入~~（已做，决定 12）。
2. **整理负责人的曲库图片并按版式试点**（决定 13、14）：整理与页序核定已完成（`sheets-input/triage/`，240 首、237 首有页图、页序全部核定，表在 `triage.md`，核定文件 `review.json`，试点清单 `pilot.json`）。试点 10 首第一轮暴露小节切分问题 → 决定 15 改为按行读图；第二轮试点结果见 `sheets-pilot/run.log`。接下来：看试点各首的 `review.md`，把待复核多的版式再修，然后批量跑其余曲目。
3. 在 GPU 机器上跑 `eval_chords.py --guitarset ... --aam ... --sheets <样本集> --cache ...`，报表里"曲库合成"一行就是验收依据；更新 `step-2.md` 第 12 节、README。
4. 若曲库样本集过线：按规则问负责人是否开 PR。若不过线：按错误分析（`scripts/analyze_chords.py --sheets ... --subset sheets`）定位，再一次一个决定地讨论。

## 4. 怎么跑

```bash
# 整理曲库图片（去重、选版本、定页序；--review 读入 agent 看图核定的页序与来源；--apply 写出每首的页图目录）
python scripts/sheet_triage.py /data/raw --out /data/triage --review /data/triage/review.json --apply
# 识谱 + 渲染（一条命令；读图用 Claude Code 无头模式，走当前登录，不用 API key；源可以是 PDF、单图或页图目录）
python scripts/sheet_ocr.py read /data/triage/SONG [SONG2.pdf ...] --out /data/sheets --bpm 76 --render
# 只渲染手工录入的谱子（结束时重建 manifest.json）
python scripts/sheet_synth.py docs/examples/xiaoxingxing.sheet --out /tmp/x
# 重建样本集清单（评测只认清单）
python scripts/sheet_ocr.py manifest /data/sheets
# 评测（含曲库样本集）；--cache 复用各专家的逐拍输出
python scripts/eval_chords.py --guitarset /data/guitarset --aam /data/aam --sheets /data/sheets \
    --device cuda --cache /data/eval-cache --out docs/benchmarks/step-2-chords
# 错误分析
python scripts/analyze_chords.py --sheets /data/sheets --cache /data/eval-cache --subset sheets
```

安装：`pip install -e ".[dev,analysis,sheets]"`。GPU 机器上 torch 必须与驱动匹配，lv-chordia 声明 `torch>=2.13` 会顶掉旧 torch，用约束文件 + `--no-deps` 装，见 `docs/benchmarks/README.md`。

## 5. 待负责人决定 / 已知事项

- 谱子识读出的两处疑点（`曾经的你`）：第 2 页乐句开头 `7,` 跳到 `1'` 的九度大跳（谱面如此，疑为刻错）；同乐句的附点 `2·` 在第 1 页与第 3 页印法不一致。按谱面原样保留。
- GuitarSet 伴奏 0.775 / 0.789 未达旧门槛，已随决定 11 降为参考；若将来要补，方向是训练数据含木吉他伴奏的第三模型，公开候选目前没有合格的（`step-2.md` 8.9）。
- 合成样本的分数是真录音的上限，报表与计划都已注明。
- 旧的 node28 子会话（用 Remote Control 桥派发的评测任务）已完成或归档；评测缓存 `eval-cache/` 含 860 首的逐拍后验，可直接复用。
