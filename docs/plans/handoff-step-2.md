# 交接：第 2 步已完成（2026-11-07）

> 给接手的 agent 看的。第 2 步（和弦 + 调性）验收通过，等负责人合并 PR 后进入第 3 步（编配引擎）。开始前读 `CLAUDE.md`、`docs/ROADMAP.md` 第 3 步、`docs/plans/backend.md`，以及本文第 1 节的工作规则。

## 1. 负责人的工作规则（必须遵守）

- 先写计划再写代码；每个决定**一次只讨论一个**：问题、选项、利弊、推荐与理由；负责人拍板后记入当前步计划的"负责人决定"一节，再谈下一个。
- 范围、验收口径、对外可见内容（仓库里放什么）由负责人定；工具内部实现（如样本生成器的切图、读图协议、自检）负责人已授权实现方自行决定，只汇报结果（`step-2.md` 决定 15）。
- 繁琐的人工活（逐页看图、核对文件）由 agent 自己做，规模大就分批派子代理，不要推给负责人（决定 14）。
- 实现完成后**先问负责人是否开 PR**，不要自己开；PR 描述逐条对照路线图当前步骤的验收标准。
- 验收数字必须由仓库里的脚本产出，脚本与结果文件一起提交；数据集的音频 / 标注、负责人的歌谱、渲染出的音频、曲库歌名一律不进仓库（歌名在报表里用编号，决定 18）。
- 负责人不是乐理专家，偏好机器学习的语言；汇报要主动、及时，不要等他来问；解释软件工程术语（PR、CI 等）。
- 子代理一律用 Opus 5.5。
- 负责人不持有、也不会提供任何歌曲录音（版权风险）；评测只用开放数据集和谱子渲染的合成音频。
- 中文写文档与沟通；代码、标识符、提交信息用英文。保持 `ruff format --check . && ruff check . && mypy && pytest -q` 全绿后再推送。

## 2. 第 2 步的结果

| 验收项（ROADMAP 第 2 步） | 结果 |
|---|---|
| 负责人曲库样本集上和弦 majmin ≥ 80% | **0.974**（220 首）；sevenths 0.951、转位 0.927 |
| 同一样本集上调性 ≥ 85%（关系大小调算对，决定 16） | **0.936**（精确匹配 0.686） |
| 每个和弦都落在拍上 | 融合层按拍决策，`tests/test_chord_stage.py` 断言 |
| 置信度有定义、阈值可配置 | `fusion.py` 与 `FusionParams`，计划 2.4 节 |
| 4 分钟歌曲和弦 + 调性 ≤ 30 秒 | 6.1 秒（RTX 3090，热启动） |

公开数据集参考（不设门槛）：GuitarSet 伴奏 majmin 0.775、调性 0.789；AAM 0.947、单一调性 0.931。全部数字见 `docs/benchmarks/step-2-chords.md`，已知缺口见 `step-2.md` 第 9 节。

## 3. 第 2 步留下的可复用部件

- 分析：`opencapo.analysis.chords`（BTC、lv-chordia、色度模板、低音、调性规则，拍网格上融合），`opencapo analyze` 输出 `Chord` 序列与 `Key`。
- 样本生成器 `opencapo.sheet`：谱子文本格式（第 3、4 步输出简谱时复用其解析器）、渲染器、识谱（`scripts/sheet_ocr.py`，按行读图，决定 15）、整理（`scripts/sheet_triage.py`）、样本集清单（`manifest`，决定 12）。
- 评测：`scripts/eval_chords.py`（`--sheets` 按清单读曲库样本集，`--anonymize-sheets` 隐去歌名）、`scripts/analyze_chords.py`（错误分析）。

## 4. 怎么跑

```bash
# 整理曲库图片（去重、选版本、定页序；--review 读入 agent 看图核定的页序与来源；--apply 写出每首的页图目录）
python scripts/sheet_triage.py /data/raw --out /data/triage --review /data/triage/review.json --apply
# 识谱 + 渲染（读图用 Claude Code 无头模式，走当前登录；源可以是 PDF、单图或页图目录；--skip-done 续跑）
python scripts/sheet_ocr.py read /data/triage/SONG [...] --out /data/sheets --render --skip-done
# 重建样本集清单（评测只认清单）
python scripts/sheet_ocr.py manifest /data/sheets
# 评测（--cache 复用各专家的逐拍输出；曲库歌名换成编号，对照表写到仓库外）
python scripts/eval_chords.py --guitarset /data/guitarset --aam /data/aam --sheets /data/sheets \
    --device cuda --cache /data/eval-cache --anonymize-sheets /data/songbook-key.json \
    --out docs/benchmarks/step-2-chords
```

安装：`pip install -e ".[dev,analysis,sheets]"`，另需 ffmpeg。GPU 机器上 torch 必须与驱动匹配，lv-chordia 声明 `torch>=2.13`，旧驱动机器用约束文件 + `--no-deps` 装，见 `docs/benchmarks/README.md`。
