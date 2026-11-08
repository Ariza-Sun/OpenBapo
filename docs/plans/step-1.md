# 第 1 步实现计划：分轨 + 节拍 + BPM

> 对应：`docs/ROADMAP.md` 第 1 步；需求 `docs/REQUIREMENTS.md` 5.1（输入与时长上限）、5.2（`Beat`、`Tempo`、`Stem`）、7（处理时间、可观测性、可测试性、许可证）。
> 状态：已完成。验收数字见第 11 节与 `docs/benchmarks/`。

## 1. 目标与范围

本步把分析流水线的前两个环节接进来，让 `openbapo analyze` 第一次真正读音频。做完后应当有：

1. **音频解码与时长检查**：六种格式都能解码；超过 8 分钟给出清楚的错误；`Analysis.duration` 填上。
2. **分轨**：混音 → vocals / drums / bass / other，只在流水线内部以临时文件 / 内存数组存在，流程结束（含出错）后不残留。
3. **节拍 / 强拍 / BPM**：输出 `Beat`（小节 + 拍）与全局 `Tempo`。
4. **评测脚本**：在公开数据集上算节拍 F-measure（mir_eval），结果文件提交进仓库。
5. **计时**：每个环节的耗时记录下来，并有脚本在 GPU 上产出"4 分钟歌曲分轨 + 节拍"的耗时数字。
6. **许可证**：用到的每份模型权重的许可证写进 `docs/plans/backend.md`。
7. **依赖分组**：torch 与模型依赖进可选依赖组 `openbapo[analysis]`，核心包仍只依赖 pydantic。

**本步不做**：和弦、调性（第 2 步）；节奏型推荐（第 3 步）；段落结构检测（需求明确不做，候选模型若顺带输出一律丢弃）；局部速度曲线（需求只要全局 BPM）；分轨的任何存储 / 下发 / 导出；Discord；队列。

## 2. 流水线设计

### 2.1 模块

```
src/openbapo/
    errors.py                # InputError（CLI 与流水线共用）
    analysis/
        __init__.py          # 只有文档字符串，不 import 任何重依赖
        deps.py              # 分析依赖是否装齐的检查（给 CLI 一句清楚的安装提示）
        device.py            # "auto" → cuda / cpu
        audio.py             # ffprobe + ffmpeg 解码 → float32 数组；8 分钟上限
        separate.py          # Separator 协议、Stems、临时目录、Demucs 适配器
        beats.py             # BeatTracker 协议、Beat This! 适配器
        grid.py              # 拍点 + 强拍 → Beat 元组、Tempo、TimeSignature（纯计算，无模型）
        pipeline.py          # analyze_file(...) -> AnalysisResult(analysis, timings)；StageTimings
scripts/
    make_tone.py             # 正弦波 / 节拍器点击音（--click BPM），测试与 CI 用
    eval_beats.py            # 节拍评测：候选 × GTZAN → F-measure 表（含自检）
    time_pipeline.py         # 一首歌的各环节耗时（冷 / 热）
    requirements-eval.txt    # 只有评测用到的依赖（mir_eval）
tests/fakes.py               # FakeSeparator / FakeTracker，让流水线在 CI 上无模型跑通
docs/benchmarks/
    README.md                # 数据集获取与运行说明
    step-1-beats.md / .json  # eval_beats.py 的输出（提交进仓库）
    step-1-timing.md / .json # time_pipeline.py 的输出（提交进仓库）
```

- `openbapo.analysis` 这个包名是第 0 步预留的；本步起它就是分析流水线。
- `Separator` / `BeatTracker` 做成协议（Python 的 `typing.Protocol`，即只规定方法签名的接口），流水线只依赖接口。好处：单元测试用假实现在 CPU 上跑通整条流水线；换模型只换适配器。
- `grid.py` 是纯计算，不 import torch，可在 CI 上完整测试。
- 模型包在适配器方法里延迟 import；`openbapo.analysis` 包本身与 `deps.py` 不碰 numpy / torch，所以核心安装下运行 `openbapo analyze` 得到的是一句安装提示，不是 ImportError（有测试守着）。
- 分轨先在内存里（`Stems`，四个 `(channels, frames)` 数组），再写成 32 位浮点 WAV 到临时目录供后续环节读；两者都随 `with` 块结束消失。

### 2.2 数据流

```
音频文件 ──ffmpeg──▶ 混音数组 (44.1 kHz, 立体声)
                        │
                        ├──▶ 时长检查（> 8 分钟 → InputError）
                        │
                        ├──▶ Separator ──▶ Stems（临时目录内）──▶ [本步无消费者；第 2/5/7 步用]
                        │
                        └──▶ BeatTracker ──▶ 拍点时间 + 强拍时间
                                                  │
                                                  ▼
                                          grid.py ──▶ Beat 元组、Tempo、每小节拍数
                                                  │
                                                  ▼
                                             Analysis（duration、beats、tempo 填上）
```

- 节拍模型默认吃**混音**：候选模型都是在混音上训练的，喂分轨是分布外输入。评测脚本加一个开关，顺带测"drums + bass 重混后再跟踪"是否更好，结果一并报告；只有实测更好才切换。
- 整条流水线包在一个 `tempfile.TemporaryDirectory()` 里，分轨只写在这个目录；`with` 块退出（正常或异常）即删除。测试用假 `Separator` 证明：正常结束后目录不存在；中途抛异常后目录也不存在。

### 2.3 拍点 → `Beat` 的换算（`grid.py`）

节拍模型给两列时间：所有拍点、其中的强拍。换算规则：

| 项 | 规则 |
|---|---|
| 每小节拍数 `n` | 相邻强拍之间拍数的众数（只统计完整小节） |
| `bar_index` | 从最早的小节起从 0 数。第一个强拍之前若有拍点（弱起 / 前奏不完整小节），从第一个强拍往前按周期 `n` 补隐式小节线，这些拍归到第 0、1… 小节，第一个强拍所在的小节紧随其后 |
| `beat_in_bar` | 每个强拍为 1，之后逐拍 +1。模型漏掉一个强拍时该小节拍数会超过 `n`，照实计数（模型允许任意 ≥ 1），不强行修正 |
| 小节长度测不出 | 强拍少于 2 个，或每拍都被判成强拍（合成节拍器音频上出现过）：忽略强拍，按 4 拍一小节排，`time_signature` 留 `None` |
| `Tempo.bpm` | `60 / median(相邻拍点间隔)`：中位数对个别漏拍 / 多拍不敏感；保留小数，不四舍五入 |

这些规则纯函数实现，单元测试覆盖：整齐的 4/4、弱起 1 拍、弱起超过一小节、漏强拍、只有一个拍点、零拍点（纯噪声 / 静音 → `beats` 空、`tempo` 为 `None`，不报错）。

### 2.4 耗时记录

`StageTimings`：`decode`、`separate`、`beats`、`total` 四项秒数。CLI 加 `--timing <json>` 把它写到文件（`-` 为 stderr），`time_pipeline.py` 直接调用同一个函数。不做日志框架，第 6 步接队列时再接可观测性。

## 3. 依赖与环境

### 3.1 可选依赖组

```toml
[project.optional-dependencies]
analysis = ["numpy", "torch>=2.1", "demucs>=4.1", "beat-this>=1.1", ...]   # 以确认的模型为准
```

- 核心包（`pip install openbapo`）仍只依赖 pydantic；编配 / 渲染（第 3、4 步）不装 `analysis` 也能装和测。
- 没装 `analysis` 时运行 `openbapo analyze`，报一句清楚的错："analysis stages need `pip install 'openbapo[analysis]'`"，退出码 1。
- `ffmpeg` 命令行程序是运行时外部依赖（解码 mp3 / m4a / ogg / opus 都靠它），README 与自部署文档写明；GitHub Actions 的 ubuntu runner 自带。
- torch 的 CUDA 版本由安装方按 pytorch.org 的说明选；`pyproject.toml` 不锁 CUDA 版本。

### 3.2 Python 版本

候选模型的 PyPI 元数据：demucs 4.1.0 要求 ≥ 3.10；beat-this 1.1.0 纯 Python 无版本限制；torch 支持 3.11–3.13。暂时**不动** `requires-python = ">=3.11"` 与 CI 的 3.11 / 3.12 / 3.13 矩阵。CI 新增的分析冒烟任务（见第 6 节）如果在某个版本装不上，再按 CLAUDE.md 的流程和负责人确认收紧。madmom、NATTEN 只出现在评测对比里（下一节），不影响包的版本范围。

### 3.3 评测对比专用依赖

`scripts/requirements-eval.txt`：只在跑 `eval_beats.py` 对比候选时装，含 mir_eval 与**落选 / 不能打包的候选**（如 madmom、all-in-one）。这样 `pyproject.toml` 不引用任何许可证有问题的权重。

## 4. 模型候选与许可证（核实于 2026-10-28）

### 4.1 分轨

| 候选 | 代码许可证 | 权重许可证 | 备注 |
|---|---|---|---|
| Demucs `htdemucs`（v4 Hybrid Transformer） | MIT | **作者明确写"不清楚"**：训练数据 MUSDB HQ 含仅限研究用途的曲目 + Meta 内部 800 首 | 原仓库 facebookresearch/demucs 已于 2025-01 归档；作者 fork adefossez/demucs 在维护，PyPI `demucs` 4.1.0（2026-07），Python ≥ 3.10，torch ≥ 2.1。4 轨。CPU 约 1.5 × 曲长，GPU 需约 7 GB 显存 |
| Demucs `htdemucs_ft` | 同上 | 同上 | 同一训练集的微调版，SDR 高约 0.5 dB，推理慢约 4 倍 |
| Demucs `htdemucs_6s` | 同上 | 只用 Meta 内部数据训练，作者写"大概率适用 CC-BY-NC" | 6 轨（多吉他、钢琴），本项目不需要 |
| BS-RoFormer / Mel-Band RoFormer（ZFTurbo 等社区权重） | MIT（训练代码） | **各权重不一，部分标 CC-BY-NC-SA 4.0** | 人声 SDR 最高，但公开权重几乎都是 vocals / instrumental 两轨；4 轨权重少且许可证不明；显存与耗时都高于 Demucs |

### 4.2 节拍 / 强拍

| 候选 | 代码许可证 | 权重许可证 | 训练 / 测试数据 | 备注 |
|---|---|---|---|---|
| Beat This!（CPJKU） | MIT | **MIT** | 16 个数据集训练，**GTZAN 整体留出作测试集** | PyPI `beat-this` 1.1.0；纯 Python，依赖 torch ≥ 2.0、einops、soxr、rotary-embedding-torch；GPU；默认不需要 madmom（DBN 后处理可选）。论文报告 GTZAN 节拍 F1 ≈ 0.89、强拍 ≈ 0.78（二手来源，arxiv 在本会话网络白名单外，待在服务器上用评测脚本复现） |
| madmom（RNN + DBN） | BSD | **CC BY-NC-SA 4.0**（README 明示） | Ballroom、Beatles、Hainsworth、RWC 等；GTZAN 不在训练集内 | PyPI 版本停在 2018，Python ≥ 3.10 需从 git 安装；CPU 推理，一首 4 分钟歌约 10–20 秒。非商业条款对自部署者是陷阱，**不建议打包进产品**，只作评测基线 |
| all-in-one（mir-aidj） | MIT | README 未说明 | Harmonix Set 8 折；GTZAN 不在训练集内 | 需要 NATTEN（按 torch / CUDA 版本装预编译包，常见安装失败）+ madmom + 自带一遍 Demucs 分轨；同时输出段落结构（本项目不用）。维护状态不明 |

> 许可证结论会同步进 `docs/plans/backend.md` 第 2 节的表；CC-BY-NC 类显式标出。

## 5. 评测与计时脚本

### 5.1 `scripts/eval_beats.py`

```
python scripts/eval_beats.py --audio <dir> --annotations <dir> \
    --trackers beat_this,madmom,allin1 --input mix[,drums_bass] \
    --device cuda --out docs/benchmarks/step-1-beats
```

- 数据集见第 8 节决定 4；音频与标注**不提交**进仓库，脚本按目录参数读取。
- 指标：mir_eval `beat.f_measure`（默认 70 ms 容差），分别算拍点与强拍；按曲目平均；另按流派分组报告。每个候选记录版本号、每曲平均耗时。
- 输出 Markdown 表 + JSON（含 GPU 型号、日期、各依赖版本、曲目数），直接提交到 `docs/benchmarks/`。README 的"状态"一节引用这个数字。
- 候选模型的适配器放在脚本里（或 `scripts/_trackers.py`），不进 `src/`；选定的那个再搬进 `src/openbapo/analysis/beats.py`。

### 5.2 `scripts/time_pipeline.py`

```
python scripts/time_pipeline.py <song.mp3> --device cuda --repeat 3 --out docs/benchmarks/step-1-timing
```

- 跑完整的 `analyze()`，记录 decode / separate / beats / total，取 3 次的中位数（第一次含权重加载，单独列出）。
- 歌曲由负责人任选一首约 4 分钟的 mp3（神经网络的耗时与内容无关，只与时长有关），文件不提交；输出里记录时长、采样率、GPU 型号。

### 5.3 在 GPU 服务器上的操作（由负责人本地的 Claude Code agent 执行）

```bash
git fetch origin claude/gracious-cannon-i2saip && git checkout claude/gracious-cannon-i2saip
python -m venv .venv && source .venv/bin/activate
pip install -e ".[analysis]" -r scripts/requirements-eval.txt
# 评测集下载见 docs/benchmarks/README.md
python scripts/eval_beats.py ...
python scripts/time_pipeline.py ...
git add docs/benchmarks && git commit -m "Add step-1 beat evaluation and timing results" && git push
```

结果通过分支交回，本会话再据此写文档和定模型。

## 6. 测试与 CI

| 测试 | 跑在哪 | 覆盖 |
|---|---|---|
| `grid.py` 纯函数 | CI 默认 | 2.3 节列的全部情形；输出满足 `Analysis` 的全部约束（时间递增、`bar_index` 不减、`is_downbeat ⇔ beat_in_bar == 1`） |
| 解码与时长检查 | CI 默认（runner 自带 ffmpeg） | wav / mp3 / flac 等解码得到正确时长；8 分钟 + 1 秒的合成音频报 `InputError`；损坏文件报可读错误 |
| 分轨不残留 | CI 默认（假 `Separator`） | 正常结束、中途异常两种情况下临时目录都不存在 |
| 流水线装配 | CI 默认（假 `Separator` + 假 `BeatTracker`） | `analyze()` 输出含 `beats`、`tempo`、`duration`；计时四项都有值 |
| 未装 `analysis` 时的提示 | CI 默认 | 错误信息与退出码 |
| 真模型冒烟 | 标记 `models`，CI 单独一个任务（CPU，单一 Python 版本，权重用 actions/cache 缓存） | `scripts/make_tone.py --click 120` 生成 10 秒 120 BPM 的节拍器音频；真 Demucs + 真节拍模型跑完，`bpm` 在 120 ± 2，分轨不残留 |
| GPU 实测 | 不进 CI；脚本产出数字 | 5.1、5.2 节 |

- `pytest` 默认 `-m "not models"`，现有 CI 四条检查不变。
- 新 CI 任务 `analysis-smoke`：`pip install -e ".[analysis]" --extra-index-url https://download.pytorch.org/whl/cpu`（CPU 版 torch 小得多）→ `pytest -m models` → `openbapo analyze click.wav --out - --timing -`。这个任务同时是"依赖在新版 Python 上装得上"的探针。

## 7. 数据模型改动

- `Analysis.duration` 由本步填写，字段已存在，不改。
- 新增 `TimeSignature`（`numerator ≥ 1`，`denominator ∈ {2, 4, 8, 16}`），`Analysis.time_signature: TimeSignature | None`，默认 `None`；`schema_version` 升到 2（第 8 节决定 5）。顺序：改 `REQUIREMENTS.md` 5.2 → 改模型、金样 `tests/data/analysis_full.json` 与属性测试 → 实现填写逻辑。

## 8. 负责人决定

### 1. 已定：GPU 用负责人自己的服务器（2 × RTX 3090，各 24 GB）

- 本会话运行在 Anthropic 的云容器里，出网只有一条 HTTPS 代理且限白名单域名，没有 SSH 客户端，裸 TCP 出不去，因此不能直接连任何 GPU 机器；这是容器隔离设计，改网络白名单也打不开 SSH。
- 工作流：本会话写脚本、推到分支 → 负责人本地的 Claude Code agent 拉分支、SSH 到服务器执行第 5.3 节的命令 → 结果文件提交回同一分支 → 本会话继续。验收数字始终由仓库里的脚本产出。
- 未采用：按小时租云 GPU（同样要手动操作，多花钱）；提前租第 8 步的生产服务器（同上，且闲置数月）；自托管 GitHub Actions runner（可让本会话直接触发，但要机器常开并多一次配置，先不做，后续步骤评测多了再考虑）。
- 待负责人补充：服务器的 CUDA / 驱动版本、系统 Python 版本，用来定 torch 的安装命令。

### 2. 已定：分轨用 Demucs `htdemucs`

- 权重公开、首次运行自动下载，可用；但作者未为权重单独声明许可证，训练数据含仅限研究用途的 MUSDB 曲目与 Meta 内部数据，作者表示无权澄清。**按非商业（CC-BY-NC 同级）对待**，在 `backend.md` 与自部署文档中显式标出。
- 4 分钟歌曲在 3090 上预计 10–20 秒，给后续环节留足 90 秒预算；分轨在本步没有消费者，质量差异要到第 5 / 7 步在人声轨上才看得出，届时只换 `Separator` 适配器。
- 计时脚本顺带记录 `htdemucs_ft` 的耗时作参考，不接入。
- **实测**（RTX 3090，4:00 歌曲，`docs/benchmarks/step-1-timing.md`）：`htdemucs` 分轨 6.1 秒、整条流水线热启动 7.1 秒（冷启动 13.7 秒）；`htdemucs_ft` 分轨 20.3 秒。
- 未采用：`htdemucs_ft`（约 4 倍耗时，SDR 只高约 0.5 dB，许可证状况相同）；BS-RoFormer 社区权重（人声质量最高，但 4 轨权重稀少、许可证各异且不少为 CC-BY-NC-SA，要自己拼多个模型，依赖与显存更重）；自己在 MUSDB18-HQ 上训练（远超本步范围）。

### 3. 已定：节拍模型只实测 Beat This!，过门槛即用；不过再测备选

- 实测 = 在评测集上跑 Beat This!（默认 `final0` 权重，无 DBN 后处理），拍点 F1 ≥ 85% 即选定。"三选一由实测决定"按门槛式理解：先测最可能过的，过了就不再测其它。
- 备选顺序写死：madmom（权重 CC BY-NC-SA 4.0，只能作评测基线，不能打包进产品）→ all-in-one（权重许可证未声明，依赖 NATTEN 与自带 Demucs）。
- 评测脚本按多候选设计（`--trackers` + 适配器注册表），本步只实现 Beat This! 适配器；要换候选只加一个适配器文件。
- 没有第二个模型交叉验证评测代码，用两项自检代替：（1）把标注当预测算一遍，F1 必须为 1.0；（2）结果与论文报告的 GTZAN 拍点 F1 ≈ 0.89 对照，偏差超过 2 个百分点先查代码。
- 评测口径：mir_eval `beat.f_measure`（70 ms 容差），拍点与强拍分别算、按曲目平均、按流派分组；验收门槛只对拍点 F1，强拍 F1 报数字不设门槛。保留"drums + bass 重混后跟踪"对照组。
- 未采用：三个候选全跑（madmom 需从 git 编译安装、all-in-one 的 NATTEN 常装不上，而按选定规则 Beat This! 几乎是预定答案，不值得花服务器时间）。
- **实测结果（选定）**：GTZAN 998 首（jazz.00054 损坏跳过），标注自检 1.000；Beat This! 在混音上拍点 F1 **0.891**、强拍 F1 0.786，与论文报告的 ≈ 0.89 / 0.78 吻合，过门槛，不再测备选。"drums + bass 重混"对照组为 0.770 / 0.684，明显更差，节拍输入保持混音。按流派：古典最弱（0.660），布鲁斯 0.831，其余 0.87–0.98；弹唱场景常见的 pop / country / rock 都在 0.93 以上。详见 `docs/benchmarks/step-1-beats.md`。

### 4. 已定：评测集用 GTZAN（GTZAN-Rhythm 标注）

- 1000 首 × 30 秒、10 个流派；jazz.00054 已知损坏，实际约 999 首，脚本报出缺失 / 损坏文件并记录实际曲目数。
- 需求第 7、9 节点名 GTZAN；且 GTZAN 是 Beat This! 的留出测试集，结果干净。
- 标注：CPJKU/beat_this_annotations（MIT）的 `gtzan/annotations/beats/`，源自 Marchand & Peeters 2015 的 GTZAN-Rhythm（经 Böck 整理），含拍点与强拍。
- 音频：原始 UVic 服务器已下线，从 Kaggle 或 Hugging Face `marsyas/gtzan` 镜像下载，目录结构 `genres_original/<genre>/<genre>.000NN.wav`。
- 音频与标注都不进仓库；下载与目录约定写在 `docs/benchmarks/README.md`，由负责人本地的 agent 在服务器上准备。
- 未采用：Ballroom（在 Beat This! 的训练集里，数字偏高，不能作验收依据）。

### 5. 已定：加 `TimeSignature(numerator, denominator)` 字段，本步填写，`schema_version` 升到 2

- 节拍模型只能给出每小节拍数 `n`；填写规则：`numerator = n`，`denominator = 4`，仅当 `n == 6` 时记为 6/8。
- 约定：**一个 `Beat` 对象 = 一个 denominator 音符时值**，tick（`TICKS_PER_BEAT`）与 `Tempo.bpm`（每分钟多少个 `Beat`）都以此为单位，MusicXML 的时值换算由此确定。
- 字段可选：没有拍点时为 `None`。
- 已知缺口：按附点四分音符标注习惯跟踪出的 6/8 歌会得到 2/4（内部一致，但节奏型推荐会按 2/4 给）；3/4 与 6/8 的自动区分没有可靠方法，B、C 两案同样躲不开。
- 顺序：先改 `REQUIREMENTS.md` 5.2 的对象表 → `schema_version` 升 2、改金样与测试 → 实现。
- 未采用：只加 `beats_per_bar`（分母迟早要定，只是把同一规则挪到第 4 步）；不加字段、第 3 步现算（三个消费者各算一遍，无单一事实来源）。

## 9. 已知缺口（不在本步补）

| 缺口 / 风险 | 说明 | 处理 |
|---|---|---|
| Demucs 权重许可证不清楚 | 作者无法澄清；MUSDB 含仅限研究的曲目 | 在 backend.md 与自部署文档显式标出；分轨做在接口后面，发现许可证更干净且质量够的 4 轨模型可换 |
| 速度倍数歧义 | 节拍模型可能把 80 BPM 的歌跟成 160（或反之） | 本步照模型输出；节奏型推荐（第 3 步）对倍速不敏感时无害，否则第 3 步加规则 |
| 分轨在本步没有消费者 | 第 2 步（bass + other 做和弦）、第 5 / 7 步（vocals）才用 | `Stems` 接口可能随消费者需要微调（采样率、单声道） |
| 歌曲中途变速 / 变拍号 | 需求只要全局 BPM 与一个拍号 | 照实输出拍点，BPM 取中位数；页眉显示全局值 |
| 8 分钟歌曲的内存 | 4 轨 × 8 分钟 × 44.1 kHz × 立体声 × float32 ≈ 680 MB | 服务器内存足够；如需降低改为单声道 / 逐轨落盘 |

## 10. 交付顺序

一个 PR，按提交推进，每个提交本地检查全过：

1. 本计划 + 决定记录（每定一条提交一次）。
2. `audio.py`：ffmpeg 解码、时长上限、`duration` 填写；CLI 错误信息；测试。
3. `grid.py`：拍点 → `Beat` / `Tempo` / 每小节拍数；测试。
4. `separate.py`：`Separator` 协议、临时目录管理、Demucs 适配器；残留测试（假实现）+ `models` 标记的真模型测试。
5. `scripts/eval_beats.py` + `requirements-eval.txt` + `docs/benchmarks/README.md`（数据集获取说明）。
6. **负责人在 GPU 上跑评测**，结果提交；据此定节拍模型，适配器进 `beats.py`。
7. `analysis/__init__.py` 装配 + 计时 + CLI `--device` / `--timing`；`make_tone.py --click`；CI `analysis-smoke` 任务。
8. `scripts/time_pipeline.py`；**负责人在 GPU 上跑计时**，结果提交。
9. `backend.md` 许可证表、README 状态与数字、本计划第 11 节对照填实。

## 11. 验收对照

| 验收项（ROADMAP 第 1 步） | 如何证明 | 状态 |
|---|---|---|
| 节拍 F-measure ≥ 85%（评测脚本输出，数字进 README 或 docs） | `scripts/eval_beats.py` → `docs/benchmarks/step-1-beats.md`；README 引用 | **通过**：拍点 F1 0.891（GTZAN 998 首，RTX 3090） |
| 4 分钟歌曲在一张 GPU 上：分轨 + 节拍 ≤ 90 秒 | `scripts/time_pipeline.py` → `docs/benchmarks/step-1-timing.md`（RTX 3090） | **通过**：热启动 7.1 秒，冷启动 13.7 秒 |
| `openbapo analyze` 输出含 `Beat`、`Tempo` | `tests/test_pipeline.py`、`tests/test_cli.py`（假模型）；`tests/test_models.py` 与 CI `analysis-smoke` 任务（真模型，节拍器音频 BPM ≈ 120） | 已实现 |
| 分轨文件在流程结束后不残留 | `tests/test_separate.py`（临时目录）、`tests/test_pipeline.py`（流水线正常与异常路径）、`tests/test_models.py`（真模型） | 已实现 |
| 模型权重许可证逐个列在 `docs/plans/backend.md` | backend.md 第 2.1 节 | 已写 |
