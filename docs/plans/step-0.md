# 第 0 步实现计划：项目骨架与数据模型

> 对应：`docs/ROADMAP.md` 第 0 步；需求 `docs/REQUIREMENTS.md` 5.2（数据模型）、5.3 / 5.5（调整参数）。
> 状态：第 7 节决定已全部确认，实施中。

## 1. 目标与范围

本步只搭骨架，不接任何模型、不读音频内容。做完后应当有：

1. 可安装的 Python 包 `opencapo` + 格式化 / lint / 类型检查 / 测试工具链 + GitHub Actions CI。
2. 数据模型：`Beat`、`Tempo`、`Key`、`Chord`、`Note`、`LyricSyllable`、顶层 `Analysis`，带 JSON 序列化 / 反序列化。
3. 调整参数对象 `Arrangement`。
4. 空的 CLI：`opencapo analyze <audio> --out <json>`，跑通参数解析并输出合法的空 `Analysis`。

**本步不做**：分轨 / 节拍 / 和弦等任何推理（第 1、2 步）；编配换算（变调夹、指法库、节奏型库，第 3 步）；渲染；Discord；数据库与队列。

**边界说明**：
- `Stem`（分轨）**不进** `Analysis`，也不进 JSON。它只在第 1 步的流水线内部以临时文件存在，数据模型里不给它留字段，从结构上杜绝被存储或下发。
- `analyze` 输出的 JSON 是流水线内部的中间结果（Worker 写、编配 / 渲染读），**不是**面向用户的"项目 JSON 导出"，也不会在 Discord 里提供下载。
- CLI 是开发与 Worker 的入口，不是面向用户的命令行产品（需求第 2 节"不做本地命令行软件"）。

## 2. 仓库结构

```
pyproject.toml
src/opencapo/
    __init__.py          # __version__
    __main__.py          # python -m opencapo
    cli.py               # argparse 入口
    model/
        __init__.py      # 对外导出全部模型
        music.py         # PitchClass、Mode、ChordQuality 等基础类型
        analysis.py      # Beat、Tempo、Key、Chord、Note、LyricSyllable、Analysis
        arrangement.py   # Arrangement
tests/
    data/analysis_full.json   # 手写的"字段全填"样例，作为格式金样
    test_model_analysis.py
    test_model_arrangement.py
    test_roundtrip.py         # 金样 + 属性测试的 JSON 往返
    test_cli.py
.github/workflows/ci.yml
```

- 采用 `src/` 布局，避免测试误用未安装的源码。
- 数据模型放 `opencapo.model`，把 `opencapo.analysis` 这个名字留给第 1 步起的分析流水线；之后的 `opencapo.arrange`、`opencapo.render` 同理。Discord 适配层第 6 步再建，且只依赖 `opencapo` 的公开接口。
- 核心安装保持轻量：本步运行时依赖只有 `pydantic`。第 1 步起的 torch / 模型依赖放进可选依赖组（如 `opencapo[analysis]`），这样编配 + 渲染可以在没有 GPU 依赖的环境里装和测。

## 3. 工具链

| 项 | 选择 | 说明 |
|---|---|---|
| 构建后端 | hatchling | 纯 Python，配置少 |
| Python 版本 | `requires-python = ">=3.11"`；CI 跑 3.11 / 3.12 / 3.13 | 第 1 步确定模型依赖后再收紧下限或加上限（madmom、部分 torch 生态对新版本支持滞后） |
| 格式化 + lint | ruff（`ruff format` + `ruff check`） | 一个工具覆盖 black / isort / flake8 |
| 类型检查 | mypy（`src/` 开 strict） | 数据模型是全项目的契约，值得严格 |
| 测试 | pytest + hypothesis | hypothesis 用于 JSON 往返的属性测试 |
| 开发依赖 | `pip install -e ".[dev]"` | `dev` 组包含 ruff、mypy、pytest、hypothesis |

不引入 pre-commit、tox、nox 等额外层；CI 里直接跑这四条命令，与本地开发者跑的一致。

## 4. 数据模型

### 4.1 实现方式

用 **pydantic v2** 的冻结模型（`frozen=True`、`extra="forbid"`）：
- 构造即校验（区间、取值范围、排序），坏数据在写入端就报错，而不是在渲染时才炸。
- 自带 JSON 序列化 / 反序列化；`Analysis.model_dump_json()` / `Analysis.model_validate_json()`，外面再包一层 `to_json()` / `from_json()` 作为项目内的稳定接口。
- `extra="forbid"` 保证未知字段不会被静默丢弃（"往返无损"的另一面）。
- 标量字段用 pydantic 的 `StrictInt` / `StrictFloat` / `StrictBool` / `StrictStr`，拒绝静默类型转换（`"3"` → 3、`true` → 1）；float 字段仍接受整数。没有用模型级 `strict=True`：模型一旦带 before 校验器，pydantic 会按 Python 模式校验 JSON，模型级严格模式随之拒绝 JSON 数组充当 tuple。
- 列表字段用 `tuple`，保证冻结后真正不可变。

### 4.2 基础类型（`model/music.py`）

| 类型 | 表示 | 说明 |
|---|---|---|
| `PitchClass` | 整数 0–11，C = 0 | JSON 里存整数，不存音名。升 / 降号写法（F# 还是 Gb）取决于调，是编配 / 渲染的事，不在分析结果里定死 |
| `Mode` | `"major"` / `"minor"` | |
| `ChordQuality` | 字符串枚举，沿用 Harte 记法的简写 | `maj`、`min`、`7`、`maj7`、`min7`、`maj6`、`min6`、`sus2`、`sus4`、`5`、`dim`、`aug`，再加 `dim7`、`hdim7`（m7b5），共 14 种（见第 7 节决定 2）。用 Harte 记法便于第 2 步直接用 mir_eval 评测 |

### 4.3 分析结果（`model/analysis.py`）

时间一律为秒（float，相对原音频开头）。所有 float 字段要求有限值（拒绝 NaN / inf，否则 JSON 无法无损往返）。

| 对象 | 字段 | 校验 |
|---|---|---|
| `Beat` | `time`, `is_downbeat`, `bar_index`, `beat_in_bar` | `time ≥ 0`；`bar_index ≥ 0`；`beat_in_bar ≥ 1`（从 1 数，与"第几拍"的说法一致）；`is_downbeat ⇔ beat_in_bar == 1` |
| `Tempo` | `bpm` | `bpm > 0`；只有全局值 |
| `Key` | `tonic: PitchClass`, `mode: Mode`, `confidence` | `confidence ∈ [0, 1]` |
| `Chord` | `start`, `end`, `root: PitchClass`, `quality: ChordQuality`, `bass: PitchClass \| None`, `confidence` | `0 ≤ start < end`；`bass` 为 `None` 表示原位，否则是转位低音（如 D/F# 的 F#），且 `bass ≠ root`；`confidence ∈ [0, 1]` |
| `Note` | `start`, `end`, `pitch`, `start_tick`, `duration_ticks` | `0 ≤ start < end`；`pitch ∈ [0, 127]`（MIDI）；`start_tick ≥ 0`、`duration_ticks ≥ 1` |
| `LyricSyllable` | `start`, `end`, `text`, `line_index`, `note_ref: int \| None` | `0 ≤ start ≤ end`；`text` 非空；`line_index ≥ 0`；`note_ref` 为 `Analysis.notes` 的下标，M2 之前为 `None` |
| `Analysis` | `schema_version`, `title`, `artist`, `duration`, `beats`, `tempo`, `key`, `chords`, `notes`, `lyrics` | 见下 |

**`Note` 的量化时值**：量化结果用整数刻度存，`TICKS_PER_BEAT = 12`（作为模块常量，也写进文档），刻度从 `beats[0]` 起算。12 能整除 2 / 3 / 4 / 6，覆盖十六分音符和三连音。存整数而不是分数或浮点，保证 JSON 往返精确。附点、连音线是**记谱**层面的事：一个跨小节线的音在渲染时才拆成两段加连音线，附点也由渲染根据 `duration_ticks` 推出，因此都不单独存储（见第 7 节决定 3）。

**`Analysis` 顶层**：
- `schema_version: Literal[1]`，从 1 开始。阶段 1 内字段有增改就加 1；读到不认识的版本、或 JSON 里缺这个字段，都直接报错，不做猜测性兼容（第 6 步之前没有持久化数据，迁移成本为零）。在代码里新建 `Analysis` 时自动填当前版本。
- `title`、`artist`：可选，页眉用；CLI 未给 `--title` 时用文件名（去扩展名）。
- `duration`：音频时长（秒），可选，第 1 步解码后填。
- `tempo`、`key`：可选（`None` = 尚未分析）。
- `beats`、`chords`、`notes`、`lyrics`：列表，默认空。**空 `Analysis` 是合法的**，这就是本步 CLI 的输出。
- 跨对象校验：`beats` 按时间严格递增；`chords` 按 `start` 排序且互不重叠（没有和弦的区间就是空着，不用 "N" 占位）；`notes` 按 `start` 排序且在秒和 tick 上都互不重叠（人声旋律是单声部）；`beats` 的 `bar_index` 单调不减；`lyrics` 按 `start` 排序、`line_index` 单调不减；`note_ref` 不越界。
- "和弦边界落在拍上"是第 2 步的验收项，不做成模型层的硬校验（否则第 2 步之前的测试数据都得造拍网格）。

### 4.4 调整参数（`model/arrangement.py`）

`Arrangement` 只包含需求 5.5 允许用户调的东西，全部有默认值；**"重置"就是 `Arrangement()`**。

| 字段 | 类型 / 默认 | 含义 |
|---|---|---|
| `transpose` | `int`，默认 `0`，范围 `−6 … +6` | 演唱调相对原调的半音偏移（与前端下拉的范围一致） |
| `fingering_key` | `PitchClass \| None`，默认 `None` | 指法调主音；调式跟随原调（小调歌曲对应 Am / Em / Dm 等）。`None` = 由系统选品数最小的 |
| `strum_pattern` | `str`，默认 `"auto"` | 节奏型库里的 ID；保留值 `"auto"`（按速度 / 拍号推荐）与 `"none"`（不显示节奏型） |
| `simplify_chords` | `bool`，默认 `True` | 简化和弦显示（需求已定默认开） |
| `lyrics_text` | `str \| None`，默认 `None` | 用户粘贴的歌词；`None` = 用识别结果 |

本步只做字段与取值范围校验。"指法调是否让品数为负""节奏型 ID 是否存在"依赖分析结果和第 3 步的库，留给第 3 步。手动覆盖变调夹、改单个和弦属于需求第 10 节，**不设字段**。

## 5. CLI

```
opencapo analyze <audio> --out <json> [--title TEXT] [--artist TEXT]
```

- 用标准库 `argparse`，不引入 click / typer。`python -m opencapo` 等价于 `opencapo`（`[project.scripts]` 注册）。
- 本步行为：检查文件存在、扩展名属于需求 5.1 的六种（mp3 / m4a / ogg / opus / wav / flac）→ 构造空 `Analysis`（只填 `title` / `artist`）→ 写到 `--out`。`--out -` 写到标准输出。
- 失败不静默：文件不存在、格式不支持、输出文件写不进去时，向 stderr 输出一句可读的原因，退出码 1；参数错误由 argparse 处理，退出码 2。
- 8 分钟时长上限要解码音频才能判断，放到第 1 步。
- `--lyrics` 等参数等第 5 步真正用到时再加，本步不预留空参数。

**示例音频**：不往仓库里提交任何真实歌曲。`scripts/make_tone.py` 用标准库 `wave` 生成一段 3 秒的正弦波 wav；测试里把它写到 pytest 的 `tmp_path`，对它跑 CLI，再用 `Analysis.from_json()` 读回校验。CI 里另跑一次同样的冒烟命令，产出的 JSON 作为 CI 日志的一部分，便于对照验收项。

## 6. 测试

| 测试 | 覆盖 |
|---|---|
| 每个模型的合法 / 非法构造 | 第 4.3 / 4.4 节每条校验至少一正一反（`start ≥ end`、置信度越界、`bass == root`、`beat_in_bar = 0`、NaN、未知字段、乱序列表、`note_ref` 越界、`transpose = 7` 等） |
| 金样往返 | `tests/data/analysis_full.json`（每个字段都有值、含转位与低置信和弦、中英文歌词）→ `from_json` → `to_json` → 再解析，两次对象相等，且与原文件解析结果相等 |
| 属性测试往返 | hypothesis 生成任意合法 `Analysis` / `Arrangement`，断言 `from_json(to_json(x)) == x`，覆盖 float 精度、Unicode 歌词、空列表、`None` 字段 |
| 版本 | `schema_version` 不认识时报错 |
| CLI | 正常输出、`--out -`、默认标题取文件名、文件不存在、格式不支持、缺参数 |

## 7. 负责人决定（已全部确认）

1. **已定：数据模型用 pydantic v2**。备选是标准库 dataclass + 手写校验与序列化（零依赖，但多一块自维护代码），未采用。
2. **已定：第 13、14 种和弦性质为 `dim7`、`hdim7`（m7b5）**。需求列了 12 种，"等 14 种"。选这两个是因为它们在 Isophonics（Harte 记法）标注和 BTC 等大词表和弦模型的类别里都有，模型可直接输出，mir_eval 评测也无需额外映射。未采用 `add9`、`7sus4`：吉他谱常见，但 Harte 词表模型不输出。`minmaj7` 同在这些词表里，但弹唱谱罕见，暂不加。第 2 步选型时报告各候选模型能输出哪些性质，届时如需调整枚举，升 `schema_version` 即可。
3. **已定：`Note` 只存"声音"，不存"记谱"**。存观测值 `start` / `end` / `pitch` 与量化值 `start_tick` / `duration_ticks`（整数，`TICKS_PER_BEAT = 12`，定义在拍网格上）。附点、连音线、三连音由渲染推出：附点由时值决定，单独存会冗余且可能矛盾；连音线取决于拍号、按拍分组规则和换行，属于排版规则；music21 也是由时值自动拆分加 tie。需求 5.2 的措辞已同步修改。未采用：照字面额外存 `dotted` / `tie` 并做一致性校验。
4. **已定：`requires-python = ">=3.11"`，CI 测 3.11 / 3.12 / 3.13**。第 1 步确定模型栈后再收紧。未采用：`>=3.12`（放弃仍在维护期的 3.11）、锁定单一版本（本地开发与自部署被卡死）。部署镜像另行锁定确切版本。

## 8. 已知缺口（不在本步补，记下来以免遗漏）

数据模型只覆盖需求 5.2 列出的对象。以下字段后续步骤会需要，到对应步骤再加并升 `schema_version`：

| 缺口 | 谁需要 | 预计在 |
|---|---|---|
| 拍号（区分 3/4 与 6/8，页眉也要显示） | 节奏型推荐、页眉 | 第 1 步（节拍模型给出每小节拍数）或第 3 步 |
| 英文音节的词边界（Des-pe-ra-do 的连字符、MusicXML 的 `syllabic`） | 歌词渲染、MusicXML | 第 4 / 5 步 |
| 歌词语言（中 / 英） | 选择对齐模型 | 第 5 步 |

拍号会改动 5.2 的对象表，按 CLAUDE.md 要求，届时先改需求再写代码。

讨论中发现的、属于后续步骤的风险：

| 风险 | 说明 | 处理步骤 |
|---|---|---|
| `5`（强力和弦）可能没有模型能输出 | BTC 的 14 种性质里没有 `5`；Isophonics 里有但样本很少。需选词表含 `5` 的模型，或用后处理规则（如三音缺失判定） | 第 2 步 |
| Isophonics 测试集污染 | BTC 等模型的训练数据通常包含 Isophonics，用公开权重在 Beatles 上测，准确率会偏高。**已消解**：第 2 步改用候选模型都没见过的 GuitarSet 与 AAM（Beatles 音频也拿不到），见 `step-2.md` 决定 2 | 第 2 步 |
| tick 的拍单位依赖拍号 | 6/8 中检测到的"拍"可能是附点四分或八分，tick → MusicXML quarterLength 的换算需要知道，与上表"拍号"缺口一并解决 | 第 1 步 |
| 第一拍之前的人声无法表示 | `start_tick ≥ 0`；若出现，在网格前补拍或丢弃 | 第 7 步 |
| 简化规则未覆盖 dim / aug / dim7 / hdim7 | `docs/plans/backend.md` 第 3 节的简化映射没写这四种（例如 hdim7 简化成 dim 还是 min） | 第 3 步 |

## 9. 交付顺序

一个 PR，按以下提交顺序推进，每个提交本地四条检查都过：

1. `pyproject.toml`、包骨架、`__version__`、ruff / mypy / pytest 配置、一个冒烟测试。
2. `.github/workflows/ci.yml`：`push` 到 main 与 `pull_request` 触发；3.11 / 3.12 / 3.13 矩阵；`pip install -e ".[dev]"` → `ruff format --check` → `ruff check` → `mypy src` → `pytest` → CLI 冒烟。
3. 基础类型 + 分析模型 + 校验测试。
4. `Arrangement` + 测试。
5. 金样 JSON + 往返属性测试。
6. CLI + 测试。
7. README "状态"一节更新为"第 0 步完成"，写明开发环境搭建的三条命令。

## 10. 验收对照

| 验收项（ROADMAP 第 0 步） | 如何证明 |
|---|---|
| `pip install -e .` 成功 | CI 的安装步骤（干净的 runner 上执行） |
| `pytest` 通过 | CI 的测试步骤 |
| CI 在 PR 上自动跑并通过 | 本步 PR 页面上的 CI 检查为绿 |
| 数据模型有单元测试，JSON 往返无损 | `tests/test_model_*.py`、`tests/test_roundtrip.py`（金样 + hypothesis 属性测试） |
| 一首示例音频跑 `opencapo analyze` 能输出合法（可为空）的 JSON | `tests/test_cli.py` + CI 冒烟步骤：生成的 wav → `opencapo analyze` → `Analysis.from_json` 校验通过 |
