# 第 3 步实现计划：编配引擎

> 对应：`docs/ROADMAP.md` 第 3 步；需求 `docs/REQUIREMENTS.md` 5.3（编配）、5.5（调整能力）、9（换算验收）；`docs/plans/backend.md` 第 3 节。
> 状态：实现完成，待负责人审阅 PR（2026-11-11）。负责人决定见第 8 节。

## 1. 目标与范围

把第 2 步的分析结果（`Analysis`：拍网格、和弦、调性）和用户的调整参数（`Arrangement`：演唱调偏移、指法调、节奏型、简化开关）变成一份**可渲染结构**：页眉、每小节的和弦名与按法、节奏型、低置信标记。第 4 步只管把这个结构画出来，不再做任何音乐判断。

**本步做**：

1. 调的换算：演唱调 = 原调 + 偏移；候选指法调及其变调夹品数、排序与可选性。
2. 指法库（JSON）：12 根音 × 14 性质，每个至少一种按法，带难度与把位；默认选法规则。
3. 节奏型库（JSON）：≥ 10 种，覆盖 4/4、3/4、6/8，每种带口诀与推荐 BPM 区间；默认推荐规则。
4. 简化和弦映射。
5. 和弦名拼写（升降号、斜线和弦）。
6. 页眉：原调、演唱调、指法调 + 变调夹、BPM、拍号、简谱 `1=`。
7. `openbapo arrange` 命令：读 `analysis.json`，写可渲染结构的 JSON，供第 4 步和测试用。

**本步不做**：任何渲染（SVG / PNG / PDF / MusicXML，第 4 步）；简谱音符的时值记号换算（第 4 步按 `Note.start_tick / duration_ticks` 做）；歌词（第 5 步）；逐和弦编辑、变调夹手动覆盖（需求第 10 节）。

## 2. 设计

### 2.1 模块

```
src/openbapo/arrange/
    __init__.py      # arrange(analysis, arrangement) -> ArrangedSheet；公开类型
    keys.py          # 音名拼写、演唱调、候选指法调与变调夹
    chords.py        # 和弦显示名、简化映射
    shapes.py        # 指法库的加载与默认选法
    strums.py        # 节奏型库的加载与推荐
    sheet.py         # ArrangedSheet 及其子结构（pydantic），装配逻辑
    data/shapes.json # 指法库（由 scripts/make_shapes.py 生成并提交）
    data/strums.json # 节奏型库（手工维护）
scripts/make_shapes.py   # 开放把位按法 + 可移动模板 → shapes.json
```

`openbapo.arrange` 只依赖 `openbapo.model` 与 pydantic，不依赖 torch，符合"编配 + 渲染可在无 GPU 环境装和测"的约定（第 0 步决定）。

### 2.2 调的换算（`keys.py`）

- 音级用 0–11（C = 0）。演唱调 `sung = (原调主音 + transpose) mod 12`，调式随原曲；原调缺失（分析没给出 `Key`）时演唱调按 C 大调处理并在 `warnings` 里说明。
- 候选指法调固定（决定 1）：大调歌 C、G、D、A、E、F；小调歌 Am、Em、Dm。
- `capo = (sung − fingering) mod 12`。品数 0–7 为"推荐"，8–11 不推荐（排后、前端标灰），不会出现负数（取模后无负数；但"演唱调 − 指法调 < 0"的组合品数 ≥ 8 自然落到不推荐一档，等价于需求的"负品数不可选"，见测试）。
- `fingering_options(sung, mode)` 返回按品数升序、同品数按候选表顺序的列表，每项含 `tonic`、显示名、`capo`、`recommended`。
- 用户未指定指法调时取列表第一项（品数最小，同品数按候选表顺序）。用户指定的指法调不在候选表里，或在当前演唱调下品数 > 11（不可能）时，退回默认并记 `warnings`（前端据此在嵌入里说明，见 `frontend-discord.md` 的演唱调下拉）。

### 2.3 和弦名拼写（`chords.py`）

- 显示名 = 根音名 + 性质后缀 + 可选的斜线低音。后缀：maj ""、min "m"、7 "7"、maj7 "maj7"、min7 "m7"、maj6 "6"、min6 "m6"、sus2 "sus2"、sus4 "sus4"、5 "5"、dim "dim"、aug "aug"、dim7 "dim7"、hdim7 "m7-5"。
- 升降号写在音名前（`#F`、`bB`），按中文吉他谱习惯（决定 2）。
- 拼法随指法调的调号：升号调（G、D、A、E、B 及其关系小调）全用升号；降号调（F、bB、bE、bA、bD 及其关系小调）全用降号；C 大调 / A 小调用最接近的：`#C`、`#F` 用升号，`bE`、`bA`、`bB` 用降号。
- 和弦按指法调书写：`written_root = (chord.root − capo) mod 12`，斜线低音同理。简谱 `1=` 写演唱调（需求 5.3）。

### 2.4 简化和弦（`chords.py`）

开关默认开（需求 8 节）。映射只影响显示名与按法选择，底层 `Chord` 不变；输出里同时保留未简化的名字，供低置信提示与导出用。

| 原性质 | 简化为 |
|---|---|
| maj7、7、maj6、sus2、sus4、5 | maj |
| min7、min6 | min |
| dim7、hdim7 | dim |
| maj、min、dim、aug | 不变 |
| 斜线和弦 X/Y | X（去掉低音） |

### 2.5 指法库（`shapes.py`、`data/shapes.json`）

- 每个条目：`root`、`quality`、`frets`（6 个值，低弦在前，`null` = 不弹）、`base_fret`（图上第一品对应的实际品位，开放把位为 1）、`barre`（横按从哪根弦起，无则 null）、`difficulty`（1 开放把位；2 有横按但 ≤ 3 根弦或把位 ≤ 3；3 全横按；4 把位 > 7）、`source`（`open` 手工录入 / `E-form` / `A-form` / `D-form` 模板）。
- 开放把位按法手工录入（C、D、E、F、G、A、B 常用大小三、七、大七、小七、挂留等约 40 个）；其余由三套可移动模板按根音平移生成，模板逐弦核对过音高（E 形 12 种性质、A 形 14 种、D 形 1 种）。生成脚本提交，JSON 也提交，`tests/test_arrange_shapes.py` 断言 12 × 14 全覆盖且每个按法的音级集合与性质一致。
- 默认选法：难度最低；并列时取与上一个和弦把位最近的（`base_fret` 差最小）；再并列取开放把位。整首歌同一个和弦名固定用同一按法（第一次出现时定），和弦图条才不会重复画。
- 斜线和弦：按法取 X 本身的按法；低音在图上不体现（阶段 1 不做逐和弦指法）。

### 2.6 节奏型库（`strums.py`、`data/strums.json`）

- 每个条目：`id`、`name`（中文）、`mnemonic`（口诀，如"下 下上 上下上"）、`meter`（`[numerator, denominator]`）、`kind`（`strum` 扫弦 / `arpeggio` 分解）、`bpm`（推荐区间）、`tags`、`events`（每小节内的动作列表：`beat` 以拍为单位从 0 起，`stroke` 为 `down` / `up` / `mute`，分解则给 `strings`，1 = 最高音弦，`bass` = 该和弦的根音弦）。
- 初始内容手工整理 14 种：4/4 八种（四分下扫、民谣 下 下上 上下上、下 下上 下 下上、切音、两种分解、慢摇、节奏布鲁斯感的八分下扫）、3/4 三种、6/8 三种。
- 推荐规则：先按拍号筛；再取推荐 BPM 区间包含歌曲 BPM 的；多条时按库里的顺序取第一条；没有包含的就取区间中点离 BPM 最近的。`Arrangement.strum_pattern` 为 `auto` 走推荐，为 `none` 则六线谱行只显示和弦变化点，否则按 `id` 取（不存在的 id 退回推荐并记 `warnings`）。
- 6/8 的拍以八分音符计（第 1 步决定 5），所以 `events.beat` 的范围是 0–5。

### 2.7 可渲染结构（`sheet.py`）

```
ArrangedSheet
  header: Header(title, artist, original_key, sung_key, fingering_key, capo, bpm,
                 time_signature, jianpu_key, transpose)        # 名字都是显示字符串
  fingering_options: [FingeringOption(tonic, name, capo, recommended, selected)]
  strum: StrumPattern | None                                   # None = 只显示和弦变化点
  strum_options: [StrumOption(id, name, mnemonic, kind, recommended)]  # 同拍号的全部
  shapes: {name: Shape}                                        # 本谱用到的按法，画和弦图条用
  bars: [Bar(index, start, end, beats, chords: [BarChord(beat, name, original_name,
             shape, low_confidence)])]
  notes, lyrics                                                # 原样带过（第 4 / 5 步用）
  low_confidence_count: int
  warnings: [str]
```

- 小节由 `Beat.bar_index` 分组；和弦落在拍上（第 2 步保证），`BarChord.beat` 是它在小节内的起拍（0 起）。一个和弦跨小节时每个小节开头都重写一次。没有和弦的拍不填占位（第 2 步"和声稀薄留空"的语义）。
- 低置信：`Chord.confidence < LOW_CONFIDENCE`（0.5，模块常量，可配置）记 `low_confidence=True`，总数进 `low_confidence_count`（Discord 嵌入提示用）。
- 页眉文字：`原调 G · 演唱调 F · E 调指法 夹 1 品`；小调写 `Em`；`jianpu_key` 写演唱调的大调名（小调歌写关系大调，与第 2 步决定 16 一致，如 `1=G`）。

### 2.8 不变量（测试直接断言）

- 只改 `transpose` 不改 `fingering_key`：所有 `BarChord.name` 不变，只有页眉的演唱调与 `capo` 变。
- 只改 `fingering_key`：`capo` 随之变，`name` 按新指法调重写，但每个和弦的**发声音级**（`written_root + capo`）不变。
- `simplify_chords` 只影响 `name` 与 `shape`，`original_name` 永远是未简化的。
- 候选指法调里没有负品数；品数 ≤ 7 的排在前面。

## 3. 测试与 CI

| 测试 | 覆盖 |
|---|---|
| `test_arrange_keys.py` | 12 演唱调 × 12 指法调的品数全覆盖（参数化 144 组）；大调 / 小调候选表；排序与推荐标记；默认指法调；拼写表（每个音级在升号调 / 降号调 / C 调的写法） |
| `test_arrange_chords.py` | 14 种性质的显示名；简化映射；斜线和弦；按指法调重写 |
| `test_arrange_shapes.py` | 12 × 14 全覆盖，缺失项报出；每个按法的音级集合 ⊆ 性质音级且含根音；默认选法（难度、把位相近、整首固定） |
| `test_arrange_strums.py` | ≥ 10 条、三种拍号都有；`events.beat` 在拍号范围内且递增；推荐规则；`none` / 未知 id |
| `test_arrange_sheet.py` | 用合成的 `Analysis`（含跨小节和弦、空拍、低置信和弦、小调歌）装配出完整结构；2.8 的四条不变量；缺 `Key` / 缺 `Tempo` 的降级；JSON 往返 |
| `test_cli.py` | `openbapo arrange` 读写 |

全部在 CI 的 CPU 作业里跑，不需要模型。

## 4. 数据模型改动

不改 `Analysis` 与 `Arrangement`（`Arrangement` 的五个字段是第 0 步定的用户可调项，测试锁定）。`ArrangedSheet` 是新的输出类型，放在 `openbapo.arrange`，不进 `openbapo.model`（它不是存储格式，每次由参数现算）。

## 5. 依赖

无新依赖。

## 6. 与前后步骤的接口

- 输入：第 2 步的 `Analysis`（`analyze` 的 JSON）。
- 输出：`ArrangedSheet`，第 4 步据此画和弦图条（`shapes`）、六线谱行（`strum.events` 或和弦变化点）、小节与和弦名（`bars`）、页眉（`header`）、简谱（`notes` + `header.sung_key`）。
- Discord（第 6 步）的三个下拉直接用 `fingering_options`、`strum_options` 与 `warnings`。

## 7. 验收数字

本步验收全是功能性断言（见路线图），没有需要在 GPU 上产出的数字；`pytest` 结果即验收依据。

## 8. 负责人决定

> 负责人指示"写计划并执行"，以下按推荐项实现；要改只需换一张表或一个常量。

### 1. 候选指法调：大调 C、G、D、A、E、F；小调 Am、Em、Dm

- 需求 5.3 写"C / G / D / A / E 等好按的调"，`backend.md` 写"C / G / D / A / E / F / Am / Em / Dm 等常用调"，这里取两者的并集。
- 没有 B、#F、bB 等调：它们的常用和弦几乎全是横按，不符合"好按"的目的；想用时等价于选相邻的调再换变调夹。
- 小调歌用小调指法（Am 调指法 = C 调的和弦集合，但主和弦是 Am），和弦名与谱面习惯一致。

### 2. 升降号写在音名前（`#F`、`bB`）

- 负责人曲库的 220 份谱子和参考排版 `desperado-sheet.webp` 都是这种写法；第 2 步的谱子解析器已经两种都认。
- 导出 MusicXML 时（第 4 步）转成标准拼法，不受影响。

### 3. 简化映射把 dim7 / hdim7 归为 dim，aug 不动

- `backend.md` 只写了 maj / min 两族；减七、半减七对初学者同样难按，归为减三和弦；增三和弦没有更简单的替代，保留。

## 9. 已知缺口

| 缺口 | 处理 |
|---|---|
| 指法库只有一种默认选法，用户不能换按法 | 需求第 10 节已移出阶段 1 |
| 节奏型只按拍号 + BPM 推荐，不看风格 | 阶段 1 够用；风格标签已留在库里 |
| 开放把位按法是手工录入的约 40 个 | 覆盖常用调的常用和弦；其余走横按模板，难度标 3–4 |
| 分解节奏型的"根音弦"在斜线和弦上取原根音 | 阶段 1 不做逐和弦指法 |

## 10. 交付顺序

1. 本计划。
2. `keys.py` + `chords.py` + 测试。
3. 指法库生成脚本 + JSON + `shapes.py` + 测试。
4. 节奏型库 JSON + `strums.py` + 测试。
5. `sheet.py` 装配 + 不变量测试 + CLI。
6. 文档：`backend.md` 第 3 节对照、README 进度、本计划第 11 节；问负责人是否开 PR。（已做）

## 11. 验收对照

| 验收项（ROADMAP 第 3 步） | 如何证明 | 状态 |
|---|---|---|
| 12 演唱调 × 12 指法调换算有单元测试全覆盖；负品数不出现在候选里 | `test_arrange_keys.py`：`test_capo_covers_every_key_pair` 参数化 144 组；`test_no_negative_capo_and_recommended_first` 遍历 24 个演唱调 × 候选表 | 已做 |
| 指法库覆盖 12 根音 × 14 种性质，每个至少一种按法；缺失项有测试报出 | `test_arrange_shapes.py`：`shapes.missing() == []`，每种按法的音级逐一核对；`scripts/make_shapes.py --check` 保证 JSON 与生成脚本一致 | 已做（346 种按法） |
| 给定 `Analysis` + `Arrangement` 得到完整的可渲染结构 | `test_arrange_sheet.py`（跨小节和弦、空拍、低置信、小调、缺调性 / 缺拍网格的降级、JSON 往返）；`openbapo arrange` 在合成和弦进行与《曾经的你》的渲染音频上跑通 | 已做 |
| 只改演唱调不改指法调时和弦名不变 | `test_arrange_sheet.py::test_transposing_keeps_the_chord_names_when_the_fingering_key_is_fixed`（−6 … +6 全部偏移） | 已做 |
