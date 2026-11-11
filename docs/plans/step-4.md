# 第 4 步实现计划：渲染与导出（三层谱）

> 对应：`docs/ROADMAP.md` 第 4 步；需求 `docs/REQUIREMENTS.md` 5.6（导出）、附录排版参考；`docs/plans/backend.md` 第 4 节。
> 状态：实现完成，排版待负责人看图确认（2026-11-11）。负责人决定见第 8 节。

## 1. 目标与范围

把第 3 步的 `ArrangedSheet` 画成谱：和弦图条、六线谱节奏型行、简谱行（本步留空位）、歌词行（本步占位）、页眉、小节线与小节号；出 SVG、PNG（第一页预览）、多页 PDF，以及可导入 MuseScore 精修的 MusicXML。排版以 `docs/reference/desperado-sheet.webp` 为标准。

**本步做**：

1. SVG 排版引擎：A4 纵向，一个"系统"= 和弦图条 + 六线谱行 + 简谱行 + 歌词行，每行 4 小节；页眉（标题、演唱者、`♩= BPM`、`1=X 拍号`、原调 / 演唱调 / 指法调 / 变调夹）；自动分页。
2. 六线谱行：按节奏型画。扫弦画成各弦上的 ×、符干与符杠（八分连杠）、上下箭头；分解画成被拨弦的品位数字；"无"节奏型只在和弦变化点画一个 × 加符干。
3. 简谱行：本步画出行的位置与小节线；音符（M2，第 7 步）暂不画。`notes` 不为空时按最简规则画数字（音级、高低点、整拍一个数字），不做下划线 / 附点 / 连音线的细排版。
4. 歌词行：本步占位（`lyrics` 为空时留白；不为空时按音节起始时间摆到对应拍的位置）。
5. 低置信和弦：和弦名用灰色并加 `?`。
6. SVG → PNG（cairosvg）、SVG 多页 → PDF（cairosvg 逐页 + pypdf 合并）。
7. MusicXML：和弦符号（`harmony`）+ 六线谱（TAB 谱表，节奏型的弦 / 品）+ 歌词音节 + 一条标准谱表（放和弦符号与以后的旋律）。
8. `openbapo render` 命令：读 sheet JSON，写 `--svg DIR`、`--png FILE`（第一页）、`--pdf FILE`、`--musicxml FILE`。

**本步不做**：简谱时值记号的细排版（下划线、附点、连音线、跨行连线，第 7 步）；真实歌词（第 5 步）；Discord 交互；排版的可配置项（字号、每行小节数等先固定）。

## 2. 设计

### 2.1 模块

```
src/openbapo/render/
    __init__.py      # render_svg / render_png / render_pdf / render_musicxml
    layout.py        # 尺寸常量；把 bars 切成系统与页；每小节内拍的 x 坐标
    svg.py           # 画：页眉、和弦图、六线谱行、简谱行、歌词行、小节线、小节号
    export.py        # SVG → PNG / PDF
    musicxml.py      # ArrangedSheet → MusicXML 文本
```

依赖新增可选组 `openbapo[render]`：`cairosvg`（PNG / PDF，需系统 libcairo）、`pypdf`（合并页）。MusicXML 用标准库 `xml.etree` 直接写，不引入 music21；测试用 `music21` 读回校验（只进 `dev` 组）。

### 2.2 坐标与分页（`layout.py`）

- 页面 A4：595 × 842 pt，边距 40 pt。所有尺寸以 pt 为单位写成常量。
- 系统高度固定：和弦图条 56 pt + 六线谱 6 线 × 7 pt + 间距 + 简谱行 24 pt + 歌词行 16 pt ≈ 150 pt；第一页页眉 90 pt；A4 一页放 4 个系统（第一页）/ 5 个（后续页）。
- 每行 4 小节，小节等宽；小节内第 k 拍的 x = 左边 + (k + 0.5 的余量) × 拍宽。`BarChord.beat`、节奏型 `events.beat`、音符与歌词的拍位置都映射到同一套 x。
- 第一小节左侧留出谱号区（TAB 字样 + 拍号）。

### 2.3 画法（`svg.py`）

- **和弦图**：5 品 × 6 弦的网格，30 × 36 pt；`base_fret > 1` 时左侧写品位数；× / ○ 画在上沿；横按画粗线。和弦名在图上方，居中于和弦起拍的 x。同一小节两个和弦并排。
- **六线谱行**：6 条线，左端 `T A B`；扫弦事件：每根发声弦画 ×，最低弦下方符干，相邻八分用符杠连，事件上方画 ↓（down）/ ↑（up），切音（mute）画 × 加小圆点；分解事件：按法的品位数字写在对应弦上，`bass` 取按法最低发声弦。节奏型为"无"时只在和弦起拍画一个 × 和符干。
- **简谱行**：`1=X` 已在页眉；行内画小节线；有音符时每个整拍位置写音级数字，高低八度画上下点，休止写 0。
- **歌词行**：音节文字放到其起拍 x。
- **低置信**：和弦名 `#666` 灰色，后缀 `?`。
- 字体：`font-family="Noto Sans CJK SC, Noto Sans SC, PingFang SC, Microsoft YaHei, sans-serif"`；数字与英文用同一字体。PNG 渲染依赖系统装有其中之一，README 写明（Linux：`fonts-noto-cjk`）。

### 2.4 MusicXML（`musicxml.py`）

- `score-partwise` 4.0 版；两个 part：P1 标准谱表（高音谱号，`harmony` 和弦符号，本步只有休止符占位，以后放旋律与歌词），P2 TAB（`clef sign=TAB`，`staff-details` 六弦标准调弦，节奏型事件写成 `note` + `technical/string` + `technical/fret`，扫弦事件用和弦音 chord 叠加）。
- 歌词音节挂在 P1 的音符上；本步 P1 无音符时歌词不写（第 5 步接）。
- 和弦符号按**发声**音高写（`root-step` / `root-alter`），另加 `capo` 的 `transpose` 到 P2，MuseScore 据此显示指法调的和弦名。简化开关按显示名写。
- 测试用 music21 解析：part 数、measure 数、harmony 数、TAB 弦品数都能读回。

### 2.5 CLI

```
openbapo render SHEET.json [--svg DIR] [--png FILE] [--pdf FILE] [--musicxml FILE] [--dpi 150]
```

### 2.6 性能

SVG 是字符串拼接；一首 4 分钟歌约 100 小节、25 个系统、6 页。cairosvg 栅格 150 dpi 一页约 0.3 s。预算 5 秒内。`scripts/time_pipeline.py` 加 `render` 环节记录。

## 3. 测试与 CI

| 测试 | 覆盖 |
|---|---|
| `test_render_layout.py` | 分页与分行：小节数 → 系统数 → 页数；拍的 x 单调 |
| `test_render_svg.py` | 合成 `ArrangedSheet` → SVG：含标题 / 页眉文字、每个和弦名、和弦图数量、× 与数字的数量随节奏型变化、低置信标记；换指法调后和弦名变而 × 数量不变；"无"节奏型只有变化点 |
| `test_render_export.py` | PNG 尺寸与非空；PDF 页数 = SVG 页数（需要 cairo，缺库时 skip） |
| `test_render_musicxml.py` | music21 读回：2 个 part、小节数、harmony 数、TAB 弦品；MuseScore 打开由负责人确认 |
| `test_cli.py` | `openbapo render` 四种输出 |

CI 的 CPU 作业装 `.[dev,render]`，Ubuntu 自带 libcairo；中文字体不装（测试不比对像素）。

## 4. 验收数字

- 渲染一首 4 分钟歌曲 ≤ 5 秒：`scripts/time_pipeline.py` 加 `render` 环节，结果写进 `docs/benchmarks/step-4-timing.md`。
- 视觉：负责人看图。样张用公有领域的《小星星》（`docs/examples/xiaoxingxing.sheet` 渲染音频 → 分析 → 编配 → 渲染）放进 PR；负责人自己的曲库样张只在本地看。

## 5. 依赖

`render` 可选组：`cairosvg>=2.7`、`pypdf>=4`。`dev` 组加 `music21`（只用于测试读回 MusicXML）。

## 6. 与前后步骤的接口

- 输入：`ArrangedSheet`（第 3 步）。
- 输出：SVG 页列表、PNG 字节、PDF 字节、MusicXML 文本。第 6 步 Discord 发 PNG 第一页与 PDF。
- 第 5 步接真歌词、第 7 步接简谱音符时，只在 `svg.py` 的两行里换画法，排版框架不动。

## 7. 开放问题

1. 简谱行与歌词行在 M1 都基本是空的，第一页会显得空；是否先把空行缩小、等 M2 再放大。看图后定。
2. 节奏型在每小节重复画一遍（参考谱如此）还是只画一次示意（省空间）。看图后定。

## 8. 负责人决定

> 负责人指示"做第四步，一起看结果"：以下按推荐项实现，排版细节以看图反馈为准。

### 1. 四层全部自绘 SVG，不引入排版库

- 中文简谱式弹唱谱没有现成库能画（简谱是硬需求），六线谱节奏型行很简单，自绘最可控；alphaTab / Verovio 画的是西式六线谱，风格与参考谱不合，还要多一套运行时。
- PNG / PDF 用 cairosvg（纯 Python 包 + 系统 libcairo，PDF 是矢量）。

### 2. MusicXML 直接写 XML，不用 music21 生成

- 要写的元素很少（和弦符号、TAB 音符、歌词），标准库够用；music21 包 20 MB，只在测试里读回校验。

### 3. 字体用系统的思源黑体（Noto Sans CJK），不随仓库分发

- 字体文件十几 MB，仓库不放；SVG 里写字体族回退列表，PNG 渲染机器装 `fonts-noto-cjk`（OFL 许可）。

## 9. 交付顺序

1. 本计划。
2. `layout.py` + `svg.py` + 测试；看第一张图。
3. `export.py` + CLI；PNG / PDF。
4. `musicxml.py` + 测试。
5. 计时、文档、README；问负责人是否开 PR。

## 10. 验收对照

| 验收项（ROADMAP 第 4 步） | 如何证明 | 状态 |
|---|---|---|
| 用第 3 步的输出渲染出的 PNG 与参考谱并排放在 PR 里，负责人目视确认"像那张谱" | 《小星星》样张进 PR | 待负责人确认 |
| 渲染一首 4 分钟歌曲 ≤ 5 秒 | `docs/benchmarks/step-4-timing.md`：编配 + SVG + PNG + PDF + MusicXML 共 **2.07 s**（113 小节，CPU） | 已做 |
| MusicXML 能被 MuseScore 4 打开，和弦、六线谱、歌词完整，可以直接改单个和弦并重新排版 | `tests/test_render.py::test_musicxml_reads_back_with_music21`（两个 part、和弦符号按发声音高、TAB 的弦与品都能读回）；MuseScore 打开待负责人确认 | 待负责人确认 |
| 切换指法调 / 节奏型 / 简化开关后重新渲染，结果正确（截图对比测试） | `tests/test_render.py::test_rendering_follows_the_arrangement`：比对 SVG 中的和弦名、箭头与品位数字的数量 | 已做 |
