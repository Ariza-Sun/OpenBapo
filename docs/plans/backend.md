# 后端计划（阶段 1）

> 对应需求：`docs/REQUIREMENTS.md` 第 5.1–5.3、5.6、7 节。本文档回答"服务端怎么做"。
> 状态：草稿。

## 1. 架构草图 **[建议]**

```
前端（Discord Bot）
   │  提交任务 / 查状态 / 渲染 / 重对齐歌词 / 导出
   ▼
API 层
   │
   ├──▶ 任务队列（Redis 或数据库表）
   │        │
   │        ▼
   │   GPU Worker：分轨 → 节拍 → 和弦 → 调性 → 人声旋律 → 歌词识别与对齐
   │        │
   │        ▼
   ├──▶ 结果存储：结构化 JSON（分析结果）+ 当前调整参数
   │
   └──▶ 编配 + 渲染服务：指法 / 节奏型 / 简谱 → SVG → PNG / PDF / MusicXML
```

设计原则：
- **分析与渲染分离**。模型推理只在提交任务时跑一次；之后用户每次调整（改和弦、换指法调、换节奏型）只走编配 + 渲染，秒级返回。
- **核心库与适配层分离**。"音频 → 结构化结果 → 谱"做成独立 Python 包，Discord 只是一个适配层；以后接别的前端不动核心。
- **调整即参数**。用户能改的只有演唱调偏移、指法调、节奏型、简化开关、歌词文本；都是作用在同一份分析结果上的参数，不修改分析结果本身，因此不需要版本历史，"重置"就是参数归零。精修在专业软件里做（MusicXML 导出）。

## 2. 分析流水线

| 环节 | 输入 | 输出 | 候选组件（均有权重、许可证明确） |
|---|---|---|---|
| 分轨 | 混音 | vocals / drums / bass / other | **已定：Demucs `htdemucs`**（第 1 步）；备选 BS-RoFormer |
| 节拍 / 强拍 / BPM | 混音（实测 drums + bass 重混更差：拍点 F1 0.770 对 0.891） | `Beat`、`Tempo`、`TimeSignature` | **已定：Beat This!**（第 1 步，GTZAN 拍点 F1 0.891）；备选 madmom、all-in-one |
| 和弦 | 混音去鼓（bass + other）+ 拍网格 | `Chord`（含转位、置信度） | BTC、ChordFormer、autochord |
| 调性 | `Chord` 序列 + 音频 | `Key` | **已定：规则法**（第 2 步）：色度模板（Krumhansl 轮廓）与分窗和弦统计互校，关系大小调按主和弦时长与首尾和弦投票；不依赖 essentia |
| 人声旋律 | vocals | 音高曲线 → 切分 → 量化到拍网格 → `Note` | RMVPE / CREPE 提取音高；音符切分与量化自写 |
| 歌词识别 | vocals | 文本（用户提供歌词时跳过） | faster-whisper |
| 歌词对齐 | vocals + 文本 | `LyricSyllable` | WhisperX 的 wav2vec2 强制对齐；中文对齐模型需单独评估；英文音节切分 pyphen / CMUdict；音节挂到 `Note` 用时间最近匹配 + 单调约束 |

分轨结果只在流水线内部使用，任务完成后与原音频一起删除。

### 2.1 已接入的模型权重与许可证

每接入一份权重在此登记（CLAUDE.md 要求）。CC-BY-NC 类及许可证不明的显式标出。

| 环节 | 模型 / 权重 | 代码许可证 | 权重许可证 | 训练数据 | 备注 |
|---|---|---|---|---|---|
| 分轨 | Demucs v4 `htdemucs`（adefossez/demucs，PyPI `demucs` ≥ 4.1） | MIT | **未声明；按非商业对待。** 作者原话："The license of the weights is not clear: MusDB contains some data that were only usable for research purpose"，并表示无权替 Meta 或 MUSDB 版权方澄清 | MUSDB HQ + Meta 内部 800 首 | 第 1 步接入（`docs/plans/step-1.md` 决定 2）。权重首次运行时自动从 Hugging Face / Meta 文件服务器下载，不随代码分发。自部署文档须转述这一许可证状态；如发现许可证更干净且质量够的 4 轨模型，只换 `Separator` 适配器 |
| 节拍 / 强拍 | Beat This!（CPJKU/beat_this，PyPI `beat-this` ≥ 1.1，checkpoint `final0`） | MIT | **MIT** | 16 个公开数据集，GTZAN 整体留出 | 第 1 步接入（决定 3）。不用 DBN 后处理，因此不依赖 madmom |

| 和弦（E1） | BTC（jayg996/BTC-ISMIR19，`btc_model_large_voca.pt`，commit 2682317） | MIT（模型代码已随本仓库分发，见 `analysis/chords/models/btc_vendor/LICENSE`） | **未声明**：权重随 MIT 仓库分发，作者未单独说明 | Isophonics、Robbie Williams、USPop | 第 2 步接入。权重首次运行从作者仓库的固定 commit 下载到 `~/.cache/opencapo/`，校验 SHA-256 后再反序列化 |
| 和弦（E2） | lv-chordia（openmirlab/lv-chordia，Jiang 等 ISMIR 2019 的五网络集成） | MIT | **未单独声明**：五份权重随 MIT 仓库分发 | 论文在 Billboard / Isophonics / RWC 上报数，训练集待核 | 第 2 步接入；从 GitHub 安装（`analysis` 依赖组直接引用） |
| 调性 / 低音 / 色度（E3–E5） | 无权重（规则） | — | — | — | 第 2 步，纯代码 |

未接入但评估过的候选：madmom 的模型文件为 **CC BY-NC-SA 4.0**（README 明示），只可作评测基线，不打包；all-in-one 权重许可证未声明；BS-RoFormer 社区权重许可证各异、部分为 CC-BY-NC-SA；crema（BSD-2）只能在 Keras 2 上运行，与 Python 3.12+ 的 TensorFlow 不兼容，第 2 步按环境原因淘汰；accordoai 为本科论文级实验、只用 Beatles 训练，未进入对比。

## 3. 编配

| 项 | 做法 |
|---|---|
| 演唱调 | `演唱调 = 原曲调 + 用户偏移（半音）`；默认偏移 0 |
| 指法调 → 变调夹 | `capo = 演唱调 − 指法调`（0–11，负值不可选）；候选指法调固定为 C / G / D / A / E / F / Am / Em / Dm 等常用调，按品数升序 |
| 页眉 | 原调、演唱调、指法调、capo 四项都输出；简谱 `1=` 写演唱调；和弦名按指法调 |
| 和弦 → 指法 | 指法库（JSON）：每个和弦名若干候选按法，带难度与把位；默认选难度最低、与前一个和弦把位最近的 |
| 节奏型库 | JSON：名称、拍号、每拍的扫 / 拨弦模式、推荐 BPM 区间、风格标签；默认按拍号 + BPM 选 |
| 简化和弦 | 规则映射：maj7/7/6/sus → 大三；min7/m6 → 小三；X/Y → X |
| 简谱 | `Note` 的 MIDI 音高 − 调式主音 → 音级 1–7 与八度点；时值按拍网格量化后映射到下划线 / 附点 / 连音线 |

## 4. 渲染

| 层 | 做法 |
|---|---|
| 和弦图 | 自绘 SVG（指法库直接给出按法，绘制简单） |
| 六线谱（节奏型 / 和弦变化点） | 自绘 SVG，或 alphaTab / Verovio 渲染 MusicXML 的 tab 部分 |
| 简谱 | **自绘 SVG**（主流库不支持简谱） |
| 歌词 | 自绘，音节对齐到简谱音符的 x 坐标 |
| 页面 | SVG → PNG（第一页预览）/ PDF（多页） |
| MusicXML | music21 生成；含和弦符号、六线谱、歌词音节 |

排版标准：`docs/reference/desperado-sheet.webp`。

## 5. 存储

| 数据 | 保留 | 存放 |
|---|---|---|
| 原音频、分轨 | 处理完成即删 | 临时盘 |
| 分析结果 JSON | 30 天 **[待定]** | 对象存储 / 数据库 |
| 当前调整参数 + 用户粘贴的歌词 | 同上 | 数据库 |
| 渲染产物（PNG / PDF） | 可随时重算，缓存即可 | 对象存储，带过期 |
| 用户配额计数 | 按天 | Redis / 数据库 |

## 6. 运行与部署

| 项 | 做法 |
|---|---|
| 硬件 | 租用云 GPU 服务器；**[建议]** 起步单卡 24 GB 显存级别，单 GPU 串行跑任务 |
| 处理时间预算 | 4 分钟歌曲：分轨 ~30–60 s、节拍 + 和弦 + 调性 ~10–20 s、旋律 ~10 s、歌词识别 + 对齐 ~30–60 s、渲染 ~5 s；合计目标 < 5 分钟，含排队 |
| 部署 | Docker Compose：bot、api、worker、redis、db、对象存储（或本地盘）；一键自部署 |
| 模型权重 | 首次启动下载并缓存；镜像不内置权重（体积与许可证） |
| 可观测性 | 每任务各环节耗时、失败原因、队列长度；日志 + 指标 |

## 7. 开放问题

1. 中文歌词强制对齐模型的选择与效果评估。
2. 人声音符切分 / 量化算法（滑音、颤音、装饰音的处理）；是否引入现成的歌声转录模型。
3. 和弦模型的最终选型：按 `step-2.md` 的多专家方案，在 GuitarSet 与 AAM 上实测各候选单独与融合后的准确率与推理速度。
4. 简谱排版规则细节（换行、连音线跨行、高低八度点）。
5. 指法库与节奏型库的初始内容从哪来（手工整理 vs 从开源谱库提取）。
