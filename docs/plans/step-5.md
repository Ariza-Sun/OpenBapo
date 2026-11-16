# 第 5 步实现计划：歌词（附歌词对齐 + 识别）

> 对应：`docs/ROADMAP.md` 第 5 步；需求 `docs/REQUIREMENTS.md` 5.4（歌词）、第 9 节验收；`docs/plans/backend.md` 第 2 节（歌词识别 / 对齐）、第 7 节开放问题 1。
> 状态：局部人声证据修复完成 77 条回归，已确认的 7 个幻听词不再输出，3 个有官方人声标注支持的词保留；本次质量门槛通过。过滤可能漏掉孤立短演唱，开发检查与局限见第 13 节（2026-11-16）。负责人决定见第 8 节。

## 1. 目标与范围

给定人声轨，产出音节级的歌词时间（`LyricSyllable`：起止时间、文字、行号），接到第 4 步的歌词行与 MusicXML。

**本步做**：

1. 歌词文本处理：把上传时附的歌词切成行与音节。中文一字一音；英文用 pyphen 断字位置切分显示音节；中英混排的行逐词处理；去掉标点、空行与段落标记（`[Chorus]`、`副歌：` 之类）。
2. 人声活动检测：在人声轨上找出有人声的区间；整首没有人声时返回"无歌词"（`lyrics` 为空，附一条说明），不报错。
3. 识别：faster-whisper 只在有人声的区间上识别，出带行结构的文本；中文统一转简体。
4. 对齐：歌词文本 + 人声轨 → 每个音节的起止时间。中、英各选一个对齐器（选法见第 8 节决定 2）。
5. 流水线接入：`openbapo analyze` 增加 `--lyrics FILE`（提供歌词时跳过识别、只对齐）；`Analysis.lyrics` 写入结果。
6. 渲染接入：SVG 歌词行按音节起始时间摆到对应拍的 x 坐标；MusicXML 里歌词按行写成文字标注（见 2.6）。
7. 测试集与评测脚本，产出验收数字。

**本步不做**：出谱后修改歌词 / 重新对齐（第 8 节决定 3）；音节挂到音符（第 7 步）；联网抓歌词（需求明确不做）；段落结构检测；逐字编辑歌词时间。

## 2. 设计

### 2.1 模块

```
src/openbapo/lyrics/            # 纯文本，无模型依赖
    text.py                     # 歌词 → 行 → 词 → 音节；语言按字符集判断
    syllables.py                # 英文音节切分（pyphen）
src/openbapo/analysis/lyrics/   # 模型推理，进 analysis 可选依赖
    activity.py                 # 人声活动检测
    recognize.py                # faster-whisper 识别
    align.py                    # 对齐器接口 + 各候选实现
    stage.py                    # 流水线环节：有歌词 → 对齐；无歌词 → 识别 → 对齐；无人声 → 空
```

核心库与模型分开（CLAUDE.md 工程约定）：文本切分可单测、不装 torch。

### 2.2 音节与行

- 中文：每个汉字一个音节；行内的英文单词按英文规则切。
- 英文：用 pyphen（MPL 1.1 许可选项）的断字位置把单词拆成音节；同一单词的非末音节带连字符（`beau-ti-ful`）。不用 CMUdict：音节只用于显示，时间在单词内按字母数等分，排版断字已够；常见 Python 封装为 GPL，也不宜引入。短词 pyphen 不拆（`awake` 整体显示），可以接受。
- 对齐器在"字 / 词"粒度上出时间；英文单词内的音节时间按字母数在单词时长内等分（MusicXML 和渲染只需要起始时间，误差限在一个单词内）。
- 行号沿用所附歌词的分行；识别结果的分行按 faster-whisper 的分段，再在长停顿处切开。

### 2.3 人声活动与"无歌词"

- 两种模式分开：自动识别仅用 Silero VAD（MIT，阈值 0.5），累计检测时长低于 5 秒 → 无歌词；用户附歌词时才允许能量补检。`LyricsStage` 的 `alignment_activity` 参数只影响附歌词分支。
- 附歌词时，累计语音证据达到 1 秒后，用人声轨持续能量补回漏检的演唱：50 ms 音量帧，阈值为该曲第 95 百分位 RMS 的 −15 dB，且不低于 −50 dBFS；连过 ≤ 0.3 秒短空隙，保留 ≥ 0.25 秒片段并向两端各扩 0.1 秒。合并区间低于 5 秒仍返回无歌词；没有语音证据时不补检。
- 参数在独立开发集上确定，报告见 `docs/benchmarks/step-5-activity-real-dev.md`；冻结记录见 `docs/benchmarks/step-5-retest-protocol.json`。
- 识别只喂有人声的区间，并设 `condition_on_previous_text=False`、按 `no_speech_prob` 过滤，抑制 Whisper 在纯伴奏上的幻听。

### 2.4 识别

- 基线：faster-whisper，Whisper `large-v3`（MIT 权重）。语言默认自动检测；中文输出用 OpenCC 规则转简体。
- 中文可加一个对照模型（如 FireRedASR，Apache 2.0），只出对照数字；主线是否更换在评测后再议。

### 2.5 对齐候选

| 候选 | 做法 | 许可证 |
|---|---|---|
| A. wav2vec2 CTC 强制对齐 | 英文用 torchaudio 的 LibriSpeech 模型，中文用 XLSR-53 中文微调模型（WhisperX 的默认组合）；`torchaudio.functional.forced_align` | MIT / Apache 2.0 |
| B. Whisper 对齐（stable-ts） | 把已知文本喂给 Whisper，用交叉注意力定词时间；中英同一模型 | MIT |
| C. MMS 强制对齐 | 多语言 CTC，中文先转拼音 | **CC BY-NC 4.0** |

长歌处理：CTC 对整首一次性做强制对齐（行序天然单调），声学模型在完整人声轨上按 ≤ 30 秒分块推理。人声活动区间两侧扩 0.3 秒后，区间外的非空白标签施加 2.0 的有限对数惩罚并重新归一化；允许强声学证据覆盖 VAD 漏检，不能把未推理的帧硬设为空白。模型词表拼不出的字（如生僻字）不计时，由前后字插值并记警告。

### 2.6 渲染

- SVG：已有 `lyric_bar`，补：音节按起始时间落到拍网格的 x；同一小节内音节过密时缩小字号（最小 5 pt），并依次右推，不重叠。不做低置信灰显：`LyricSyllable` 没有置信度字段，为此改数据模型不值得。
- MusicXML：歌词要挂在音符上，而旋律第 7 步才有。本步把每行歌词写成该行起始拍上的文字标注（`direction/words`），MuseScore 里可见可改；第 7 步再改成音符下的音节歌词。

## 3. 测试集（决定 1）

数据音频与标注都不进仓库；仓库只放清单生成脚本、评测脚本和结果。歌名在提交的结果里匿名（沿用第 2 步决定 18）。

| 用途 | 数据 | 内容 | 行起始时间的来源 |
|---|---|---|---|
| 英文验收 | JamendoLyrics（`en`，20 首） | 真实成品歌，CC 授权音频 | 数据集的人工行级标注 |
| 中文验收 | GTSinger 中文正常唱法组，10 首不同的歌 | 专业歌手干声；数据按乐句切成小段，按序拼回整首（段间随机插 0.5–2.5 秒） | 人工逐字时间。一行 = 两次停顿之间的字：每个小段开头、以及 ≥ 0.4 秒的换气 / 静音标记处断行；不足 3 个字的行并入前一行 |
| 中文验收（加伴奏版） | 同上，人声与 AAM 器乐曲等响度（人声 / 伴奏 0 dB）混合，前加 8 秒前奏、后加 4 秒尾奏 | 合成的带伴奏版本，跑完整的"分轨 → 对齐"；段间空档是纯伴奏 | 同上 |
| 中文参考（不设门槛） | MIR-1K 中带人工逐字时间的 17 段 | 业余演唱 + 真实卡拉 OK 伴奏（左右声道相加成混音） | 人工逐字时间；相邻两字间隔 ≥ 0.4 秒处断行，其余同上 |
| 纯音乐 | AAM、GTZAN 古典 / 爵士（验收 20 首，开发 10 首，互不重叠） | 无人声 | — |
| 开发集（选模型、定阈值） | GTSinger 未进验收集的中文 10 首、英文 10 首 | 干声 + 加伴奏版 | 同中文验收 |

## 4. 验收数字

### 4.1 指标

- **行级对齐**：每行起始时间与标注的绝对误差 ≤ 0.5 秒的比例（中、英分开报，门槛 ≥ 90%）；另报中位误差。
- **识别**：中文字错率（CER，先转简体、去标点）、英文词错率（WER，小写、去标点，参考文本用 Jam-ALT 的规范化歌词）；不设门槛。
- **幻听**：纯伴奏中凭空识别出的字 / 词数，目标为 0。歌词时间戳间 ≥ 3 秒的空隙（两端各缩 0.5 秒）只用于检出待核验候选，不能证明该处没有人声；需结合官方修订人声标注或与音频校验值绑定的人工听辨。原始候选计数始终保留；另报纯音乐曲目返回"无歌词"的比例（目标 100%）。
- **耗时**：报告一首 4 分钟歌曲歌词环节（识别 + 对齐）的耗时，计入整体 5 分钟预算。

### 4.2 选模型不碰验收集

对齐器、VAD 阈值、识别参数都只在开发集上比较和调整；定下后在验收集上跑一次，出最终数字。比较过程与结果写进 `docs/benchmarks/step-5-lyrics.md`。

### 4.3 脚本与结果

- `scripts/lyrics_testset.py`：从本地数据生成测试清单（拼接、混音、行起始时间），输出到数据目录，不进仓库。
- `scripts/eval_lyrics.py`：跑识别与对齐、算全部指标。历史回归原始结果保存在 `docs/benchmarks/step-5-lyrics-retest.{md,json}`；`lyrics_retest_report.py` 汇总首轮、历史候选及回归结果到 `step-5-lyrics.{md,json}`。
- GPU 评测在负责人的 GPU 机器上跑。

## 5. 依赖

`analysis` 可选组新增：`faster-whisper`（MIT）、`stable-ts`（MIT）、`silero-vad`（MIT）、`opencc-python-reimplemented`（Apache 2.0，简繁转换）、`transformers`（Apache 2.0，中文 wav2vec2）、`pypinyin`（MIT，MMS 候选用）；对齐器选定后删掉用不到的。核心组新增 `pyphen`。评测另需 `jiwer`（`scripts/requirements-eval.txt`）。每个模型权重在 `backend.md` 2.1 节登记许可证。

## 6. 与前后步骤的接口

- 第 4 步：`ArrangedSheet.lyrics` 原样带过，渲染读它。
- 第 6 步（Discord）：`/transcribe` 的 `lyrics` 参数传入歌词；没有修改歌词的入口（决定 3）。
- 第 7 步：`LyricSyllable.note_ref` 由第 7 步填；本步保持为空。

## 7. 开放问题

1. 歌词里的重复记号（`×2`、"重复副歌"）：本步不展开，按原文对齐，对不上的部分记警告。
2. 中文字错率的规范化：数字（"1 2 3" 对 "一二三"）、语气词、重复字的处理写进评测脚本，并在结果里说明。

## 8. 负责人决定

### 1. 测试集：只用直接能下载的公开数据（2026-11-12）

- 负责人：第 5 步用开源数据集；需要申请的数据集一律不用。
- 英文验收：JamendoLyrics `en` 20 首（真实成品歌，人工行 / 词时间，Hugging Face 直接下载）。
- 中文验收：GTSinger 中文正常唱法组（专业干声，CC BY-NC-SA 4.0，Hugging Face 官方仓库直接下载），出干声版与混入纯器乐的加伴奏版；结果标明"干声 / 合成伴奏"。
- 中文参考：MIR-1K 的 17 段（真实伴奏，业余演唱），只作参考，不设门槛。MIR-1K 原站已下线，用 Hugging Face 上的原始压缩包转存；逐字时间标注来自 GitHub 公开仓库 LyricAlignment。
- 不用：Opencpop、M4Singer、PopCS（需申请）；MIR-MLPop、DALI（音频要从 YouTube 抓取商业录音）。
- 风险：中文没有"真实成品歌 + 专业歌手"的验收数据；报告里干声、合成伴奏、真实伴奏三类数字分开写。

### 2. 模型：识别用 faster-whisper；对齐器按开发集实测选（2026-11-12）

- 识别基线 faster-whisper `large-v3`（路线图既定），中文可加对照模型。
- 对齐器在 2.5 节三个候选中按开发集上的行级 ≤ 0.5 秒比例选，中、英可以不同；选型细节由 agent 定，对比表与理由写进评测报告。
- 例外：若最优候选的权重是非商用许可（如 MMS），先问负责人再用。

### 3. 删除"出谱后粘贴歌词、重新对齐"（2026-11-12）

- 起因：需求规定原音频与分轨处理完即删，事后重新对齐没有人声可用（可选的做法是存对齐模型的逐帧输出、保留人声轨 30 天，或只做文本比对）。
- 负责人决定：删除这个功能。歌词只能在上传时附上（有则只对齐，无则识别）；出谱后不能修改歌词。
- 已同步修改：`REQUIREMENTS.md`（所有提到出谱后粘贴歌词的地方；第 10 节新增一行）、`ROADMAP.md`（第 0、5、6 步）、`backend.md`、`frontend-discord.md`（去掉歌词弹窗与重新对齐接口）、`CLAUDE.md`；数据模型 `Arrangement` 去掉 `lyrics_text` 字段。
- 验收里"重新对齐一首歌 ≤ 30 秒"随之删除。

## 9. 交付顺序

1. 本计划。
2. 数据下载与 `scripts/lyrics_testset.py`（清单、拼接、混音）。
3. `lyrics/text.py`、`syllables.py` + 单测。
4. 人声活动、识别、对齐候选；开发集比较，选定对齐器。
5. 流水线与 CLI 接入、渲染接入。
6. 验收集评测、计时、文档、README；问负责人是否开 PR。

## 10. 验收对照

| 验收项（ROADMAP 第 5 步） | 如何证明 | 状态 |
|---|---|---|
| 小测试集：中、英各 ≥ 5 首，人工标注行起始时间 | 第 3 节；`scripts/lyrics_testset.py` | 已准备：英文 20 首，中文 10 首各两版本，另有 MIR-1K 17 段参考 |
| 附歌词对齐：行级误差 ≤ 0.5 秒的比例 ≥ 90% | `docs/benchmarks/step-5-lyrics.md` | 回归通过：英文 94.6%（首轮 82.5%）；中文干声 98.1%、合成伴奏 97.6%；MIR-1K 参考 96.9% |
| 识别：中、英各自的字错率 / 词错率；纯乐器段落不出现幻听文本 | 同上；`step-5-local-regression.md` | 本次回归通过：英文 WER 41.5%，确认的 7 词已移除，剩余 3 词有官方人声标注支持；中文 CER 干声 12.1%、混音 19.1%。孤立短人声有漏检代价，见第 13 节 |
| 无人声的歌曲正确返回"无歌词" | 同上；单测 | 回归通过：纯音乐 20/20；全部 77 个评测条目完成 |
| 歌词耗时计入 4 分钟歌曲预算 | `docs/benchmarks/step-5-timing-local.md` | RTX 3090，3 次热运行中位：歌词 17.6 秒，分析合计 30.4 秒；不含排队、渲染及尚未实现的旋律，也不含 Whisper 对象加载 |

## 11. 2026-11-15 接手后的修复与复测协议

负责人要求接手原 agent 的工作，优先修复谱面；谱面确认后继续第 5 步。采用独立开发数据修复的方案，保留首轮结果，避免用英文验收曲目调参。

- `step-5-lyrics-first-run.{json,md}` 原样保留首轮文件；后续指标与其并列报告。
- `lyrics_activity_dev.py` 用固定版本的 JamendoLyrics 德语、西语、法语 59 首作为人声活动开发集，与英文验收集按歌曲名称检查无交集。只用于语言无关的人声检测，不宣称产品新增这些语言的歌词支持。
- 实验对比低 VAD 阈值与“先有语音证据，再用持续能量补检”的方案，同时检验原 GTSinger 开发集和 10 首开发纯音乐。分轨只驻留内存；可复用的开发缓存仅含音量曲线、区间与原文件校验值。
- 对齐器对整段人声推理；VAD 之外只给非空白标签有限惩罚，不再把模型尚未听过的区间强制设为空白。`lyrics_alignment_dev.py` 在原开发集对比惩罚 0、2、5：0 与 2 四组达标率相同，5 降低中文表现，采用 2。原开发集的新结果：英文干声 96.8%、合成伴奏 95.5%；中文干声 96.5%、合成伴奏 96.9%。中文干声较首轮开发值 98.2% 有回落，保留这一差异。
- 修正评测：走实际 `LyricsStage` 的无人声判断；缺失行计失败，不能从分母删去；中文 CER 直接按规范化后的汉字串计算，不能给每个字符插空格稀释错误。首轮中文 CER 因口径有误，不直接用于比较改进幅度。
- 参数在开发数据上确定后，冻结候选并跑原验收集；结果完整保留。脚本的中断续跑仅复用代码、输入、参数签名完全相同的逐首结果，避免因中断重复挑选结果。模型性能由独立计时脚本测量。
- 已选能量补检参数：相对 −15 dB、绝对下限 −50 dBFS、触发证据 1 秒。59 首真实开发歌曲的逐首平均召回率从 93.53% 到 98.82%，精确率从 87.83% 到 85.60%；原干声 / 合成伴奏开发版本召回率为 99.80% / 99.67%，开发纯音乐 10/10 无歌词。更低能量阈值损失更多精确率；Silero 阈值降到 0.1 则在开发纯音乐出现 1 首误报。1 / 5 秒触发证据及 −50 / −60 dB 下限在开发集上打平；采用较短触发证据帮助恢复漏检，保留更高噪声下限及最终 5 秒总量门槛。
- 当时按旧指标把能量候选的英文未标注区间词由 10 增至 120 解释为幻听回退，故搁置了该候选。第 12 节核查发现这一推断不成立：漏标人声也会计入，不能仅据此淘汰方案。`step-5-lyrics-energy-candidate.{json,md}` 和 `step-5-energy-candidate-protocol.json` 原样保留，避免改写历史结果。
- 最终分离两种模式：附歌词保留独立开发集选出的补检与软惩罚；自动识别恢复原来的语音检测区间，不改识别阈值。随后使用同一验收集做回归验证，明确它不是新盲测，也不把三次测量描述为一次。最终结果由 `lyrics_retest_report.py` 从原始文件生成。
- 回归完成：77/77 条目，英文对齐 94.6%，中文干声 98.1%、合成伴奏 97.6%，MIR-1K 参考 96.9%；纯音乐 20/20。自动识别的未标注区间词恢复至 10 词，但不能等同已确认幻听，后续核查见第 12 节。此时核心回归 623 项通过（7 项需模型的测试默认排除，模型行为由真实数据评测另验），ruff 与 mypy 通过。

### 复现

在装好 `.[analysis,dev,render]` 和 `scripts/requirements-eval.txt` 的环境中运行。`DATA_ROOT` 指向本地歌词数据目录；音频、模型缓存与开发特征缓存均留在仓库外。

```bash
export DATA_ROOT=/path/to/lyrics
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

python scripts/lyrics_activity_dev.py \
  --jamendo "$DATA_ROOT/jamendolyrics" \
  --manifest "$DATA_ROOT/testset/manifest.json" \
  --cache "$DATA_ROOT/activity-dev-features" \
  --out docs/benchmarks/step-5-activity-real-dev

python scripts/lyrics_alignment_dev.py \
  --manifest "$DATA_ROOT/testset/manifest.json" \
  --cache "$DATA_ROOT/activity-dev-features" \
  --out docs/benchmarks/step-5-alignment-soft-dev

python scripts/eval_lyrics.py \
  --manifest "$DATA_ROOT/testset/manifest.json" --split test \
  --aligners wav2vec2 --seed 0 --out "$DATA_ROOT/lyrics-new-run"

python scripts/lyrics_retest_report.py --gap-audit docs/benchmarks/step-5-gap-audit.json
```

首次准备真实人声开发集时，给 `lyrics_activity_dev.py` 加 `--download` 获取脚本固定版本的公开音频。首轮、候选与已冻结的回归结果都是历史记录，不用当前代码覆盖；汇总脚本会校验原始结果与冻结协议。中断后只可在实现、数据、参数完全相同的情况下加 `--resume` 接续逐首检查点。当前评测协议已更新为第 4 版，新输出写到独立路径。

后续自动识别的重点是歌曲内部间奏的幻听，而非仅检查整首纯音乐；需要独立的中英文真实演唱开发样本同时验证漏词和幻听，再评估新参数。不能继续根据这 20 首英文验收歌逐首调整过滤条件。

## 12. 2026-11-16：未标注人声审计与复现修正

此前把英文 10 个未标注区间词全部称为幻听，依据不足。官方 [Jam-ALT 修订标注](https://audioshake.github.io/jam-alt/)包含背景人声及非词语发声；旧 JamendoLyrics 行间空隙只能用来发现候选，不能证明纯伴奏。本轮不按验收曲目删除词、不添加黑名单、不改变生产识别参数。

- 当前默认配置的 10 个候选已完整复现。`en-test-04` 的 2 词和 `en-test-19` 的 1 词落在官方修订人声区间，不能计作纯伴奏幻听；这也不等于识别拼写完全正确。
- `en-test-07` 的 7 词集中在 181.80–182.76 秒。独立 wav2vec2 无提示贪心解码在人声轨上给出相近的说话内容，但原混音的结果不清楚。当时未获得人工结论，保留为待核验。负责人随后明确回复“没有说话声”，该项现已记录为人工确认幻听，见第 13 节。记录保存在 `step-5-gap-review.json`，绑定音频 SHA-256 和审核人；模型一致不能自动转为人工确认。
- 历史能量候选的 120 词不能直接作为失败依据。涉及的 5 首歌曲本次重新解码得到 103 词，其中 80 词的时间点有官方人声标注覆盖，23 词仍待核验。重复结果有差异，未用 103 覆盖历史 120，也未把当前分类套到历史输出上。详见 [审计报告](../benchmarks/step-5-gap-audit.md)。
- 独立开发集 50 条（中英文各 10 首、各两版本，另加纯音乐 10 首）比较显示：英文干声 / 混音 WER 从 15.6% / 17.0% 到 13.2% / 15.3%；中文干声 / 混音 CER 从 11.2% / 18.4% 到 9.5% / 16.1%。两种配置均无未标注区间词，纯音乐均 10/10 无词。这是单次开发比较，不能替代真实歌曲的间奏听辨，也不保证跨次波动下每项都改善。详见 [开发报告](../benchmarks/step-5-asr-audit-dev.md)。
- 旧评测未固定温度回退抽样的随机种子。当前评测按基准种子与条目 ID 生成逐曲种子，并在设置种子后新建 Whisper 工作线程；只给已存在的线程重新设置全局种子不足以重置 CTranslate2 的线程内随机状态。基准种子纳入续跑签名，候选保留时间戳。此改动仅用于评测复现，不改变生产识别配置。
- 真实模型抽查：进程 A 顺序运行 `en-dev-01-mix`、`en-dev-02-mix`；进程 B 只运行第二首，模拟续跑跳过第一首。两次均用基准种子 0、能量补检 −15 dB；第二首的完整解码元数据一致，包括文本、时间戳、概率。仅能说明这首曲目在当前环境可复现，不保证所有曲目或跨硬件一致。详见 [抽查记录](../benchmarks/step-5-asr-seed-check.md)。
- 验证：629 项核心测试通过，7 项模型测试按默认配置排除；ruff 与 mypy（58 个源文件）通过。另用真实模型运行新版评测脚本：`en-dev-01` 干声及混音共 2 条、各 18 行，均完成识别与对齐，候选时间戳、逐曲种子和协议元数据正常输出。还验证了空筛选必须报错，避免误把空报告当成通过。

随机种子的实现依据：[CTranslate2 API](https://opennmt.net/CTranslate2/python/ctranslate2.set_random_seed.html)与[线程内随机状态源码](https://github.com/OpenNMT/CTranslate2/blob/master/src/random.cc)。本轮修改的是评测复现与证据分类，生产模型行为未改，因此沿用此前完整 77 条回归和流水线计时，不将两条冒烟检查另算一次全量验收。

### 复现审计

`DATA_ROOT` 沿用上节。原始解码与交叉检查的文本元数据在仓库外；分轨只驻留内存。以下命令中的开发比较保留旧配置未设随机种子的行为，新实验可加 `--seed 0` 并使用新的缓存目录。已有缓存不允许用不同参数复用。

```bash
python scripts/audit_lyrics_gaps.py \
  --manifest "$DATA_ROOT/testset/manifest.json" --split dev \
  --cache "$DATA_ROOT/asr-gap-audit" --out "$DATA_ROOT/asr-gap-audit/dev-diagnostic.json"

python scripts/audit_lyrics_gaps.py \
  --manifest "$DATA_ROOT/testset/manifest.json" --split dev --energy-relative-db -15 \
  --cache "$DATA_ROOT/asr-gap-audit-energy" \
  --compare "$DATA_ROOT/asr-gap-audit/dev-diagnostic.json" \
  --out docs/benchmarks/step-5-asr-audit-dev.json

python scripts/audit_lyrics_gaps.py \
  --manifest "$DATA_ROOT/testset/manifest.json" --split test \
  --ids en-test-04,en-test-07,en-test-19 --cache "$DATA_ROOT/asr-gap-audit" \
  --out "$DATA_ROOT/asr-gap-audit/test-diagnostic.json"

python scripts/lyrics_gap_evidence.py \
  --manifest "$DATA_ROOT/testset/manifest.json" \
  --audit "$DATA_ROOT/asr-gap-audit/test-diagnostic.json" \
  --jam-alt "$DATA_ROOT/jam-alt/metadata.jsonl" \
  --out "$DATA_ROOT/asr-gap-audit/crosscheck.json"

python scripts/lyrics_gap_review.py \
  --baseline docs/benchmarks/step-5-lyrics-retest.json \
  --diagnostic "$DATA_ROOT/asr-gap-audit/test-diagnostic.json" \
  --crosscheck "$DATA_ROOT/asr-gap-audit/crosscheck.json" \
  --development "$DATA_ROOT/asr-gap-audit/dev-diagnostic.json" \
  --reviews docs/benchmarks/step-5-gap-review.json \
  --candidate-baseline docs/benchmarks/step-5-lyrics-energy-candidate.json \
  --candidate-crosscheck "$DATA_ROOT/asr-gap-audit-energy/crosscheck.json" \
  --out docs/benchmarks/step-5-gap-audit

python scripts/lyrics_retest_report.py --gap-audit docs/benchmarks/step-5-gap-audit.json
```

能量候选的交叉检查用同一 `lyrics_gap_evidence.py`，输入 `asr-gap-audit-energy/test-diagnostic.json`；该输入由 `audit_lyrics_gaps.py --energy-relative-db -15` 对 `en-test-04,en-test-07,en-test-13,en-test-17,en-test-19` 生成。

## 13. 2026-11-16：人工确认后的局部证据修复

负责人听辨后明确回复“没有说话声”，因此原版本 `en-test-07` 的 7 词记录为确认幻听，而不是删去评测记录。原有全曲人声总量门槛允许孤立短检测借用其他演唱部分的证据：该检测仅约 1.6 秒，与前一段相隔约 50 秒，仍会进入 Whisper。

自动识别现在将 Silero 区间按至少 30 秒的间隔分组，每组沿用已有的至少 5 秒累计人声门槛；间隔本身及重叠 padding 不重复计数。采用的是通用时间规则，不按歌名、词语或参考歌词过滤。30 秒取现用 Whisper 声学窗口，5 秒取项目既有人声门槛；这是本项目的保守工程选择，并非上游推荐的通用防幻听参数。附歌词对齐不应用该过滤。

独立检查包含 59 首非英语真实歌曲、40 个中英文干声 / 合成伴奏版本和 10 首纯音乐：原 50 条识别开发集的缓存输入区间全部不变；59 首真实歌曲的人声帧召回率从 93.9746% 到 93.9196%，损失 3 首合计 3.649 秒已标注人声。该规则确实可能漏掉独立的短句、呼喊或演唱，不能宣称零漏词。见 [开发检查](../benchmarks/step-5-local-activity-dev.md)。

参数在开发检查后写入 [冻结协议](../benchmarks/step-5-local-protocol.json)，再对原 77 条验收 / 参考条目进行回归。这个已查看过的验收集不作为新盲测，也不将过滤造成的遗漏称为识别能力提升。

复现开发检查与新版回归（`DATA_ROOT` 同上；重跑时使用新的输出目录，避免覆盖已冻结结果）：

```bash
python scripts/lyrics_local_activity_dev.py \
  --jamendo "$DATA_ROOT/jamendolyrics" --manifest "$DATA_ROOT/testset/manifest.json" \
  --features "$DATA_ROOT/activity-dev-features" --asr-cache "$DATA_ROOT/asr-gap-audit" \
  --out "$DATA_ROOT/local-activity-dev-repeat"

python scripts/eval_lyrics.py --manifest "$DATA_ROOT/testset/manifest.json" \
  --split test --sets en-test --aligners wav2vec2 --seed 0 \
  --out "$DATA_ROOT/local-en-repeat"

python scripts/eval_lyrics.py --manifest "$DATA_ROOT/testset/manifest.json" \
  --split test --sets zh-test,zh-ref,instr --aligners wav2vec2 --seed 0 \
  --out "$DATA_ROOT/local-other-repeat"

python scripts/lyrics_local_report.py \
  --runs docs/benchmarks/step-5-local-en.json docs/benchmarks/step-5-local-other.json \
  --manifest "$DATA_ROOT/testset/manifest.json" \
  --protocol docs/benchmarks/step-5-local-protocol.json \
  --baseline docs/benchmarks/step-5-lyrics-retest.json \
  --references docs/benchmarks/step-5-gap-audit.json \
  --reviews docs/benchmarks/step-5-gap-review.json \
  --development docs/benchmarks/step-5-local-activity-dev.json \
  --timing docs/benchmarks/step-5-timing-local.json \
  --out docs/benchmarks/step-5-local-regression

python scripts/lyrics_retest_report.py \
  --gap-audit docs/benchmarks/step-5-gap-audit.json \
  --local-regression docs/benchmarks/step-5-local-regression.json
```

汇总脚本验证两个分片覆盖完整清单、没有重复条目，且代码、数据、参数和人工记录与冻结协议相符；新输出的所有未标注区间词逐个按证据重新分类。原 77 条回归及首轮、能量候选结果继续保留。

运行记录：中文 / 参考 / 纯音乐分片累计完成 43 条、51 条时，分别在初始化下一首 Whisper 时显存不足退出。按同一冻结签名和逐曲种子使用 `--resume` 两次接续，不丢弃或重选已完成条目。当前逐曲新建 Whisper 的长批次评测仍可能需要断点续跑；生产流水线只初始化一次该模型。两张卡均为 RTX 3090；独立计时使用另一张卡，期间同主机仍有评测任务，数值按实测负载解释。

最终结果：[77 条回归报告](../benchmarks/step-5-local-regression.md)。英文对齐 94.6%，中文干声 98.1%、混音 97.6%，纯音乐 20/20 无歌词；确认幻听 0，未决候选 0，另外 3 个官方人声标注支持的词完整保留。原版本与新版本的中英文总字 / 词错率按一位小数显示一致。637 项测试通过、7 项模型测试默认排除；实际模型行为由本次完整数据回归另验。ruff 与 mypy 通过。

[计时](../benchmarks/step-5-timing-local.md)：4 分钟原曲，三次热运行中位数为歌词 17.6 秒、分析合计 30.4 秒；首轮分析 42.4 秒。Whisper 对象初始化在该计时之外，因此首轮不是完整进程冷启动时间；不包含排队、渲染及未来旋律环节。新的计时在同主机另一张卡仍有评测任务时完成，不据此宣称相对上一轮加速或减速。

## 14. 收尾前补充：英文专用 Parakeet 对比

负责人要求再尝试一个英文语音识别模型。候选为 NVIDIA Parakeet TDT 0.6B v2，
使用 Ilya Stupakov 的未量化 ONNX 转换；原模型、转换权重的许可与固定修订见
[后端许可登记](backend.md#21-已接入的模型权重与许可证)。这次是英文识别实验，
不重做 Demucs 选型，也不改变中文识别或附歌词对齐。

脚本为 `scripts/compare_english_asr.py`，协议为
[`step-5-english-asr-protocol.json`](../benchmarks/step-5-english-asr-protocol.json)。
开发集为 10 首英文歌的干声 / 合成伴奏两个版本及 10 条纯音乐；
回归集为原 20 首真实英文歌及 20 条纯音乐。真实歌曲此前已查看过，不是新盲测。
两模型均使用固定逐曲种子、同一 Demucs / Silero 配置与局部人声过滤，指定英文，
不传入参考文本。开发样本从行标注取参考歌词，真实英文歌从 Jam-ALT 修订全文取参考。

Parakeet 按保留的 VAD 区间独立解码，超过 30 秒则不重叠切分；Whisper 沿用
`clip_timestamps`。候选原生 token 时间只用于空隙词检查，词尾估计为最后 token
发射时间加 80 ms，并截断到切片末尾；不将此当作强制对齐成绩。
空隙词沿用第 12–13 节的修订标注 / 人工听辨分类，未覆盖的词保留为待核验。

使用仓库外实验环境（`onnx-asr==0.12.0`、`onnxruntime-gpu==1.23.2`），未修改生产依赖。
两张 RTX 3090 分别运行基线与候选，CUDA 12 / cuDNN 9 动态库需完整可见。
脚本检查实际 ONNX session 使用 CUDA，拒绝静默 CPU 回退。模型文件按固定修订的
LFS SHA-256 校验。Whisper 每首新进程，避免旧批处理中的 GPU 分配累积；
模型下载期间可用 `--baseline-only` 预跑基线，之后重新分轨并逐字节校验人声哈希、
检测区间和协议签名后复用识别输出。分轨仍不落盘，只保存文本诊断在仓库外。

计时口径仅为识别调用，不含分轨、VAD、模型加载、CTC 对齐、渲染与排队；
Whisper 为 float16，Parakeet 为未量化 ONNX，且模型复用方式不同，不能当作同精度
架构吞吐率基准，也不能直接与第 13 节“识别 + 对齐”的歌词总耗时比较。

在安装原评测依赖与上述隔离运行时、下载固定模型文件后，从仓库根目录运行：

```bash
# MODEL_DIR 指向固定修订的 ONNX snapshot；DATA_ROOT 为原歌词数据目录。
python scripts/compare_english_asr.py \
  --manifest "$DATA_ROOT/testset/manifest.json" --model-dir "$MODEL_DIR" \
  --jam-alt "$DATA_ROOT/jam-alt/metadata.jsonl" \
  --reviews docs/benchmarks/step-5-gap-review.json \
  --protocol "$DATA_ROOT/english-asr-repeat-protocol.json" \
  --cache "$DATA_ROOT/english-asr-repeat" --split dev \
  --out "$DATA_ROOT/english-asr-repeat-dev"
# 保持协议、配置和缓存目录不变，再以 --split test 跑真实歌曲 / 纯音乐回归。
```

开发与回归报告分别写入 `docs/benchmarks/step-5-english-asr-dev.*` 和
`docs/benchmarks/step-5-english-asr-test.*`。更换默认模型的预先条件为真实英文 WER
降低、没有新增确认幻听 / 未决空隙词、纯音乐全部通过；不按验收输出调参或过滤词语。

### 本次结果与决定

完整比较共 70 条，两个模型均完成：

| 条件 | 数量 | Whisper large-v3 WER | Parakeet TDT 0.6B v2 WER |
|---|---:|---:|---:|
| 开发干声 | 10 | 15.6% | 19.3% |
| 开发合成伴奏 | 10 | 17.0% | 23.7% |
| 真实英文歌曲回归 | 20 | 41.5% | 36.3% |
| 纯音乐（开发 + 回归） | 30 | 30/30 无歌词 | 30/30 无歌词 |

详见 [开发报告](../benchmarks/step-5-english-asr-dev.md) 和
[真实歌曲 / 纯音乐报告](../benchmarks/step-5-english-asr-test.md)。
真实歌曲的 6,237 个参考词中，Whisper 为 566 替换 / 1,965 删除 / 57 插入，
Parakeet 为 858 替换 / 1,318 删除 / 87 插入：候选减少漏词，但替换与插入更多。
总 WER 低 5.2 个百分点；逐曲 8 首更好、11 首更差、1 首相同。
按歌曲配对 bootstrap 的 WER 差 95% 区间为 −13.2 至 +2.4 个百分点，包含零，
不能把这 20 首的平均优势推广成普遍结论。

Whisper 的 3 个空隙词均有修订人声标注支持；Parakeet 的 47 个空隙词中，
31 个有标注支持，**16 个待核验（4 首歌）**。另外开发混音有 1 个待核验词。
它们不是已经确认的幻听，原生时间戳误差与漏标人声均可能影响分类。
没有把它们直接删除、认定为真实人声或擅自补填人工结论。
纯音乐无输出由共用 VAD 门控保证，本次不能据此独立证明候选模型不会幻听。

真实歌曲识别调用中位耗时：Whisper 9.67 秒、Parakeet 7.26 秒。
该口径及精度 / 模型复用差异见上文；不是完整出谱耗时。
20 首 Whisper 的逐曲 WER 全部复现了此前 77 条回归里的英文结果。

**决定：保留 Whisper large-v3 为默认识别模型。** Parakeet 有减少真实歌曲漏词的潜力，
但开发集退化且仍有未决空隙词，未满足预先设定的更换条件。
本次保留实验脚本、固定权重修订、结果和核验时间点，生产依赖不增加 ONNX / Parakeet。
继续评估该候选时，应先听辨未决区间并在独立留出歌曲上验证；不能将待核验词数宣称为幻听率。

交付检查：642 项本地测试通过，ruff 和 mypy 通过。远程 CI 额外发现精简依赖环境与
Python 3.11 的可选库 / NumPy 类型声明差异，随后仅补充 `Any` / `cast` 注解与
`huggingface_hub` 的可选导入配置，未改变推理数值计算、模型或阈值。
实测签名保留实际实验版本；模型模块的数值实现对应提交 `b496367`，不覆写历史协议来伪装重测。
