# 基准测试结果

这个目录存放验收数字，全部由 `scripts/` 里的脚本生成；手工不改数字。音频与标注文件**不提交**进仓库。

| 文件 | 生成脚本 | 对应验收项 |
|---|---|---|
| `step-1-beats.md` / `.json` | `scripts/eval_beats.py` | 节拍 F-measure ≥ 85%（GTZAN） |
| `step-1-timing.md` / `.json` | `scripts/time_pipeline.py` | 4 分钟歌曲分轨 + 节拍 ≤ 90 秒（单张 GPU） |
| `step-2-chords.md` / `.json` | `scripts/eval_chords.py` | 和弦 majmin ≥ 80%、调性 ≥ 85%（GuitarSet 报数半与 AAM 500 首） |
| `step-2-timing.md` / `.json` | `scripts/time_pipeline.py --out docs/benchmarks/step-2-timing` | 4 分钟歌曲和弦 + 调性 ≤ 30 秒 |

## 第 1 步：准备 GTZAN

1. **标注**（拍点 + 强拍，MIT）：

   ```bash
   git clone --depth 1 https://github.com/CPJKU/beat_this_annotations /data/beat_this_annotations
   # 用到的目录：/data/beat_this_annotations/gtzan/annotations/beats/  （gtzan_<流派>_<编号>.beats）
   ```

   文件格式：每行 `时间(秒)<TAB>拍序号`，拍序号为 1 的是强拍。来源是 Marchand & Peeters 2015 的 GTZAN-Rhythm 标注（经 Böck 整理）。

2. **音频**（1000 首 × 30 秒，10 个流派；原始服务器已下线，用镜像）：
   - Kaggle "GTZAN Dataset - Music Genre Classification"（约 1.2 GB），解压后用 `Data/genres_original/`；
   - 或 Hugging Face `marsyas/gtzan`。

   脚本按 `<流派>.<编号>.wav`（或原版的 `.au`）递归匹配，所以放在任意子目录都行。已知 `jazz.00054` 损坏，会被报出并跳过，实际曲目数写在结果里。

3. **安装与运行**（在 GPU 服务器上）：

   ```bash
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128   # 按服务器的 CUDA 版本选，见 pytorch.org
   pip install -e ".[analysis]" -r scripts/requirements-eval.txt
   python scripts/eval_beats.py --audio /data/gtzan --annotations /data/beat_this_annotations/gtzan/annotations/beats \
       --inputs mix,drums_bass --device cuda
   python scripts/time_pipeline.py /data/some-4-minute-song.mp3 --device cuda
   ```

   先加 `--limit 5` 做一次干跑确认目录对了。两个脚本都把结果写到本目录，提交即可。

## 第 2 步：准备 GuitarSet 与 AAM

两个数据集都在 Zenodo，用它的 API 列文件再下载（`curl -s https://zenodo.org/api/records/<id>` 的 `files` 字段）。

1. **GuitarSet**（Zenodo record 3371780；360 段约 30 秒的原声吉他录音，谱面 / 实奏和弦、调性、拍点）：下载 `annotation.zip` 与 `audio_mono-mic.zip`，解压到 `/data/guitarset/`，得到 `annotation/*.jams` 与 `audio_mono-mic/*_mic.wav`。脚本默认用谱面和弦（lead sheet）与麦克风录音；吉他手 00–02 调参、03–05 报数。
2. **AAM**（Zenodo record 5794629，CC BY 4.0；3000 首合成多轨）：下载标注包 `0001-1000-annotations-v1.1.0.zip`（8 MB）和混音包 `0001-1000-audio-mixes.zip`（14.7 GB；混音按 1000 首一包切，没有更小的单位，所以前 500 首也得下整包），解压到 `/data/aam/`。每首只有三个 ARFF：`<id>_beatinfo.arff`（每个和弦起点一行：起始秒、小节数、拍数、`Chord name`，**不是**每拍一行；这个文件没有 `@DATA` 行，读取器已兼容）、`<id>_segments.arff`（每段一行：起始秒、段落标记、`Tempo`、`Key`，最后一行 `end`）、`<id>_onsets.arff`（不用）。脚本按编号取前 500 首；和弦取 beatinfo 的行并合并相邻同名；调性取各段里累计时长最长的 `Key`（有的曲子中途转调）；拍网格按各段速度合成、用小节 / 拍数定强拍；音频是 `/data/aam/` 下任何以编号开头的音频文件（名字含 `mix` 者优先）。
3. 运行：

   ```bash
   pip install -e ".[analysis]" -r scripts/requirements-eval.txt     # 见下面关于 torch 的提醒
   python scripts/eval_chords.py --guitarset /data/guitarset --aam /data/aam --inspect      # 先核对格式
   python scripts/eval_chords.py --guitarset /data/guitarset --aam /data/aam --limit 3 --no-tune --out /tmp/dryrun
   python scripts/eval_chords.py --guitarset /data/guitarset --aam /data/aam --device cuda --cache /data/eval-cache
   python scripts/time_pipeline.py /data/song-4min.mp3 --device cuda --out docs/benchmarks/step-2-timing
   ```

   `--cache` 把各专家的逐拍输出存下来，调参和复跑不用重算模型。

4. **曲库合成评测（不需要任何录音）**：谱子可以从 PDF / 图片自动识谱（`scripts/sheet_ocr.py read`，用 Claude Code 的无头模式读图，走自己的登录；每小节读两遍、第三遍裁决，输出谱子文本与待复核清单），也可以手工录成文本（格式见 `docs/examples/xiaoxingxing.sheet`：头部是 `1=` 调、变调夹、速度、拍号，正文每行一小节，左边和弦指法、右边简谱）。再用 `scripts/sheet_synth.py` 渲染成音频加带时间戳的参考（按谱上指法扫弦的吉他 + 钢琴旋律，全部由代码合成），喂给评测：

   ```bash
   python scripts/sheet_ocr.py read my-songs/*.pdf --out /data/sheets --render   # 识谱 + 渲染，一条命令
   python scripts/sheet_synth.py my-songs/*.sheet --out /data/sheets             # 或：手工录入的谱子只渲染
   python scripts/eval_chords.py --sheets /data/sheets --device cuda --cache /data/eval-cache --out /tmp/sheets-report
   python scripts/analyze_chords.py --sheets /data/sheets --cache /data/eval-cache --subset sheets
   ```

   报表里这一组标"曲库合成（参考）"，不设门槛：合成音频和弦干净、没有人声，分数是真录音的上限而非估计。你录入的歌谱和渲染结果都不要提交，仓库只保留公有领域的示例曲。

   **torch 提醒**：lv-chordia 的依赖声明是 `torch>=2.13`，在只能用旧 CUDA 驱动（因而只能装旧 torch）的机器上，`pip install -e ".[analysis]"` 会把 torch 换成不匹配驱动的新版本。先装好与驱动匹配的 torch / torchaudio，再用约束文件钉住它们、lv-chordia 用 `--no-deps` 装并手动补其纯 Python 依赖：

   ```bash
   printf 'torch==%s\ntorchaudio==%s\n' "$(pip show torch | awk '/^Version/{print $2}')" "$(pip show torchaudio | awk '/^Version/{print $2}')" > /tmp/torch-pin.txt
   pip install -c /tmp/torch-pin.txt -e ".[analysis]" -r scripts/requirements-eval.txt   # 若在这里报 torch 冲突，改走下面两行
   pip install -c /tmp/torch-pin.txt --no-deps "lv-chordia @ git+https://github.com/openmirlab/lv-chordia.git"
   pip install -c /tmp/torch-pin.txt "librosa>=0.10" "pretty-midi>=0.2.9" "pydub>=0.23.1"
   ```

   pip 会打印一行 `lv-chordia requires torch>=2.13` 的警告，可忽略：实测 torch 2.5.1 能跑它的五个网络（`pytest -m models tests/test_chord_models.py`）。
