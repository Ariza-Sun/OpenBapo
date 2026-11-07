# OpenCapo

[![CI](https://github.com/Ariza-Sun/OpenCapo/actions/workflows/ci.yml/badge.svg)](https://github.com/Ariza-Sun/OpenCapo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**听歌出弹唱谱的开源工具。** 上传一首歌，得到一张吉他弹唱谱：和弦图、六线谱节奏型、简谱旋律和歌词。演唱调、指法调、节奏型都可以换，导出 PNG / PDF / MusicXML。中文、英文歌曲都支持，最终以 Discord Bot 的形式提供。

> ⚠️ **开发中。** 目前可用的是命令行分析工具：从音频中识别节拍、BPM、和弦与调性，输出 JSON。编配、出谱和 Discord Bot 还在路上，进度见下文。

## 为什么做这个

"听歌扒谱"类产品（Capo、Chordify、Moises 等）都是闭源收费的；国内吉他谱平台的弹唱谱靠人工编配，冷门歌、英文歌常常没有谱。OpenCapo 想做一个任何人都能自己部署的开源替代，输出吉他弹唱者熟悉的简谱式弹唱谱。

## 进度

| 步 | 内容 | 状态 |
|---|---|---|
| 0 | 项目骨架、数据模型、命令行 | ✅ 完成 |
| 1 | 分轨、节拍、BPM、拍号 | ✅ 完成 |
| 2 | 和弦、调性 | ✅ 完成 |
| 3 | 编配：变调夹、指法调、和弦简化、节奏型 | ⏳ 下一步 |
| 4 | 渲染与导出：和弦图、六线谱、简谱，PNG / PDF / MusicXML | — |
| 5 | 歌词：粘贴对齐与识别 | — |
| 6 | Discord Bot | — |
| 7 | 人声旋律 → 简谱 | — |

完整计划见 [路线图](docs/ROADMAP.md)，需求见 [需求文档](docs/REQUIREMENTS.md)。

## 快速开始

需要 Python ≥ 3.11 和 `ffmpeg`。模型在 GPU 上运行较快，CPU 也能跑。

```bash
git clone https://github.com/Ariza-Sun/OpenCapo.git
cd OpenCapo
python -m venv .venv && source .venv/bin/activate

# 先按 pytorch.org 的说明装与你的 CUDA 匹配的 torch 和 torchaudio；纯 CPU：
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[analysis]"

opencapo analyze 你的歌.mp3 --out analysis.json
```

模型权重在首次运行时自动下载。手头没有音频的话，可以先用仓库自带的脚本合成一段 C–F–G7–Am 和弦进行试一下：

```bash
python scripts/make_tone.py demo.wav --progression 120 --seconds 16
opencapo analyze demo.wav --out demo.json
```

输出包含拍点与强拍、BPM、拍号、调性，以及落在拍上的和弦序列（根音用 0–11 表示 C–B，另有和弦性质、转位低音、置信度）。下面是 `demo.json` 的节选：

```json
{
  "beats": [{"time": 0.0, "is_downbeat": true, "bar_index": 0, "beat_in_bar": 1}, ...],
  "tempo": {"bpm": 120.0},
  "time_signature": {"numerator": 4, "denominator": 4},
  "key": {"tonic": 9, "mode": "minor", "confidence": 0.004},
  "chords": [
    {"start": 0.0, "end": 2.0, "root": 0, "quality": "maj", "bass": null, "confidence": 0.96},
    {"start": 2.0, "end": 4.0, "root": 5, "quality": "maj", "bass": null, "confidence": 0.92},
    ...
  ]
}
```

这段进行每轮都停在 Am，所以调性判成 A 小调（C 大调的关系小调），置信度很低，表示两者几乎打平。

常用选项：`--device cpu|cuda`、`--timing -` 打印各环节耗时、`--chord-experts lvchordia` 只用一个和弦模型。

## 工作原理

```
音频 ──解码──▶ 分轨（Demucs）──▶ 节拍 / 强拍 / BPM（Beat This!）
                    │
                    ▼
           和弦：BTC + lv-chordia + 色度模板 + 低音检测
           在拍网格上融合，Viterbi 平滑
                    │
                    ▼
           调性：和弦统计 + 色度模板互校
                    │
                    ▼
     编配 → 渲染 → Discord（开发中）
```

- **按拍决策。** 两个深度学习和弦模型和两个规则专家各自给出逐拍的后验，融合层在拍网格上加权投票，再用 Viterbi 平滑：强拍换和弦代价低、小节中间代价高。所以每个和弦的起止都落在拍上。
- **分轨只在内部用。** 分轨音频是中间产物，处理完即删除，不存储、不下发。
- **分析与编配分离。** 模型推理只跑一次，生成的 JSON 可以反复编配、出谱，后者是秒级的纯计算。

## 准确率

数字由仓库里的脚本生成，见 [docs/benchmarks](docs/benchmarks/README.md)。RTX 3090 上测得：

| 任务 | 数据 | 结果 |
|---|---|---|
| 节拍 | GTZAN（998 首） | 拍点 F-measure **0.891**，强拍 0.786 |
| 和弦（大小三和弦级别） | 220 首中文 / 英文流行歌的吉他谱，渲染成合成音频 | **0.974** |
| 和弦（大小三和弦级别） | AAM 合成全乐队（500 首） | **0.947** |
| 和弦（大小三和弦级别） | GuitarSet 原声吉他伴奏（90 段） | 0.775 |
| 调性 | 同上 220 首吉他谱（关系大小调算对） | **0.936** |
| 调性 | GuitarSet 伴奏 / AAM 单一调性曲目 | 0.789 / 0.931 |
| 速度 | 4 分钟歌曲，分轨 + 节拍 + 和弦 + 调性 | 13.4 秒 |

需要注意的两点：

- 吉他谱那组用的是从谱子渲染的合成音频，没有人声和真实混音，分数应看作真实录音的上限。
- GuitarSet 上偏低的部分集中在 Funk 和爵士伴奏，这是现有开源和弦模型的弱项。

## 项目结构

```
src/opencapo/
  model/      数据模型：Beat、Tempo、Key、Chord、Analysis，JSON 序列化
  analysis/   分析流水线：解码、分轨、节拍、和弦与调性（chords/）
  sheet/      谱子文本格式、渲染器、识谱：评测用的样本生成器
scripts/      评测、计时、识谱、样本渲染脚本
tests/        单元测试；标记为 models 的测试会加载真模型
docs/         需求、路线图、各步实现计划、评测结果
```

## 开发

```bash
pip install -e ".[dev]"
ruff format --check . && ruff check . && mypy && pytest   # 提交前全绿
pip install -e ".[dev,analysis]" && pytest -m models      # 真模型的端到端测试
```

- torch 与 torchaudio 要来自同一个构建。lv-chordia 声明依赖 `torch>=2.13`，在只能装旧版 torch 的机器上，pip 会把 torch 换掉，解决办法见 [docs/benchmarks/README.md](docs/benchmarks/README.md)。
- 设计决定和取舍记录在 [docs/plans/](docs/plans/) 各步的实现计划里。动手前先读对应那一步。

欢迎提 Issue 和 PR。文档用中文，代码、标识符和提交信息用英文。

## 模型与许可证

本仓库代码以 [MIT](LICENSE) 许可发布。用到的模型权重不随代码分发，首次运行时从各自来源下载：

| 用途 | 模型 | 权重许可证 |
|---|---|---|
| 分轨 | Demucs `htdemucs` | 未声明，**按非商业使用对待** |
| 节拍 | Beat This! | MIT |
| 和弦 | BTC | 未单独声明（随 MIT 代码仓库分发） |
| 和弦 | lv-chordia | 未单独声明（随 MIT 代码仓库分发） |

详细说明和训练数据见 [docs/plans/backend.md](docs/plans/backend.md) 第 2.1 节。商用或公开部署前，请自行核对各权重的许可证。OpenCapo 只输出谱子，不提供分轨或变调后的音频。
