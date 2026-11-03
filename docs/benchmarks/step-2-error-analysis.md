# 和弦 / 调性错误分析：report_comp

融合参数：`FusionParams(weights={'btc': 0.5, 'lvchordia': 0.5, 'chroma': 0.5}, key_prior=0.5, change_cost_downbeat=0.1, change_cost_beat=3.0, no_chord_bias=-1.0, foreign_bass_penalty=0.7, min_bass_strength=0.3, fill_gaps_under_bar=True)`

## 错误分解（90 首，按帧，0.05 s 一帧）

| 系统 | correct | ref_N | est_N | third | relative | fourth_fifth | other_root | 错误中在参考边界一拍内 |
|---|---|---|---|---|---|---|---|---|
| btc | 76.4% | 0.0% | 1.5% | 2.2% | 6.7% | 3.9% | 9.3% | 40% |
| lvchordia | 80.0% | 0.0% | 2.4% | 2.2% | 4.7% | 4.0% | 6.7% | 40% |
| chroma | 65.3% | 0.0% | 0.7% | 1.5% | 6.3% | 11.4% | 14.7% | 41% |
| committee | 79.0% | 0.0% | 2.0% | 2.4% | 5.6% | 4.1% | 7.0% | 42% |

任一模型专家正确（融合的上限）：84.5%

## 按风格（committee）

| 风格 | correct | ref_N | est_N | third | relative | fourth_fifth | other_root |
|---|---|---|---|---|---|---|---|
| BN | 86.7% | 0.0% | 0.0% | 2.3% | 3.1% | 3.3% | 4.6% |
| Fu | 48.9% | 0.0% | 8.0% | 2.5% | 15.4% | 9.3% | 15.9% |
| Ja | 64.8% | 0.0% | 1.4% | 1.2% | 10.0% | 7.9% | 14.6% |
| Ro | 91.9% | 0.0% | 0.2% | 5.7% | 0.3% | 1.0% | 1.0% |
| SS | 97.9% | 0.0% | 0.0% | 0.4% | 0.6% | 0.2% | 0.9% |

## 最差的 15 首（committee，majmin 帧正确率）

- `03_Jazz1-130-D_comp` 0.10
  - 参考：D:maj(7.4) G:maj(3.7) D:maj(3.7) A:maj(1.8) G:maj(1.8) D:maj(3.7)
  - 估计：A:min7(1.4) E:min(3.7) A:min7(1.8) E:min(3.7) F#:min7(1.4) A:7(1.4) E:min(1.8) D:maj(0.9) A:min(2.3) A:min7(0.5)
- `04_Funk2-119-G_comp` 0.11
  - 参考：C:min(2.0) F:7(2.0) A#:maj(2.0) D#:maj(2.0) A:hdim7(2.0) D:7(2.0) G:min(4.0) C:min(2.0) F:7(2.0) A#:maj(2.0) D#:maj(2.0) A:hdim7(2.0) D:7(2.0) G:min(4.0)
  - 估计：C:min7(1.0) D#:maj(1.0) G#:maj(2.0) D:min(2.0) G:min(2.0) C:min(2.0) F:maj(1.5) D:7(0.5) A#:maj7(2.0) G:min(2.0) G#:maj7(2.0) A:hdim7(2.0) G:min(4.0) C:min(2.0) D:min7(2.0) A#:maj7(2.0) A#:maj(2.0)
- `05_Jazz1-130-D_comp` 0.12
  - 参考：D:maj(7.4) G:maj(3.7) D:maj(3.7) A:maj(1.8) G:maj(1.8) D:maj(3.7)
  - 估计：B:min(2.8) G:7(1.8) B:min(3.7) G:7(1.4) C#:7(1.4) C:7(0.9) B:7(0.9) E:min7(1.8) A:7(2.3) D:maj(1.4) B:min(1.8)
- `03_Jazz3-150-C_comp` 0.14
  - 参考：C:maj(1.6) G:maj(1.6) A:min(1.6) E:min(1.6) F:maj(1.6) C:maj(1.6) F:maj(1.6) G:maj(1.6) C:maj(1.6) G:maj(1.6) A:min(1.6) E:min(1.6) F:maj(1.6) C:maj(1.6) F:maj(1.6) G:maj(1.6)
  - 估计：E:min(1.6) G:7(1.2) E:min(4.4) F:maj(0.4) E:min(2.0) F:maj7(0.4) E:min(1.2) B:dim(0.8) E:min(2.0) E:min7(1.6) E:min(1.6) A:min7(3.2) E:min(1.6) A:min(1.2) E:min(0.4) E:7(0.8) B:dim(0.8)
- `05_BN1-147-Gb_comp` 0.17
  - 参考：F#:maj(6.5) B:maj(3.3) F#:maj(3.3) C#:maj(1.6) B:maj(1.6) F#:maj(3.3)
  - 估计：D#:min(1.6) G#:min6(1.6) D#:min(1.6) G#:min6(1.6) G#:min/b3(1.6) C#:min7(1.6) D#:min(1.6) G#:min6(1.6) C#:7(1.6) B:7(1.6) D#:min(3.3)
- `05_Funk2-119-G_comp` 0.20
  - 参考：C:min(2.0) F:7(2.0) A#:maj(2.0) D#:maj(2.0) A:hdim7(2.0) D:7(2.0) G:min(4.0) C:min(2.0) F:7(2.0) A#:maj(2.0) D#:maj(2.0) A:hdim7(2.0) D:7(2.0) G:min(4.0)
  - 估计：D#:maj(1.5) F:maj(2.0) A#:maj(1.0) A:dim(1.0) G:min(2.5) G:7(1.5) D#:maj(2.0) F:7(1.0)
- `03_Funk1-97-C_comp` 0.21
  - 参考：C:maj(9.9) F:maj(4.9) C:maj(4.9) G:maj(2.5) F:maj(2.5) C:maj(4.9)
  - 估计：A#:7(1.9) A#:maj(2.5) A:min7(2.5) G:min(2.5) F:maj(1.2) A:min7(6.2) A#:maj7(1.2) G:7(3.7) B:7(2.5) C:7(2.5) G:min(2.5)
- `03_Jazz1-200-B_comp` 0.25
  - 参考：B:maj(4.8) E:maj(2.4) B:maj(2.4) F#:maj(1.2) E:maj(1.2) B:maj(2.4)
  - 估计：G#:min(4.8) E:7(2.4) G#:min(2.4) A#:hdim7(0.9) E:maj(1.5) G#:min(2.4)
- `04_Funk2-108-Eb_comp` 0.25
  - 参考：G#:min(2.2) C#:7(2.2) F#:maj(2.2) B:maj(2.2) F:hdim7(2.2) A#:7(2.2) D#:min(4.4) G#:min(2.2) C#:7(2.2) F#:maj(2.2) B:maj(2.2) F:hdim7(2.2) A#:7(2.2) D#:min(4.4)
  - 估计：G#:min(4.4) C#:maj(2.2) G#:min7(2.2) G#:min(2.2) A#:7(2.2) F#:maj(4.4) G#:min7(4.4) A#:min7(2.2) D#:min(2.2) G#:min(2.2) A#:7(2.2) F#:maj(4.4)
- `05_Funk1-97-C_comp` 0.29
  - 参考：C:maj(9.9) F:maj(4.9) C:maj(4.9) G:maj(2.5) F:maj(2.5) C:maj(4.9)
  - 估计：A:min7(3.1) F:maj(1.9) E:maj(1.9) F:min(1.2) A:min7(2.5) C:maj(2.5) G:7(4.9) A:min7(2.5) C:maj(1.9)
- `05_Funk3-112-C#_comp` 0.36
  - 参考：C#:maj(2.1) G#:maj(2.1) A#:min(2.1) F:min(2.1) F#:maj(2.1) C#:maj(2.1) F#:maj(2.1) G#:maj(2.1) C#:maj(2.1) G#:maj(2.1) A#:min(2.1) F:min(2.1) F#:maj(2.1) C#:maj(2.1) F#:maj(2.1) G#:maj(2.1)
  - 估计：G#:maj(3.2) A#:min(1.6) F:min7(2.1) C#:maj(3.2) A#:min7(2.1) G#:7(3.2) C#:maj7(1.6) G#:7(2.7) C#:maj(1.1) A#:min(2.7) G#:7(2.1)
- `05_Jazz2-187-F#_comp` 0.41
  - 参考：B:min(1.3) E:7(1.3) A:maj(1.3) D:maj(1.3) G#:hdim7(1.3) C#:7(1.3) F#:min(2.6) B:min(1.3) E:7(1.3) A:maj(1.3) D:maj(1.3) G#:hdim7(1.3) C#:7(1.3) F#:min(2.6)
  - 估计：B:min7(1.3) E:7(1.3) A:7(1.3) F#:min7(1.3) B:min7(2.2) E:7(0.3) A:maj(2.6) D:maj(1.3) E:7(1.3) A:7(1.3) D:maj(1.3) B:min7(2.6) D:maj(1.3) F#:min(0.6) D#:dim(0.6)
- `03_Jazz3-137-Eb_comp` 0.42
  - 参考：D#:maj(1.8) A#:maj(1.8) C:min(1.8) G:min(1.8) G#:maj(1.8) D#:maj(1.8) G#:maj(1.8) A#:maj(1.8) D#:maj(1.8) A#:maj(1.8) C:min(1.8) G:min(1.8) G#:maj(1.8) D#:maj(1.8) G#:maj(1.8) A#:maj(1.7)
  - 估计：G:min(1.3) A#:7(1.8) A#:sus4(0.9) D#:maj(0.9) A#:maj(1.8) G#:maj7(1.3) G:min(2.2) A#:7(3.5) G:min(1.8) A#:7(1.8) D#:maj(0.9) A#:maj(0.9) A#:7(1.8) G#:maj7(1.8) G:min(1.8) G#:maj7(1.8) A#:7(1.7)
- `05_Jazz1-200-B_comp` 0.46
  - 参考：B:maj(4.8) E:maj(2.4) B:maj(2.4) F#:maj(1.2) E:maj(1.2) B:maj(2.4)
  - 估计：B:7(1.2) E:7(1.2) B:7(2.4) E:7(3.6) A:7(1.2) E:7(1.2) F#:7(1.2) B:7(0.6) D:7(0.6) G:maj(0.6) C:7(0.6)
- `05_Funk3-98-A_comp` 0.48
  - 参考：A:maj(2.4) E:maj(2.4) F#:min(2.4) C#:min(2.4) D:maj(2.4) A:maj(2.4) D:maj(2.4) E:maj(2.4) A:maj(2.4) E:maj(2.4) F#:min(2.4) C#:min(2.4) D:maj(2.4) A:maj(2.4) D:maj(2.4) E:maj(2.4)
  - 估计：A:maj(3.1) E:maj(3.1) F#:min7(2.4) E:maj(0.6) A:maj(2.4) B:min(1.2) E:7(3.7) A:maj(1.8) E:maj(1.8) A:maj(3.1) C#:min(1.8) E:maj(0.6) D:maj(1.8) A:maj(3.7) D:maj(2.4) E:maj(1.2)

## 调性错误（committee，19 首）

- `03_Funk1-114-Ab_comp` 参考 G# major，估计 C# major；和弦：G#:7(1.6) G#:min7(6.3) C#:7(4.2) G#:7(4.2) D#:7(1.6) C#:7(1.6) E:maj(1.1) G#:7(4.2)
- `03_Funk1-97-C_comp` 参考 C major，估计 F major；和弦：A#:7(1.9) A#:maj(2.5) A:min7(2.5) G:min(2.5) F:maj(1.2) A:min7(6.2) A#:maj7(1.2) G:7(3.7) B:7(2.5) C:7(2.5) G:min(2.5)
- `03_Funk2-108-Eb_comp` 参考 D# minor，估计 F# major；和弦：B:maj(2.2) G#:min(2.2) F#:maj(2.2) B:maj(2.2) F:hdim7(2.2) A#:7(2.2) D#:min(1.1) F#:maj(3.3) B:maj(2.2) G#:min(2.2) F#:maj(2.2) B:maj(2.2) F:hdim7(2.2) A#:7(2.2) F#:maj(4.4)
- `03_Jazz1-130-D_comp` 参考 D major，估计 E minor；和弦：A:min7(1.4) E:min(3.7) A:min7(1.8) E:min(3.7) F#:min7(1.4) A:7(1.4) E:min(1.8) D:maj(0.9) A:min(2.3) A:min7(0.5)
- `03_Jazz1-200-B_comp` 参考 B major，估计 G# minor；和弦：G#:min(4.8) E:7(2.4) G#:min(2.4) A#:hdim7(0.9) E:maj(1.5) G#:min(2.4)
- `03_Jazz3-150-C_comp` 参考 C major，估计 E minor；和弦：E:min(1.6) G:7(1.2) E:min(4.4) F:maj(0.4) E:min(2.0) F:maj7(0.4) E:min(1.2) B:dim(0.8) E:min(2.0) E:min7(1.6) E:min(1.6) A:min7(3.2) E:min(1.6) A:min(1.2) E:min(0.4) E:7(0.8) B:dim(0.8)
- `04_Funk1-97-C_comp` 参考 C major，估计 F major；和弦：C:7(2.5) A#:maj(2.5) F:7(4.9) F:maj(4.9) G:7(1.2) F:maj(1.2) F:7(2.5) C:7(2.5) C:maj(2.5)
- `04_Funk2-108-Eb_comp` 参考 D# minor，估计 G# minor；和弦：G#:min(4.4) C#:maj(2.2) G#:min7(2.2) G#:min(2.2) A#:7(2.2) F#:maj(4.4) G#:min7(4.4) A#:min7(2.2) D#:min(2.2) G#:min(2.2) A#:7(2.2) F#:maj(4.4)
- `04_Jazz1-130-D_comp` 参考 D major，估计 G major；和弦：D:7(7.4) G:7(3.7) D:7(3.7) A:7(1.4) G:7(2.3) D:7(1.8) A:7(1.8)
- `04_Jazz1-200-B_comp` 参考 B major，估计 E major；和弦：B:7(4.8) E:7(2.1) B:7(0.9) A:7(0.6) G#:7(1.2) C#:min7(1.2) F#:min7(1.2) B:7(2.4)
- `05_BN1-147-Gb_comp` 参考 F# major，估计 D# minor；和弦：D#:min(1.6) G#:min6(1.6) D#:min(1.6) G#:min6(1.6) G#:min/b3(1.6) C#:min7(1.6) D#:min(1.6) G#:min6(1.6) C#:7(1.6) B:7(1.6) D#:min(3.3)
- `05_BN2-131-B_comp` 参考 B minor，估计 D major；和弦：E:min7(1.8) A:7(1.8) D:maj(1.8) G:maj7(1.8) C#:hdim7(1.8) F#:7(1.8) B:min7(1.8) B:7(1.8) E:min7(1.8) A:7(1.8) D:maj(1.8) G:7(1.8) C#:hdim7(1.8) F#:7(1.8) G:maj/3(3.7)
- `05_Funk1-97-C_comp` 参考 C major，估计 A minor；和弦：A:min7(3.1) F:maj(1.9) E:maj(1.9) F:min(1.2) A:min7(2.5) C:maj(2.5) G:7(4.9) A:min7(2.5) C:maj(1.9)
- `05_Jazz1-130-D_comp` 参考 D major，估计 B minor；和弦：B:min(2.8) G:7(1.8) B:min(3.7) G:7(1.4) C#:7(1.4) C:7(0.9) B:7(0.9) E:min7(1.8) A:7(2.3) D:maj(1.4) B:min(1.8)
- `05_Jazz1-200-B_comp` 参考 B major，估计 E major；和弦：B:7(1.2) E:7(1.2) B:7(2.4) E:7(3.6) A:7(1.2) E:7(1.2) F#:7(1.2) B:7(0.6) D:7(0.6) G:maj(0.6) C:7(0.6)
- `05_Jazz2-110-Bb_comp` 参考 A# minor，估计 C# major；和弦：D#:min7(2.2) G#:7(2.2) C#:maj(2.2) F#:7(2.2) C:hdim7(2.2) F:7(2.2) A#:min7(2.2) A#:7(2.2) D#:min7(2.2) G#:7(2.2) C#:maj(2.2) F#:maj7(2.2) C:hdim7(2.2) F:7(1.6) E:maj(0.5) A#:min7(0.5) A:dim(1.6) C#:maj(2.2)
- `05_Jazz2-187-F#_comp` 参考 F# minor，估计 A major；和弦：B:min7(1.3) E:7(1.3) A:7(1.3) F#:min7(1.3) B:min7(2.2) E:7(0.3) A:maj(2.6) D:maj(1.3) E:7(1.3) A:7(1.3) D:maj(1.3) B:min7(2.6) D:maj(1.3) F#:min(0.6) D#:dim(0.6)
- `05_Jazz3-150-C_comp` 参考 C major，估计 A minor；和弦：C:maj7(1.6) G:7(1.6) A:min(1.6) G:maj(1.6) F:maj7(1.6) C:maj7(1.6) F:maj7(1.6) G:7(1.6) C:maj7(1.6) G:7(1.6) A:min7(2.4) E:min7(0.8) F:maj(1.6) C:maj(1.6) A:min(2.0) G:7(1.2)
- `05_Rock2-85-F_comp` 参考 F minor，估计 G# major；和弦：A#:min7(2.8) D#:7(2.8) G#:maj(2.8) C#:maj(2.8) G:hdim7(2.8) C:7(2.8) F:min7(2.8) F:7(2.8) A#:min7(2.8) D#:7(2.8) G#:maj(2.8) C#:maj(2.8) G:min7(2.8) C:7(2.8) F:maj(5.6)
# 和弦 / 调性错误分析：tune_comp

融合参数：`FusionParams(weights={'btc': 0.5, 'lvchordia': 0.5, 'chroma': 0.5}, key_prior=0.5, change_cost_downbeat=0.1, change_cost_beat=3.0, no_chord_bias=-1.0, foreign_bass_penalty=0.7, min_bass_strength=0.3, fill_gaps_under_bar=True)`

## 错误分解（90 首，按帧，0.05 s 一帧）

| 系统 | correct | ref_N | est_N | third | relative | fourth_fifth | other_root | 错误中在参考边界一拍内 |
|---|---|---|---|---|---|---|---|---|
| btc | 87.3% | 0.0% | 0.1% | 7.9% | 1.0% | 1.2% | 2.4% | 42% |
| lvchordia | 88.7% | 0.0% | 0.0% | 7.8% | 0.8% | 1.2% | 1.4% | 43% |
| chroma | 77.1% | 0.0% | 0.1% | 3.2% | 3.6% | 7.7% | 8.2% | 43% |
| committee | 88.7% | 0.0% | 0.1% | 7.8% | 0.9% | 0.9% | 1.6% | 44% |

任一模型专家正确（融合的上限）：91.8%

## 按风格（committee）

| 风格 | correct | ref_N | est_N | third | relative | fourth_fifth | other_root |
|---|---|---|---|---|---|---|---|
| BN | 87.7% | 0.0% | 0.0% | 5.9% | 2.5% | 2.3% | 1.5% |
| Fu | 84.6% | 0.0% | 0.5% | 11.7% | 0.4% | 0.5% | 2.4% |
| Ja | 87.4% | 0.0% | 0.1% | 4.9% | 2.4% | 1.7% | 3.4% |
| Ro | 91.6% | 0.0% | 0.0% | 7.5% | 0.0% | 0.8% | 0.1% |
| SS | 91.2% | 0.0% | 0.0% | 7.5% | 0.2% | 0.1% | 1.0% |

## 最差的 5 首（committee，majmin 帧正确率）

- `01_Funk1-114-Ab_comp` 0.44
  - 参考：G#:maj(8.4) C#:maj(4.2) G#:maj(4.2) D#:maj(2.1) C#:maj(2.1) G#:maj(4.2)
  - 估计：G#:min(8.4) C#:min(1.6) C#:maj(2.6) G#:min(4.2) D#:maj(2.1) C#:maj(2.1) G#:maj(4.2)
- `02_Funk2-108-Eb_comp` 0.50
  - 参考：G#:min(2.2) C#:7(2.2) F#:maj(2.2) B:maj(2.2) F:hdim7(2.2) A#:7(2.2) D#:min(4.4) G#:min(2.2) C#:7(2.2) F#:maj(2.2) B:maj(2.2) F:hdim7(2.2) A#:7(2.2) D#:min(4.4)
  - 估计：G#:maj(2.2) C#:maj(2.2) F#:maj(2.2) B:maj(2.2) F:maj(2.2) A#:maj(2.2) D#:maj(4.4) G#:maj(2.2) C#:maj(2.2) F#:maj(2.2) B:maj(2.2) B:dim(2.2) A#:maj(2.2) D#:maj(3.9)
- `02_SS2-107-Ab_comp` 0.63
  - 参考：C#:min(2.2) F#:7(2.2) B:maj(2.2) E:maj(2.2) A#:hdim7(2.2) D#:7(2.2) G#:min(4.5) C#:min(2.2) F#:7(2.2) B:maj(2.2) E:maj(2.2) A#:hdim7(2.2) D#:7(2.2) G#:min(4.5)
  - 估计：C#:maj(2.2) F#:maj(2.2) B:maj(2.2) E:maj(2.2) A#:min(2.2) D#:min(2.2) G#:min(4.5) C#:maj(2.2) F#:maj(2.2) B:maj(2.2) E:maj(2.2) A#:maj(2.2) D#:maj(2.2) G#:maj(4.5)
- `00_Jazz3-137-Eb_comp` 0.64
  - 参考：D#:maj(1.8) A#:maj(1.8) C:min(1.8) G:min(1.8) G#:maj(1.8) D#:maj(1.8) G#:maj(1.8) A#:maj(1.8) D#:maj(1.8) A#:maj(1.8) C:min(1.8) G:min(1.8) G#:maj(1.8) D#:maj(1.8) G#:maj(1.8) A#:maj(1.7)
  - 估计：G:min(3.5) C:min7(1.8) G:min(1.8) G#:maj(1.8) D#:maj(1.3) G#:maj(2.2) G:min(5.3) C:min7(1.8) G:min(1.8) G#:maj(1.8) D#:maj(1.8) G#:maj(1.8) A#:maj(0.9) D#:maj(0.9)
- `02_BN2-166-Ab_comp` 0.65
  - 参考：C#:min(1.4) F#:7(1.4) B:maj(1.4) E:maj(1.4) A#:hdim7(1.4) D#:7(1.4) G#:min(2.9) C#:min(1.4) F#:7(1.4) B:maj(1.4) E:maj(1.4) A#:hdim7(1.4) D#:7(1.4) G#:min(2.9)
  - 估计：C#:min(1.4) F#:7(1.4) B:maj(1.4) E:maj(1.4) A#:maj(1.4) D#:maj(1.4) G#:min(2.9) C#:maj(1.4) F#:maj(0.7) D#:min(0.7) B:maj(1.4) E:maj(1.4) A#:maj(1.1) D#:maj(1.8) G#:maj(2.9)

## 调性错误（committee，5 首）

- `00_Funk2-108-Eb_comp` 参考 D# minor，估计 F# major；和弦：B:maj6(2.2) C#:7(2.2) F#:maj(2.2) B:maj(2.2) G#:min6(2.2) A#:7(2.2) D#:min(2.2) D#:min7(2.2) G#:min7(2.2) C#:7(2.2) F#:maj(2.2) B:maj(2.2) F:hdim7(2.2) A#:7(2.2) D#:maj(4.4)
- `01_Funk2-108-Eb_comp` 参考 D# minor，估计 G# minor；和弦：G#:min(2.2) C#:maj(2.2) F#:maj(2.2) B:maj(2.2) F:maj(2.2) A#:7(2.2) D#:maj(4.4) G#:min(2.2) C#:maj(2.2) F#:maj(2.2) B:maj(2.2) F:maj(2.2) A#:maj(2.2) D#:min(4.4)
- `02_BN2-131-B_comp` 参考 B minor，估计 D major；和弦：E:min(1.8) A:maj(1.8) D:maj(1.8) G:maj(1.8) C#:maj(1.8) F#:min(1.8) B:min(3.7) E:min(1.8) A:maj(1.8) D:maj(1.8) G:maj(1.8) C#:maj(1.8) F#:maj(1.8) B:maj(3.7)
- `02_Funk2-108-Eb_comp` 参考 D# minor，估计 G# minor；和弦：G#:maj(2.2) C#:maj(2.2) F#:maj(2.2) B:maj(2.2) F:maj(2.2) A#:maj(2.2) D#:maj(4.4) G#:maj(2.2) C#:maj(2.2) F#:maj(2.2) B:maj(2.2) B:dim(2.2) A#:maj(2.2) D#:maj(3.9)
- `02_SS2-88-F_comp` 参考 F minor，估计 G# major；和弦：A#:min(2.7) D#:maj(2.7) G#:maj(2.7) C#:maj(2.7) G:min(1.4) A#:maj(1.4) C:maj(2.7) F:maj(5.5) E:maj(0.7) A#:maj(2.0) D#:maj(2.7) G#:maj(2.7) C#:maj(2.7) G:min(2.7) C:maj(2.7) F:maj(5.4)
# 和弦 / 调性错误分析：aam_single

融合参数：`FusionParams(weights={'btc': 0.5, 'lvchordia': 0.5, 'chroma': 0.5}, key_prior=0.5, change_cost_downbeat=0.1, change_cost_beat=3.0, no_chord_bias=-1.0, foreign_bass_penalty=0.7, min_bass_strength=0.3, fill_gaps_under_bar=True)`

## 错误分解（72 首，按帧，0.05 s 一帧）

| 系统 | correct | ref_N | est_N | third | relative | fourth_fifth | other_root | 错误中在参考边界一拍内 |
|---|---|---|---|---|---|---|---|---|
| btc | 91.1% | 2.3% | 0.0% | 1.8% | 1.1% | 1.5% | 2.2% | 54% |
| lvchordia | 93.9% | 2.1% | 0.0% | 1.0% | 1.1% | 0.9% | 1.0% | 49% |
| chroma | 85.7% | 2.7% | 0.0% | 0.2% | 3.1% | 5.0% | 3.3% | 46% |
| committee | 94.9% | 2.3% | 0.0% | 1.5% | 0.5% | 0.3% | 0.4% | 34% |

任一模型专家正确（融合的上限）：95.3%

## 按风格（committee）

| 风格 | correct | ref_N | est_N | third | relative | fourth_fifth | other_root |
|---|---|---|---|---|---|---|---|
| aam | 94.9% | 2.3% | 0.0% | 1.5% | 0.5% | 0.3% | 0.4% |

## 最差的 5 首（committee，majmin 帧正确率）

- `0437` 0.42
  - 参考：N(12.0) E:min(2.0) D:maj(2.0) A:min(1.0) D:maj(3.0) B:min(2.0) A:min(2.0) G:maj(2.0) A:min(2.0) D:maj(2.0) A:min(2.0) B:min(2.0) E:min(4.0) A:min(2.0) E:min(2.0) G:maj(2.0) D:maj(2.0) E:min(2.0) N(60.0) E:min(2.0) D:maj(2.0) A:min(1.0) D:maj(3.0) B:min(2.0) A:min(2.0) G:maj(2.0) A:min(2.0) D:maj(2.0) A:min(2.0) B:min(2.0) E:min(2.0) N(12.0)
  - 估计：E:min(2.0) B:maj(2.0) C:maj(2.0) A:maj(2.0) G:maj(1.0) D:maj(1.0) E:min(2.0) E:min7(2.0) D:maj(2.0) A:min(1.0) D:maj(3.0) B:min(2.0) A:min(2.0) G:maj(2.0) A:min(2.0) D:maj(2.0) A:min(2.0) B:min(2.0) E:min(4.0) A:min(2.0) E:min(2.0) G:maj(2.0) D:maj(2.0) E:min(4.0) B:min(2.0) C:maj(2.0) A:maj(2.0) G:maj(1.0) D:maj(1.0) E:min(4.0) B:min(2.0) C:maj(2.0) A:maj(2.0) G:maj(1.0) D:maj(1.0) E:maj(4.0) B:maj(2.0) C:maj(2.0) A:maj(2.0) G:maj(1.0) D:maj(1.0) E:min(4.0) B:min(2.0) C:maj(2.0) A:maj(2.0) G:maj(1.0) D:maj(1.0) E:min(4.0) B:maj(2.0) C:maj(2.0) A:maj(2.0) G:maj(1.0) D:maj(1.0) E:min(2.0) E:min7(2.0) D:maj(2.0) A:min(1.0) D:maj(3.0) B:min(2.0) A:min(2.0) G:maj(2.0) A:min(2.0) D:maj(2.0) A:min(2.0) B:min(2.0) E:min(6.0) C:maj(2.0) A:maj(2.0) G:maj(1.0) D:maj(1.0) E:min(1.5)
- `0285` 0.59
  - 参考：F:maj(1.5) E:min(1.5) G:maj(1.5) E:min(3.7) D:min(2.2) F:maj(3.0) G:maj(3.0) A:min(1.5) D:min(1.5) G:maj(1.5) E:min(1.5) A:min(1.5) N(17.8) F:maj(4.4) A:min(1.5) F:maj(1.5) E:min(1.5) G:maj(1.5) E:min(3.7) D:min(2.2) F:maj(3.0) G:maj(3.0) A:min(1.5) D:min(1.5) G:maj(1.5) E:min(1.5) A:min(1.5) F:maj(1.5) E:min(1.5) G:maj(1.5) E:min(3.7) D:min(2.2) F:maj(3.0) G:maj(3.0) A:min(1.5) D:min(1.5) G:maj(1.5) E:min(1.5) A:min(1.5) F:maj(4.4) A:min(1.5) F:maj(4.4) A:min(1.5) F:maj(4.4) A:min(1.5) N(53.3) F:maj(4.4) A:min(1.5)
  - 估计：F:maj(1.5) E:min(1.5) G:maj(1.5) E:min(3.7) D:min(2.2) F:maj(3.0) G:maj(3.0) A:min(1.5) D:min(1.5) G:maj(1.5) E:min(1.5) A:min(3.0) E:maj(1.5) F:maj(1.5) D:min(1.5) A:min(1.5) G:maj(1.5) C:maj(1.5) F:maj(1.5) G:maj(1.5) C:maj(0.7) F:maj(0.7) D:maj(1.5) A:min(1.5) F:maj(4.4) A:min(1.5) F:maj(1.5) E:min(1.5) G:maj(1.5) E:min(3.7) D:min(2.2) F:maj(3.0) G:maj(3.0) A:min(1.5) D:min(1.5) G:maj(1.5) E:min(1.5) A:min(1.5) F:maj(1.5) E:min(1.5) G:maj(1.5) E:min(3.7) D:min(2.2) F:maj(3.0) G:maj(3.0) A:min(1.5) D:min(1.5) G:maj(1.5) E:min(1.5) A:min(1.5) F:maj(4.4) A:min(1.5) F:maj(4.4) A:min(1.5) F:maj(4.4) A:min(3.0) E:maj(1.5) F:maj(1.5) D:min(1.5) A:maj(1.5) G:maj(1.5) C:maj(1.5) F:maj(1.5) G:maj(1.5) C:maj(0.7) F:maj(0.7) D:maj(1.9) A:maj(2.6) E:maj(1.5) F:maj(1.5) D:min(1.5) A:maj(1.5) G:maj(1.5) C:maj(1.5) F:maj(1.5) G:maj(1.5) C:maj(0.7) F:maj(0.7) D:maj(1.9) A:maj(2.6) E:maj(1.5) F:maj(1.5) D:min(1.5) A:min(1.5) G:maj(1.5) C:maj(1.5) F:maj(1.5) G:maj(1.5) C:maj(0.7) F:maj(0.7) D:maj(1.9) A:maj(1.1) F:maj(4.4) A:min(4.2)
- `0473` 0.67
  - 参考：A:maj(1.5) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(1.5) N(11.9) A:maj(1.5) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(1.5) N(11.9) A:maj(1.5) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(1.5) E:maj(1.5) B:min(1.5) D:maj(1.5) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(1.5) N(35.8) A:maj(1.5) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(1.5) E:maj(1.5) B:min(1.5) D:maj(1.5) A:maj(1.5)
  - 估计：A:maj(1.5) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(1.5) E:maj(1.5) B:maj(1.5) E:maj(1.5) D:maj(1.5) F#:min(1.5) F#:maj(3.0) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(1.5) E:maj(1.5) B:maj(1.5) E:maj(1.5) D:maj(1.5) F#:min(4.5) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(1.5) E:maj(1.5) B:min(1.5) D:maj(1.5) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(1.5) E:maj(1.5) B:maj(1.5) E:maj(1.5) D:maj(1.5) F#:maj(4.5) A:maj(1.5) E:maj(1.5) B:maj(1.5) E:maj(1.5) D:maj(1.5) F#:maj(4.5) A:maj(1.5) E:maj(1.5) B:maj(1.5) E:maj(1.5) D:maj(1.5) F#:maj(1.5) F#:min(3.0) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(3.0) B:min(1.5) F#:min(1.5) E:maj(1.5) A:maj(0.7) E:maj(0.7) A:maj(1.5) E:maj(1.5) B:min(1.5) D:maj(1.5) A:maj(4.2)
- `0096` 0.71
  - 参考：A:maj(4.8) C#:min(2.4) G#:min(2.4) E:maj(2.4) G#:min(2.4) A:maj(2.4) C#:min(2.4) A:maj(4.8) C#:min(2.4) G#:min(2.4) E:maj(2.4) G#:min(2.4) A:maj(2.4) C#:min(2.4) B:maj(2.4) G#:min(2.4) E:maj(2.4) A:maj(2.4) F#:min(4.8) B:maj(2.4) G#:min(2.4) E:maj(4.8) B:maj(2.4) C#:min(2.4) A:maj(2.4) F#:min(4.8) C#:min(2.4) A:maj(4.8) C#:min(2.4) G#:min(2.4) E:maj(2.4) G#:min(2.4) A:maj(2.4) C#:min(2.4) A:maj(4.8) C#:min(2.4) G#:min(2.4) E:maj(2.4) G#:min(2.4) A:maj(2.4) C#:min(2.4) A:maj(4.8) C#:min(2.4) G#:min(2.4) E:maj(2.4) G#:min(2.4) A:maj(2.4) C#:min(2.4) A:maj(4.8) C#:min(2.4) G#:min(2.4) E:maj(2.4) G#:min(2.4) A:maj(2.4) C#:min(2.4)
  - 估计：A:maj(4.8) C#:maj(2.4) G#:min(2.4) E:maj(2.4) G#:maj(2.4) A:maj(2.4) C#:maj(2.4) A:maj(4.8) C#:maj(2.4) G#:min(2.4) E:maj(2.4) G#:maj(2.4) A:maj(2.4) C#:maj(2.4) B:maj(2.4) G#:min(2.4) E:maj(2.4) A:maj(2.4) F#:min(2.4) F#:maj(2.4) B:maj(2.4) G#:min(2.4) E:maj(6.1) B:maj(1.2) C#:min(2.4) A:maj(2.4) F#:min(2.4) F#:maj(2.4) C#:maj(2.4) A:maj(4.8) C#:min(2.4) G#:min(2.4) E:maj(2.4) G#:maj(2.4) A:maj(2.4) C#:maj(2.4) A:maj(4.8) C#:min(2.4) G#:min(2.4) E:maj(2.4) G#:maj(2.4) A:maj(2.4) C#:maj(2.4) A:maj(4.8) C#:maj(2.4) G#:min(2.4) E:maj(2.4) G#:maj(2.4) A:maj(2.4) C#:maj(2.4) A:maj(4.8) C#:min(2.4) G#:min(2.4) E:maj(2.4) G#:maj(2.4) A:maj(2.4) C#:maj(5.1)
- `0339` 0.80
  - 参考：D#:maj(2.0) G#:maj(2.0) C:min(1.0) D#:maj(1.0) F:min(2.0) D#:maj(2.0) A#:min(1.0) C:min(1.0) A#:min(2.0) D#:maj(2.0) G#:maj(2.0) F:min(2.0) C#:maj(2.0) F:min(2.0) A#:min(2.0) D#:maj(3.1) F:min(3.1) C:min(4.1) A#:min(2.0) F:min(5.1) G#:maj(3.1) C:min(3.1) C#:maj(3.1) D#:maj(3.1) A#:min(3.1) D#:maj(3.1) F:min(3.1) D#:maj(2.0) A#:min(1.0) C:min(1.0) A#:min(2.0) D#:maj(2.0) G#:maj(2.0) F:min(2.0) C#:maj(2.0) F:min(2.0) A#:min(2.0) D#:maj(3.1) F:min(3.1) C:min(4.1) A#:min(2.0) F:min(2.0) D#:maj(2.0) A#:min(1.0) C:min(1.0) A#:min(2.0) D#:maj(2.0) G#:maj(2.0) F:min(2.0) C#:maj(2.0) F:min(2.0) A#:min(2.0) D#:maj(3.1) F:min(3.1) C:min(4.1) A#:min(2.0) F:min(2.0)
  - 估计：D#:maj(2.0) G#:maj(3.1) D#:maj(1.0) F:min(2.0) D#:maj(2.0) A#:maj(1.0) C:sus4(1.0) A#:maj(2.0) D#:maj(2.0) G#:maj(2.0) F:min(2.0) C#:maj(2.0) F:min(2.0) A#:min(2.0) D#:maj(3.1) F:min7(1.0) F:min(2.0) C:min7(2.0) C:min(2.0) A#:maj(2.0) F:min(5.1) G#:maj(3.1) C:min(3.1) C#:maj(3.1) D#:maj(3.1) A#:min(3.1) D#:maj(3.1) F:min(3.1) D#:maj(2.0) A#:maj(1.0) C:sus4(1.0) A#:maj(0.5) D#:maj(3.6) G#:maj(2.0) F:min(2.0) C#:maj(2.0) F:min(2.0) A#:maj(2.0) D#:maj(3.1) F:min7(1.0) F:min(2.0) C:min7(2.0) C:min(2.0) A#:maj(2.0) F:maj(2.0) D#:maj(2.0) A#:maj(1.0) C:sus4(1.0) A#:maj(2.0) D#:maj(2.0) G#:maj(2.0) F:min(2.0) C#:maj(2.0) F:min(2.0) A#:maj(2.0) D#:maj(3.1) F:min7(1.0) F:min(2.0) C:min7(2.0) C:maj(2.0) A#:maj(2.0) F:maj(1.5)

## 调性错误（committee，5 首）

- `0003` 参考 D major，估计 B minor；和弦：A:maj(10.3) E:min(3.4) F#:min(3.4) B:min(6.9) E:min(1.7) A:maj(1.7) E:min(3.4) G:maj(3.4) F#:min(3.4) D:maj(3.4) A:maj(10.3) E:min(3.4) F#:min(3.4) B:min(6.9) E:min(1.7) A:maj(1.7) E:min(3.4) G:maj(3.4) F#:min(3.4) D:maj(3.4) A:maj(10.3) E:min(3.4) F#:min(3.4) B:min(6.9) E:min(1.7) A:maj(1.7) E:min(3.4) G:maj(3.4) F#:min(3.4) D:maj(3.4) G:maj(3.4) F#:min(5.1) A:maj(1.7) B:min(3.4) G:maj(3.4) F#:min(3.4) B:min(3.4) G:maj(3.4) E:min(3.4) A:maj(1.7) E:maj(1.7) E:min(0.9) B:min(2.6) A:maj(3.4) B:sus4(3.4) B:min(1.7) A:maj(5.1) D:maj(6.1)
- `0119` 参考 F major，估计 D minor；和弦：C:maj(2.7) F:maj(2.7) C:maj(4.0) F:maj(1.3) D:min(2.7) F:maj(2.7) A#:maj(3.4) G:min(1.7) D:min(5.1) C:maj(3.4) D:min(3.4) F:maj(3.4) A#:maj(3.4) G:min(1.7) D:min(5.1) C:maj(3.4) D:min(3.4) F:maj(3.4) A#:maj(3.4) G:min(1.7) D:min(5.1) C:maj(3.4) D:min(3.4) F:maj(3.4) A#:maj(3.4) G:min(1.7) D:min(5.1) C:maj(3.4) D:min(3.4) F:maj(3.4) A#:maj(3.4) G:min(1.7) D:min(5.1) C:maj(3.4) D:min(3.4) F:maj(3.4) A#:maj(3.4) G:min(1.7) D:min(5.1) C:maj(3.4) D:min(3.4) F:maj(6.1)
- `0208` 参考 D# major，估计 C minor；和弦：D#:maj(3.5) G#:maj(8.8) C:min(1.8) F:min(3.5) C:min(2.6) D#:maj7(0.9) G:min(10.6) A#:maj(2.6) F:min(0.9) C:min(3.5) C:min7(3.5) G#:maj(3.5) G:min(3.5) C:min(1.8) G#:maj6(1.8) D#:maj(7.1) G#:maj(8.8) C:min(1.8) F:min(3.5) C:min(2.6) D#:maj7(0.9) G:min(10.6) A#:maj(2.6) F:min(0.9) C:min(3.5) C:min7(3.5) G#:maj(3.5) G:min(3.5) C:min(1.8) G#:maj6(1.8) D#:maj(7.1) C:min(3.5) G:min(3.5) D#:maj(6.3)
- `0221` 参考 A major，估计 E major；和弦：E:maj(2.7) F#:min(0.9) E:maj(1.8) A:maj(1.8) E:maj(2.7) F#:min(0.9) E:maj(1.8) A:maj(1.8) E:maj(2.7) F#:min(0.9) E:maj(1.8) A:maj(1.8) E:maj(2.7) F#:min(0.9) E:maj(1.8) A:maj(1.8) E:maj(2.7) F#:min(0.9) E:maj(0.9) A:maj(2.7) E:maj(2.7) F#:min(0.9) E:maj(0.9) A:maj(2.7) E:maj(2.7) F#:min(0.9) E:maj(1.8) A:maj(5.4) F#:min(1.8) B:min(1.8) F#:min(3.6) C#:min(1.8) A:maj(1.8) E:maj(3.6) F#:min(3.6) C#:min(3.6) A:maj(7.2) F#:min(1.8) B:min(1.8) F#:min(3.6) C#:min(1.8) A:maj(1.8) E:maj(3.6) F#:min(3.6) C#:min7(3.6) A:maj(7.2) F#:min(1.8) B:min(1.8) F#:min(3.6) C#:min(1.8) A:maj(1.8) E:maj(3.6) F#:min(3.6) C#:min(3.6) A:maj(3.6) E:maj(2.7) F#:min(0.9) E:maj(1.8) A:maj(1.8) E:maj(2.7) F#:min(0.9) E:maj(1.8) A:maj(0.9)
- `0227` 参考 D minor，估计 F major；和弦：A#:maj(3.7) D:min(1.8) F:maj(3.7) A:min(1.8) G:min(1.8) D:min(5.5) F:maj(7.4) G:min(3.7) C:maj(3.7) A:min(3.7) F:maj(1.8) A:min(1.8) F:maj(3.7) A#:maj(7.4) G:min(3.7) D:min(3.7) C:maj(5.5) G:min(1.8) F:maj(3.7) D:min(7.4) F:maj(7.4) G:min(3.7) C:maj(3.7) A:min(3.7) F:maj(1.8) A:min(1.8) F:maj(3.7) A#:maj(7.4) G:min(3.7) D:min(3.7) C:maj(5.5) G:min(1.8) F:maj(3.7) D:min(6.4)
