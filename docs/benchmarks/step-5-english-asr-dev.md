# 英文 ASR 补充对比：Whisper large-v3 与 Parakeet TDT 0.6B v2

同一内存人声、同一 Silero 区间及局部过滤、固定逐曲种子 0。两者均指定英文；不提供歌词提示。Parakeet 使用未量化 ONNX 转换，结果不等同于官方 NeMo 运行时。

| 分组 | 条目 | Whisper WER | Parakeet WER | Whisper ASR 中位秒 | Parakeet ASR 中位秒 |
|---|---:|---:|---:|---:|---:|
| dev/dry | 10 | 15.6% | 19.3% | 4.95 | 4.74 |
| dev/instrumental | 10 | — | — | 0.00 | 0.00 |
| dev/mix | 10 | 17.0% | 23.7% | 5.00 | 5.35 |

计时只含识别调用，不含分轨、VAD、模型加载、CTC 对齐、渲染或排队。Whisper 每曲新进程避免 GPU 分配累积；Parakeet 复用模型。运行时和精度不同，这里比较当前可运行配置，不是同精度架构速度基准。

Parakeet 每个保留的 VAD 区间独立解码，最长 30 秒，不重叠；Whisper 沿用 clip_timestamps。Parakeet 词尾由末 token 发射时间加一帧估计，只用于标记待核验区间，不代表对齐精度。

原标注空隙不等于纯伴奏。JSON 单独列出修订人声标注支持、人工确认幻听与待核验词。纯音乐无输出可能由共享 VAD 门控直接产生，不能归功于候选模型。

20 首英文真实歌曲此前已用于诊断，这次是配对回归而非新盲测。开发 dry/mix 是同 10 首歌曲的两个条件，不当成 20 首独立歌曲。逐曲配对 bootstrap 仅刻画本小样本的抽样波动，不保证泛化。

dev/dry / whisper：无歌词 0/10；空隙词分类 {}。

dev/dry / parakeet：无歌词 0/10；空隙词分类 {}。

dev/instrumental / whisper：无歌词 10/10；空隙词分类 {}。

dev/instrumental / parakeet：无歌词 10/10；空隙词分类 {}。

dev/mix / whisper：无歌词 0/10；空隙词分类 {}。

dev/mix / parakeet：无歌词 0/10；空隙词分类 {"unresolved": 1}。
