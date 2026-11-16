"""Paired English ASR experiment; stems stay in memory and transcripts outside git.

Install onnx-asr==0.12.0 and onnxruntime-gpu==1.23.2 in an isolated experiment
environment alongside the project's analysis/evaluation dependencies. Download
the pinned, unquantized ONNX files documented in docs/plans/backend.md first.
Run --split dev before --split test, with the same --protocol and --cache.
The candidate uses each retained Silero interval, split at 30 seconds if needed.
Whisper uses its existing clip_timestamps implementation. Neither sees references.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from eval_lyrics import gap_candidates, load_vocals, norm_en, seed_inference
from lyrics_gap_review import classify

from openbapo.analysis.lyrics.activity import ActivityParams, SileroActivity
from openbapo.analysis.lyrics.recognize import Recognition, RecognizedLine, RecognizedWord

REVISION = "0bbb45a3365852604aef28b538a8f066f4ccaa85"
MODEL = "nemo-parakeet-tdt-0.6b-v2"
SR = 16000
MAX_CHUNK_S = 30.0
WEIGHT_HASHES = {
    "encoder-model.onnx": "3987bcd28175d829d12888a996a84e8f62a0e374d9ffd640662c1515adc679d3",
    "encoder-model.onnx.data": "4dab7362d4874d85965045b1e41b2d61dd2cc0fb25671a7f6b3dc47bf120cc41",
    "decoder_joint-model.onnx": "cbb52a07bd70ab5b67f8439d4b3cd8704b18467b4430bcacb5adabe154b8d191",
    "nemo128.onnx": "a9fde1486ebfcc08f328d75ad4610c67835fea58c73ba57e3209a6f6cf019e9f",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def chunks(regions, max_seconds=MAX_CHUNK_S):
    for start, end in regions:
        while start < end:
            stop = min(end, start + max_seconds)
            yield start, stop
            start = stop


def token_words(tokens, timestamps, offset, end):
    """SentencePiece leading spaces delimit words; times are token emissions.

    ONNX TDT does not expose word-end times. Use the last token emission + 80 ms
    (one encoder frame), capped at the clip boundary. These approximate spans
    are used only to flag gap candidates, never as forced-alignment accuracy.
    """
    if len(tokens) != len(timestamps):
        raise ValueError("token/timestamp length mismatch")
    words = []
    text = ""
    start = last = offset
    for token, stamp in zip(tokens, timestamps, strict=True):
        if token.startswith(" ") and text:
            words.append(RecognizedWord(start, min(end, last + 0.08), text))
            text = ""
        if not text:
            start = min(end, offset + stamp)
        text += token.lstrip() if not text else token
        last = min(end, offset + stamp)
    if text:
        words.append(RecognizedWord(start, min(end, last + 0.08), text))
    return tuple(words)


def parakeet_recognize(model, vocals, regions):
    lines = []
    for start, end in chunks(regions):
        result = model.recognize(vocals[int(start * SR) : int(end * SR)], sample_rate=SR)
        words = token_words(result.tokens, result.timestamps, start, end)
        if result.text.strip():
            lines.append(RecognizedLine(start, end, result.text.strip(), words))
    return Recognition("en", tuple(lines))


def whisper_worker(connection, vocals, regions, entry_id):
    """A fresh process avoids accumulated GPU allocations and reseeds worker RNGs."""
    try:
        from openbapo.analysis.lyrics.recognize import WhisperRecognizer, load_whisper

        seed_inference(entry_id, 0)
        started = time.perf_counter()
        model = load_whisper(device="cuda")
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        result = WhisperRecognizer(model).recognize(vocals, regions, language="en")
        connection.send((result, time.perf_counter() - started, load_seconds))
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def run_whisper(vocals, regions, entry_id):
    context = mp.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=whisper_worker, args=(writer, vocals, regions, entry_id))
    process.start()
    writer.close()
    try:
        if not reader.poll(600):
            raise TimeoutError("Whisper did not finish within 600 seconds")
        result = reader.recv()
        process.join(30)
        if process.exitcode != 0:
            raise RuntimeError(f"Whisper worker exit code: {process.exitcode}")
        if isinstance(result, dict):
            raise RuntimeError(result["error"])
        return result
    finally:
        reader.close()
        if process.is_alive():
            process.terminate()
            process.join()


def score(entry, result, revised, reviews, audio_hash):
    import jiwer

    reference = entry.get("reference_text") or "\n".join(
        line["text"] for line in entry.get("lines", [])
    )
    measured = jiwer.process_words(norm_en(reference), norm_en(result.text))
    candidates = gap_candidates(entry, result)
    classified = [
        {
            "start": w["start"],
            "end": w["end"],
            "classification": classify(w, revised, reviews, audio_hash),
        }
        for w in candidates
    ]
    return {
        "reference_words": len(norm_en(reference).split()),
        "hypothesis_words": len(norm_en(result.text).split()),
        "substitutions": measured.substitutions,
        "deletions": measured.deletions,
        "insertions": measured.insertions,
        "wer": measured.wer if reference else None,
        "no_lyrics": not result.text.strip(),
        "dropped_segments": result.dropped,
        "transcript_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
        "gap_candidates": classified,
    }


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(f"{row['split']}/{row['condition']}", []).append(row)
    out = {}
    for group, items in sorted(groups.items()):
        summary = {"entries": len(items)}
        for name in ("whisper", "parakeet"):
            recs = [item[name] for item in items]
            n = sum(r["reference_words"] for r in recs)
            errors = {
                k: sum(r[k] for r in recs) for k in ("substitutions", "deletions", "insertions")
            }
            summary[name] = {
                "reference_words": n,
                **errors,
                "wer": sum(errors.values()) / n if n else None,
                "no_lyrics": sum(r["no_lyrics"] for r in recs),
                "gap_classifications": dict(
                    Counter(w["classification"] for r in recs for w in r["gap_candidates"])
                ),
                "median_asr_seconds": float(np.median([r["asr_seconds"] for r in recs])),
                "total_asr_seconds": sum(r["asr_seconds"] for r in recs),
            }
        if group == "test/real":
            rng = np.random.default_rng(20261116)
            delta = []
            for _ in range(10000):
                selected = [items[i] for i in rng.integers(0, len(items), len(items))]
                denominator = sum(r["whisper"]["reference_words"] for r in selected)
                errors = {
                    name: sum(
                        r[name][k]
                        for r in selected
                        for k in ("substitutions", "deletions", "insertions")
                    )
                    for name in ("whisper", "parakeet")
                }
                delta.append((errors["parakeet"] - errors["whisper"]) / denominator)
            summary["paired_song_bootstrap_wer_delta_95pct"] = np.quantile(
                delta, [0.025, 0.975]
            ).tolist()
            summary["parakeet_lower_wer_songs"] = sum(
                r["parakeet"]["wer"] < r["whisper"]["wer"] for r in items
            )
        out[group] = summary
    return out


def report(args, protocol, rows):
    summary = summarize(rows)
    args.out.with_suffix(".json").write_text(
        json.dumps({"protocol": protocol, "summary": summary, "songs": rows}, indent=2) + "\n"
    )
    lines = [
        "# 英文 ASR 补充对比：Whisper large-v3 与 Parakeet TDT 0.6B v2",
        "",
        "同一内存人声、同一 Silero 区间及局部过滤、固定逐曲种子 0。两者均指定英文；"
        "不提供歌词提示。Parakeet 使用未量化 ONNX 转换，结果不等同于官方 NeMo 运行时。",
        "",
        "| 分组 | 条目 | Whisper WER | Parakeet WER | Whisper ASR 中位秒 | Parakeet ASR 中位秒 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, item in summary.items():
        w, p = item["whisper"], item["parakeet"]
        fmt = lambda x: "—" if x is None else f"{x:.1%}"  # noqa: E731
        lines.append(
            f"| {group} | {item['entries']} | {fmt(w['wer'])} | {fmt(p['wer'])} | "
            f"{w['median_asr_seconds']:.2f} | {p['median_asr_seconds']:.2f} |"
        )
    lines += [
        "",
        "计时只含识别调用，不含分轨、VAD、模型加载、CTC 对齐、渲染或排队。"
        "Whisper 每曲新进程避免 GPU 分配累积；Parakeet 复用模型。"
        "运行时和精度不同，这里比较当前可运行配置，不是同精度架构速度基准。",
        "",
        "Parakeet 每个保留的 VAD 区间独立解码，最长 30 秒，不重叠；"
        "Whisper 沿用 clip_timestamps。Parakeet 词尾由末 token 发射时间加一帧估计，"
        "只用于标记待核验区间，不代表对齐精度。",
        "",
        "原标注空隙不等于纯伴奏。JSON 单独列出修订人声标注支持、人工确认幻听与待核验词。"
        "纯音乐无输出可能由共享 VAD 门控直接产生，不能归功于候选模型。",
        "",
        "20 首英文真实歌曲此前已用于诊断，这次是配对回归而非新盲测。"
        "开发 dry/mix 是同 10 首歌曲的两个条件，不当成 20 首独立歌曲。"
        "逐曲配对 bootstrap 仅刻画本小样本的抽样波动，不保证泛化。",
    ]
    for group, item in summary.items():
        for name in ("whisper", "parakeet"):
            metrics = item[name]
            lines.append(
                f"\n{group} / {name}：无歌词 {metrics['no_lyrics']}/{item['entries']}；"
                f"空隙词分类 {json.dumps(metrics['gap_classifications'], ensure_ascii=False)}。"
            )
    if "test/real" in summary:
        item = summary["test/real"]
        lower, upper = item["paired_song_bootstrap_wer_delta_95pct"]
        lines.append(
            f"\n真实英文歌中 Parakeet WER 更低：{item['parakeet_lower_wer_songs']}/20 首；"
            f"WER 差（Parakeet − Whisper）的配对 95% 区间：{lower:+.1%} 至 {upper:+.1%}。"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("manifest", "model-dir", "jam-alt", "reviews", "protocol", "cache", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    import onnx_asr
    import onnxruntime as ort

    from openbapo.analysis.separate import DemucsSeparator

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("GPU runtime required; do not silently benchmark a CPU fallback")
    files = [Path(__file__), Path(__file__).with_name("eval_lyrics.py")]
    files += [Path(__file__).with_name("lyrics_gap_review.py")]
    files += sorted(Path("src/openbapo/analysis/lyrics").glob("*.py"))
    protocol = {
        "model": MODEL,
        "model_revision": REVISION,
        "weight_hash_source": "pinned HF LFS IDs; verified from files before candidate inference",
        "model_files_sha256": {
            p.name: sha(p) for p in sorted(args.model_dir.iterdir()) if p.is_file()
        }
        | (WEIGHT_HASHES if args.baseline_only else {}),
        "inputs_sha256": {p.name: sha(p) for p in (args.manifest, args.jam_alt, args.reviews)},
        "source_sha256": {
            str(p.resolve().relative_to(Path(__file__).resolve().parents[1])): sha(p) for p in files
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("onnx-asr", "onnxruntime-gpu", "faster-whisper", "ctranslate2", "torch")
        },
        "seed": 0,
        "language": "en",
        "activity": dataclasses.asdict(ActivityParams(context_gap_s=30.0)),
        "candidate_chunk_max_seconds": MAX_CHUNK_S,
        "candidate_quantization": None,
        "candidate_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "timing": "recognition calls only; excludes load, separation, VAD, alignment, rendering",
        "decision_rule": "replace only with lower real-English WER, no new confirmed or unresolved "
        "gap hallucinations, all instrumental gates passing; development precedes test",
    }
    if args.protocol.exists():
        if json.loads(args.protocol.read_text()) != protocol:
            raise ValueError("frozen protocol changed; use a new experiment and cache")
    else:
        for name, expected in WEIGHT_HASHES.items():
            if protocol["model_files_sha256"].get(name) != expected:
                raise ValueError(f"weights do not match pinned revision: {name}")
        args.protocol.write_text(json.dumps(protocol, indent=2) + "\n")
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    args.cache.mkdir(parents=True, exist_ok=True)
    entries = json.loads(args.manifest.read_text())["entries"]
    selected = [
        e
        for e in entries
        if e["split"] == args.split and (e["lang"] == "en" or e["condition"] == "instrumental")
    ][: args.limit]
    revised = {e["name"]: e for e in map(json.loads, args.jam_alt.read_text().splitlines())}
    reviews = json.loads(args.reviews.read_text())["reviews"]
    detector = SileroActivity(ActivityParams(context_gap_s=30.0))
    separator = DemucsSeparator(device="cuda")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    model = None
    if not args.baseline_only:
        started = time.perf_counter()
        model = onnx_asr.load_model(
            MODEL,
            str(args.model_dir),
            providers=protocol["candidate_providers"],
            sess_options=options,
        ).with_timestamps()
        for session in (model.asr._encoder, model.asr._decoder_joint):
            if "CUDAExecutionProvider" not in session.get_providers():
                raise RuntimeError("model fell back to CPU; aborting GPU comparison")
        print(f"Parakeet model loaded in {time.perf_counter() - started:.2f}s", flush=True)
    rows = []
    for entry in selected:
        audio_path = args.manifest.parent / entry["audio"]
        audio_hash = sha(audio_path)
        cache_path = args.cache / f"{entry['id']}.json"
        cached = None
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached["signature"] != signature or cached["row"]["audio_sha256"] != audio_hash:
                raise ValueError(f"stale cache: {entry['id']}")
            if args.baseline_only or "parakeet" in cached["row"]:
                rows.append(cached["row"])
                continue
        print(f"Starting {entry['id']}", flush=True)
        seed = seed_inference(entry["id"], 0)
        vocals, separate_seconds = load_vocals(entry, args.manifest.parent, separator)
        regions = detector.regions(vocals)
        if not detector.has_vocals(regions):
            regions = []
        row = {k: entry[k] for k in ("id", "split", "condition", "duration")} | {
            "audio_sha256": audio_hash,
            "vocals_sha256": hashlib.sha256(vocals.tobytes()).hexdigest(),
            "seed": seed,
            "regions": regions,
            "separate_seconds": separate_seconds,
        }
        if cached and (
            cached["row"]["vocals_sha256"] != row["vocals_sha256"]
            or cached["row"]["regions"] != [list(r) for r in regions]
        ):
            raise ValueError("repeated separation/VAD changed; cannot reuse paired baseline")
        raw = {} if cached is None else cached["raw"]
        for name in ("whisper",) if args.baseline_only else ("whisper", "parakeet"):
            if cached and name in cached["row"]:
                row[name] = cached["row"][name]
                continue
            load_seconds = None
            if not regions:
                result, elapsed = Recognition("en", ()), 0.0
            elif name == "whisper":
                result, elapsed, load_seconds = run_whisper(vocals, regions, entry["id"])
            else:
                started = time.perf_counter()
                result = parakeet_recognize(model, vocals, regions)
                elapsed = time.perf_counter() - started
            row[name] = score(
                entry,
                result,
                revised.get(audio_path.stem, {}).get("lines", []),
                [r for r in reviews if r["id"] == entry["id"]],
                audio_hash,
            ) | {"asr_seconds": elapsed, "load_seconds": load_seconds}
            raw[name] = dataclasses.asdict(result)
        cache_path.write_text(
            json.dumps({"signature": signature, "row": row, "raw": raw}, indent=2) + "\n"
        )
        rows.append(row)
        print(json.dumps({"id": entry["id"], **{n: row[n]["wer"] for n in raw}}), flush=True)
        if not args.baseline_only:
            report(args, protocol, rows)
        del vocals
    if not args.baseline_only:
        report(args, protocol, rows)


if __name__ == "__main__":
    main()
