"""Review old gap counts without changing model output or treating missing labels as silence.

Official revised vocal intervals can explain a candidate. Model agreement alone
cannot: remaining candidates need an explicit human verdict tied to the audio hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def classify(word, revised_lines, reviews, audio_sha256):
    midpoint = (word["start"] + word["end"]) / 2
    covering = [r for r in reviews if r["start"] <= midpoint <= r["end"]]
    for review in covering:
        if review["audio_sha256"] != audio_sha256:
            raise ValueError("review belongs to different audio")
        if review["verdict"] not in ("pending", "vocal", "instrumental"):
            raise ValueError("unknown review verdict")
        if review["verdict"] != "pending" and not review.get("reviewer"):
            raise ValueError("confirmed verdict needs an identified reviewer")
    verdicts = {r["verdict"] for r in covering if r["verdict"] != "pending"}
    if len(verdicts) > 1:
        raise ValueError("conflicting human reviews")
    if "instrumental" in verdicts:
        return "confirmed_hallucination"
    if "vocal" in verdicts:
        return "human_supported_vocal"
    if any(line["start"] <= midpoint <= line["end"] for line in revised_lines):
        return "reference_supported_vocal"
    return "unresolved"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--crosscheck", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--candidate-crosscheck", type=Path)
    parser.add_argument("--candidate-baseline", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.candidate_crosscheck) != bool(args.candidate_baseline):
        parser.error("candidate crosscheck and baseline must be supplied together")
    baseline = json.loads(args.baseline.read_text())
    diagnostic = json.loads(args.diagnostic.read_text())
    crosscheck = json.loads(args.crosscheck.read_text())
    development = json.loads(args.development.read_text())
    reviews = json.loads(args.reviews.read_text())["reviews"]
    old = {s["id"]: s for s in baseline["songs"]}
    evidence = {s["id"]: s for s in crosscheck["songs"]}
    counts = dict.fromkeys(
        (
            "reference_supported_vocal",
            "human_supported_vocal",
            "confirmed_hallucination",
            "unresolved",
        ),
        0,
    )
    rows = []
    for song in diagnostic["songs"]:
        candidates = [w for w in song["suspect"] if w["annotation_gap"]]
        expected = old[song["id"]]["recognition"]["hallucinated"]
        if [w["word"].strip() for w in candidates] != expected:
            raise ValueError(f"diagnostic does not reproduce the baseline: {song['id']}")
        info = evidence[song["id"]]
        refs = [line for item in info["evidence"] for line in item["revised_lines"]]
        relevant = [r for r in reviews if r["id"] == song["id"]]
        classified = []
        for word in candidates:
            category = classify(word, refs, relevant, info["audio_sha256"])
            counts[category] += 1
            classified.append(
                {"start": word["start"], "end": word["end"], "classification": category}
            )
        rows.append(
            {
                "id": song["id"],
                "audio_sha256": info["audio_sha256"],
                "candidates": classified,
                "revised_vocal_intervals": [[r["start"], r["end"]] for r in refs],
            }
        )
    legacy_total = sum(
        len(s.get("recognition", {}).get("hallucinated", [])) for s in baseline["songs"]
    )
    if sum(counts.values()) != legacy_total:
        raise ValueError("not all original candidates were audited")
    dev_candidates = sum(w["annotation_gap"] for s in development["songs"] for w in s["suspect"])
    instrumentals = [s for s in development["songs"] if s["id"].startswith("instr-")]
    dev = {
        "entries": len(development["songs"]),
        "gap_candidates": dev_candidates,
        "instrumentals": len(instrumentals),
        "instrumentals_without_words": sum(s["stage_words"] == 0 for s in instrumentals),
        "outside_vad_words": sum(
            w["outside_vad"] for s in development["songs"] for w in s["suspect"]
        ),
    }
    gate = None if counts["unresolved"] else counts["confirmed_hallucination"] == 0
    report = {
        "protocol": "Reference-supported vocals and explicit human reviews; ASR output unchanged.",
        "baseline_file": args.baseline.name,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "revised_reference_sha256": crosscheck["reference_sha256"],
        "source": "https://huggingface.co/datasets/jamendolyrics/jam-alt/blob/main/metadata.jsonl",
        "legacy_gap_candidates": legacy_total,
        "classifications": counts,
        "no_hallucination_in_reviewed_gaps": gate,
        "independent_development": dev,
        "songs": rows,
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [args.diagnostic, args.crosscheck, args.development, args.reviews]
        },
        "notes": "No numeric reduction is an ASR improvement. Model agreement is supporting "
        "evidence, not a listening verdict. No word blacklist or song-specific output filter.",
    }
    if args.candidate_crosscheck:
        candidate = json.loads(args.candidate_baseline.read_text())
        candidate_evidence = json.loads(args.candidate_crosscheck.read_text())
        candidate_rows = []
        for song in candidate_evidence["songs"]:
            words = [w for e in song["evidence"] for w in e["suspect_words"]]
            refs = [r for e in song["evidence"] for r in e["revised_lines"]]
            supported = sum(
                classify(w, refs, [], song["audio_sha256"]) == "reference_supported_vocal"
                for w in words
            )
            relevant = [r for r in reviews if r["id"] == song["id"]]
            categories = [classify(w, refs, relevant, song["audio_sha256"]) for w in words]
            candidate_rows.append(
                {
                    "id": song["id"],
                    "repeat_candidates": len(words),
                    "reference_supported": supported,
                    "confirmed_hallucinations": categories.count("confirmed_hallucination"),
                    "unresolved": categories.count("unresolved"),
                }
            )
        report["historical_energy_candidate"] = {
            "baseline_file": args.candidate_baseline.name,
            "baseline_sha256": hashlib.sha256(args.candidate_baseline.read_bytes()).hexdigest(),
            "crosscheck_sha256": hashlib.sha256(args.candidate_crosscheck.read_bytes()).hexdigest(),
            "historical_candidates": sum(
                len(s.get("recognition", {}).get("hallucinated", [])) for s in candidate["songs"]
            ),
            "repeat_candidates": sum(s["repeat_candidates"] for s in candidate_rows),
            "reference_supported": sum(s["reference_supported"] for s in candidate_rows),
            "confirmed_hallucinations": sum(s["confirmed_hallucinations"] for s in candidate_rows),
            "unresolved": sum(s["unresolved"] for s in candidate_rows),
            "songs": candidate_rows,
            "notes": "Diagnostic repeat differs from the historical unseeded run; "
            "not a reclassification of its exact words. No candidate default change.",
        }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# 第 5 步：未标注区间词的证据核查",
        "",
        "旧报告把歌词标注空隙当成纯伴奏，将其中的识别词直接称作幻听。这一推断不成立。"
        "本报告保留全部原始计数，不改变识别结果或模型参数，也不删除哼唱词。",
        "",
        f"原始待核验词 **{legacy_total}** 个："
        f"官方修订人声标注支持 **{counts['reference_supported_vocal']}** 个，"
        f"人工确认真实人声 **{counts['human_supported_vocal']}** 个，"
        f"确认幻听 **{counts['confirmed_hallucination']}** 个，"
        f"仍待核验 **{counts['unresolved']}** 个。",
        "",
        "| 匿名样本 | 时间（秒） | 核查依据 |",
        "|---|---|---|",
    ]
    labels = {
        "reference_supported_vocal": "官方修订歌词在对应时间标注了人声",
        "human_supported_vocal": "人工确认有真实人声；不是纯伴奏",
        "confirmed_hallucination": "人工确认无真实人声，识别输出属于幻听",
        "unresolved": "待人工听辨；无提示 CTC 也识别出相近说话内容，但不能替代耳听",
    }
    for song in rows:
        if song["candidates"]:
            for category in dict.fromkeys(w["classification"] for w in song["candidates"]):
                group = [w for w in song["candidates"] if w["classification"] == category]
                lines.append(
                    f"| {song['id']} | {group[0]['start']:.2f}–{group[-1]['end']:.2f} | "
                    f"{labels[category]}（{len(group)} 词） |"
                )
    lines += [
        "",
        "官方来源：[Jam-ALT 说明](https://audioshake.github.io/jam-alt/)及"
        "[行级标注](https://huggingface.co/datasets/jamendolyrics/jam-alt/blob/main/metadata.jsonl)。"
        "该数据修订了背景人声和非词语发声；原始音频与标注校验值写入 JSON。",
        "",
        "交叉检查使用现有 wav2vec2 英文模型贪心解码，未提供任何候选文本，分别读取原混音和人声轨。"
        "人声轨只在内存中使用；不同模型的一致结果不能自行提升为人工核验。",
        "",
        f"独立开发检查：{dev['entries']} 个条目；未标注区间词 {dev['gap_candidates']} 个，"
        f"区间外词 {dev['outside_vad_words']} 个；"
        f"纯音乐 {dev['instrumentals_without_words']}/{dev['instrumentals']} 无识别词。",
        "",
        "本审计不改变被审计的历史识别输出，也不将分类更正宣称为模型准确率提高。",
        "被审计版本的验收状态："
        + (
            "待人工核验，不能以 0 个已确认幻听宣称全部通过。"
            if gate is None
            else "本次候选全部核验，未发现纯伴奏幻听。"
            if gate
            else "确认存在纯伴奏幻听，仍未通过。"
        ),
    ]
    if "historical_energy_candidate" in report:
        c = report["historical_energy_candidate"]
        lines += [
            "",
            "## 历史能量候选的补查",
            "",
            f"历史候选报告为 {c['historical_candidates']} 个未标注区间词；"
            f"对其涉及的 {len(c['songs'])} 首歌曲重新解码得到 {c['repeat_candidates']} 个，"
            f"其中 {c['reference_supported']} 个词的时间点有官方人声标注覆盖，"
            f"人工确认幻听 {c['confirmed_hallucinations']} 个，仍待核验 {c['unresolved']} 个。"
            "本次不是历史输出的逐词复现，不能用新计数替换旧计数，也不能据旧计数认定全部为幻听。",
            "",
            "本次未改识别参数。未固定随机种子的温度回退可能产生重复差异；"
            "[逐曲种子抽查](step-5-asr-seed-check.md)单独记录。",
            "[独立 50 条开发集比较](step-5-asr-audit-dev.md)显示能量候选降低了各组字 / 词错率，"
            "两种配置均保持开发纯音乐 10/10 无词；这是单次开发检查，不代替真实歌曲的未决听辨。",
        ]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
