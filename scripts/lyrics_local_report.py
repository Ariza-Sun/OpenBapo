"""Combine frozen regression shards and classify every new gap candidate by evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eval_lyrics import summarise
from lyrics_gap_review import classify


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    timing = json.loads(args.timing.read_text())
    assert timing["lyrics"] and 230 <= timing["song"]["duration_seconds"] <= 250
    measured = timing["results"][0]["warm_median"]
    budget_pass = measured["total"] < timing["analysis_budget_seconds"]
    protocol = json.loads(args.protocol.read_text())
    assert digest(args.manifest) == protocol["manifest_sha256"]
    for p in (args.baseline, args.reviews, args.development):
        assert digest(p) == protocol["input_sha256"][p.name]
    expected = {
        s["id"]: s
        for s in json.loads(args.manifest.read_text())["entries"]
        if s["split"] in ("test", "ref")
    }
    rows = []
    for path in args.runs:
        run = json.loads(path.read_text())
        signature = run["meta"]["run_signature"]
        for key in ("code_sha256", "manifest_sha256"):
            assert signature[key] == protocol[key]
        assert signature["random_seed"] == protocol["params"]["random_seed"]
        assert signature["recognition"] and signature["aligners"] == ["wav2vec2"]
        assert run["meta"]["activity_params"]["context_gap_s"] == 30
        assert run["meta"]["alignment_activity_params"]["context_gap_s"] is None
        rows.extend(run["songs"])
    assert len({s["id"] for s in rows}) == len(rows) == len(expected)
    assert {s["id"] for s in rows} == expected.keys()
    before = json.loads(args.baseline.read_text())["summary"]
    summary = summarise(rows, ["wav2vec2"])
    reference_audit = json.loads(args.references.read_text())
    refs = {s["id"]: s for s in reference_audit["songs"]}
    reviews = json.loads(args.reviews.read_text())["reviews"]
    reviewed = []
    for song in rows:
        candidates = song.get("recognition", {}).get("gap_candidates", [])
        assert len(candidates) == len(song.get("recognition", {}).get("hallucinated", []))
        for word in candidates:
            path = args.manifest.parent / expected[song["id"]]["audio"]
            audio_hash = digest(path)
            info = refs.get(song["id"])
            if info is not None:
                assert audio_hash == info["audio_sha256"]
            intervals = (
                [{"start": a, "end": b} for a, b in info["revised_vocal_intervals"]] if info else []
            )
            category = classify(
                word, intervals, [r for r in reviews if r["id"] == song["id"]], audio_hash
            )
            reviewed.append(
                {"id": song["id"], "start": word["start"], "end": word["end"], "category": category}
            )
    counts = {
        category: sum(w["category"] == category for w in reviewed)
        for category in (
            "reference_supported_vocal",
            "human_supported_vocal",
            "confirmed_hallucination",
            "unresolved",
        )
    }
    names = {
        "en-test/en/real": "英文真实歌曲",
        "zh-test/zh/dry": "中文干声",
        "zh-test/zh/mix": "中文合成伴奏",
        "zh-ref/zh/real": "MIR-1K 参考",
    }
    alignment = {
        key: summary[key]["align/wav2vec2"]["within"] >= 0.9
        for key in names
        if not key.startswith("zh-ref")
    }
    gap_pass = counts["confirmed_hallucination"] == 0 and counts["unresolved"] == 0
    instrumental = summary["instr/instrumental"]["no_lyrics_rate"] == 1.0
    acceptance = {
        "alignment": alignment,
        "instrumental_no_lyrics": instrumental,
        "no_hallucinations_in_reviewed_instrumental_gaps": gap_pass,
        "raw_gap_candidates": len(reviewed),
        "analysis_budget": budget_pass,
        "all": all(alignment.values()) and instrumental and gap_pass and budget_pass,
    }
    report = {
        "protocol": protocol,
        "input_sha256": {
            p.name: digest(p) for p in [*args.runs, args.references, args.reviews, args.timing]
        },
        "timing": measured,
        "before": before,
        "summary": summary,
        "songs": rows,
        "gap_classifications": counts,
        "reviewed_candidates": reviewed,
        "acceptance": acceptance,
        "baseline_sha256": digest(args.baseline),
        "notes": "Regression on previously inspected test data, not a new blind test. "
        "Local activity filtering has documented losses of real isolated vocals on dev data. "
        "The baseline was unseeded; small ASR differences cannot all be attributed to the filter.",
    }
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# 第 5 步：人工确认幻听后的回归结果",
        "",
        f"完整运行 {len(rows)} 条验收 / 参考数据，参数与代码校验值匹配冻结协议。"
        "这是已查看过的验收集上的回归，不是新盲测。",
        "中文 / 参考 / 纯音乐分片在累计 43 条、51 条后遇到模型初始化显存不足，"
        "按同一签名使用检查点续跑两次，保留全部已完成条目。"
        "长批次评测仍可能需要断点续跑。",
        "",
        "| 集合 | 条目 | 对齐达标率（前 / 后） | 字 / 词错率（前 / 后） | "
        "未标注区间词（前 / 后） |",
        "|---|---|---|---|---|",
    ]
    for key, name in names.items():
        old, new = before[key], summary[key]
        a, b = old["recognition"], new["recognition"]
        lines.append(
            f"| {name} | {new['songs']} | {old['align/wav2vec2']['within']:.1%} / "
            f"{new['align/wav2vec2']['within']:.1%} | "
            f"{a['error_rate']:.1%} / {b['error_rate']:.1%} | "
            f"{a['hallucinated_words']} / {b['hallucinated_words']} |"
        )
    inst = summary["instr/instrumental"]
    lines += [
        "",
        f"纯音乐无歌词：{inst['no_lyrics_rate']:.0%}（{inst['songs']} 首）。",
        "",
        f"新结果的 {len(reviewed)} 个未标注区间词全部保留：官方人声标注支持 "
        f"{counts['reference_supported_vocal']} 个、"
        f"人工确认人声 {counts['human_supported_vocal']} 个、"
        f"确认幻听 {counts['confirmed_hallucination']} 个、未解决 {counts['unresolved']} 个。",
        "用户已确认原 en-test-07 的 181–183 秒无说话声；人工记录绑定原音频校验值，"
        "不因模型一致而覆盖人工结论。",
        "",
        "当前自动识别要求相隔 ≥30 秒的每组检测区间各自满足至少 5 秒累计人声；"
        "附歌词对齐保留原有流程。原评测未固定随机种子，新评测逐曲固定种子，"
        "不能将字 / 词错率的全部变化都归因于过滤。",
        "",
        "开发检查明确记录了代价：59 首独立真实歌曲中，3 首合计少覆盖 3.649 秒人声，"
        "帧召回率下降 0.055 个百分点；原 50 条中英文及纯音乐开发输入区间不变。"
        "真实的短句、呼喊或孤立演唱也可能被过滤。"
        "[开发检查](step-5-local-activity-dev.md) · [冻结协议](step-5-local-protocol.json)。",
        "",
        f"4 分钟歌曲三次热运行中位数：歌词 {measured['lyrics']:.1f} 秒，"
        f"分析合计 {measured['total']:.1f} 秒。"
        "[计时明细](step-5-timing-local.md)不包含排队、渲染或未实现的旋律；"
        "Whisper 对象加载在计时之外，首轮也不是完整进程冷启动时间。"
        "计时期间另一张 GPU 上仍有评测任务，主机负载可能影响数字。",
        "",
        "本次回归的第 5 步质量门槛：" + ("通过。" if acceptance["all"] else "仍未全部通过。"),
        "通过只适用于本次回归与已审核的候选，不能推断任意歌曲都没有幻听或漏词。",
    ]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
