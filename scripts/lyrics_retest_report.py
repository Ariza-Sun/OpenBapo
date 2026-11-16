"""Summarize preserved baseline, energy candidate and regression verification.

Usage: python scripts/lyrics_retest_report.py [--directory docs/benchmarks]
All numbers come from preserved evaluator outputs; no manual score overrides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument("--gap-audit", type=Path, help="evidence review; raw counts are retained")
    parser.add_argument(
        "--local-regression", type=Path, help="latest frozen local-activity regression"
    )
    args = parser.parse_args()
    root = args.directory
    first_file = root / "step-5-lyrics-first-run.json"
    current_file = root / "step-5-lyrics-retest.json"
    first = json.loads(first_file.read_text())
    current = json.loads(current_file.read_text())
    candidate_file = root / "step-5-lyrics-energy-candidate.json"
    candidate = json.loads(candidate_file.read_text())
    protocol = json.loads((root / "step-5-retest-protocol.json").read_text())
    assert hashlib.sha256(first_file.read_bytes()).hexdigest() == protocol["first_run_sha256"]
    for key in ("code_sha256", "manifest_sha256"):
        assert current["meta"]["run_signature"][key] == protocol[key], key
    assert hashlib.sha256(candidate_file.read_bytes()).hexdigest() == protocol["candidate_sha256"]
    before, after = first["summary"], current["summary"]
    names = {
        "en-test/en/real": "英文真实歌曲",
        "zh-test/zh/dry": "中文干声",
        "zh-test/zh/mix": "中文合成伴奏",
        "zh-ref/zh/real": "中文 MIR-1K 参考",
    }
    alignment = {
        key: after[key]["align/wav2vec2"]["within"] >= 0.9
        for key in names
        if not key.startswith("zh-ref")
    }
    phantom_words = sum(after[key]["recognition"]["hallucinated_words"] for key in names)
    gap_audit = json.loads(args.gap_audit.read_text()) if args.gap_audit else None
    if gap_audit is not None:
        assert gap_audit["baseline_sha256"] == hashlib.sha256(current_file.read_bytes()).hexdigest()
        assert gap_audit["legacy_gap_candidates"] == phantom_words
    gap_gate = (
        gap_audit["no_hallucination_in_reviewed_gaps"]
        if gap_audit
        else True
        if phantom_words == 0
        else None
    )
    instrumental = after["instr/instrumental"]["no_lyrics_rate"] == 1.0
    acceptance = {
        "alignment": alignment,
        "instrumental_no_lyrics": instrumental,
        "no_hallucinations_in_reviewed_instrumental_gaps": gap_gate,
        "raw_gap_candidates": phantom_words,
        "all": all(alignment.values()) and instrumental and gap_gate is True,
    }
    lines = [
        "# 第 5 步歌词：首轮、候选与回归验证",
        "",
        "由 `scripts/lyrics_retest_report.py` 从保存的原始结果生成。",
        "[首轮明细](step-5-lyrics-first-run.md) · [复测明细](step-5-lyrics-retest.md) · "
        "[冻结协议](step-5-retest-protocol.json)",
        "",
        "本次包含首轮评测、能量补检候选评测、分离两种模式后的回归验证。"
        "后两轮使用同一验收集，因此最终数字属于回归验证，不是新盲测。"
        "没有根据验收歌曲调整阈值；自动识别恢复首轮的语音检测策略。",
        "",
        "## 行级对齐",
        "",
        "每行起始误差 ≤ 0.5 秒；前三组门槛为 90%，MIR-1K 只作参考。"
        "复测走实际 LyricsStage，缺失行计失败；首轮脚本绕过了部分无人声判断。",
        "",
        "| 集合 | 歌曲 / 片段数 | 行数 | 首轮 | 复测 | 复测缺失行 |",
        "|---|---|---|---|---|---|",
    ]
    for key, name in names.items():
        a, b = before[key]["align/wav2vec2"], after[key]["align/wav2vec2"]
        lines.append(
            f"| {name} | {after[key]['songs']} | {b['lines']} | {percentage(a['within'])} | "
            f"{percentage(b['within'])} | {b.get('missing', 0)} |"
        )
    lines += [
        "",
        "## 识别",
        "",
        "首轮中文 CER 计算在字之间插入空格，导致分母包含额外字符，数字被低估。"
        "原文件保留，但不能据此宣称中文识别变好或变差；复测按真实字数计算。"
        "英文 WER 的规范化规则未改变。",
        "",
        "歌词标注空隙不等于纯伴奏；以下词数保留旧脚本的原始计数，"
        "仅表示待核验候选，不能直接作为幻听数。",
        "",
        "| 集合 | 首轮字 / 词错率 | 复测字 / 词错率 | 首轮未标注区间词 | 复测未标注区间词 |",
        "|---|---|---|---|---|",
    ]
    for key, name in names.items():
        a, b = before[key]["recognition"], after[key]["recognition"]
        prior_rate = percentage(a["error_rate"]) if "/en/" in key else "口径有误，见原始报告"
        lines.append(
            f"| {name} | {prior_rate} | {percentage(b['error_rate'])} | "
            f"{a['hallucinated_words']} | {b['hallucinated_words']} |"
        )
    a, b = before["instr/instrumental"], after["instr/instrumental"]
    lines += [
        "",
        "## 纯音乐",
        "",
        f"首轮 {a['songs']} 首无歌词比例 {percentage(a['no_lyrics_rate'])}；"
        f"复测 {b['songs']} 首无歌词比例 {percentage(b['no_lyrics_rate'])}。",
        "",
        "## 历史候选保留与判断更正",
        "",
        "[能量补检候选原始结果](step-5-lyrics-energy-candidate.md)及"
        "[其冻结协议](step-5-energy-candidate-protocol.json)保留不变。"
        "该候选把能量补检用于两种模式，英文对齐达到 "
        f"{percentage(candidate['summary']['en-test/en/real']['align/wav2vec2']['within'])}，"
        "英文未标注区间的识别词从 "
        f"{before['en-test/en/real']['recognition']['hallucinated_words']} 增至 "
        f"{candidate['summary']['en-test/en/real']['recognition']['hallucinated_words']} 词，"
        "因此当时被搁置。后续核查发现原指标把未标注人声也计成幻听，"
        "不能仅凭 10 → 120 认定识别回退。原始文件保留；在证据核查完成前不切换默认方案。",
        "",
        "## 本轮变更与开发依据",
        "",
        "- 附歌词对齐使用 Silero 语音证据加持续能量补检，自动识别保持语音检测区间；"
        "[独立真实歌曲开发集](step-5-activity-real-dev.md) 共 59 首，与英文验收歌曲不重叠。",
        "- 完整人声轨 CTC 推理，VAD 区间外改为有限惩罚 2；"
        "[开发集惩罚比较](step-5-alignment-soft-dev.md) 先于本轮复测完成。",
        "- 评测使用实际流水线门槛，缺失预测仍计入分母，修正中文 CER。",
        "- 保存的回归结果中代码与数据签名匹配当时协议；首轮和历史候选的校验值不变。"
        "本次评测脚本已增加候选时间戳和逐曲随机种子，不冒充原冻结版本。",
        "- [4 分钟歌曲分析流水线计时](step-5-timing-retest.md)单独测量；"
        "评测逐首时间不作为生产延迟。该计时不包括排队、渲染及尚未实现的旋律环节。",
        "  表中首轮包含分析内部延迟加载，创建阶段对象时的 Whisper 加载在计时之外；"
        "因此不能将首轮合计当作完整进程冷启动耗时。",
        "",
        "## 未标注区间的证据核查",
        "",
    ]
    if gap_audit:
        c = gap_audit["classifications"]
        lines += [
            f"原始 {phantom_words} 个候选：官方人声标注解释 {c['reference_supported_vocal']} 个，"
            f"人工确认人声 {c['human_supported_vocal']} 个，"
            f"确认幻听 {c['confirmed_hallucination']} 个，待核验 {c['unresolved']} 个。",
            "没有修改识别输出；这属于评测判断更正，不能宣称模型识别率因此提高。"
            "[逐段证据与独立开发检查](step-5-gap-audit.md)。",
            "",
        ]
    else:
        lines += ["候选尚未完成证据核查，不将旧脚本的计数直接视为已确认幻听。", ""]
    lines += ["## 修复前的验收状态" if args.local_regression else "## 验收状态", ""]
    for key, passed in alignment.items():
        lines.append(f"- {names[key]}对齐 ≥ 90%：{'通过' if passed else '未通过'}。")
    lines.append(f"- 纯音乐全部无歌词：{'通过' if instrumental else '未通过'}。")
    lines.append(
        "- 纯伴奏间隔无幻听词："
        + ("待核验。" if gap_gate is None else "通过。" if gap_gate else "未通过。")
    )
    lines.append(
        "- 第 5 步质量验收全部通过。"
        if acceptance["all"]
        else "- 第 5 步尚未全部通过；待核验及未通过项仍保留，不将局部修复表述为完成整个阶段。"
    )
    result = {
        "first_run": first_file.name,
        "retest": current_file.name,
        "historical_candidate": candidate_file.name,
        "candidate": candidate["summary"],
        "gap_audit": gap_audit,
        "before": before,
        "after": after,
        "acceptance": acceptance,
        "frozen_code_sha256": protocol["code_sha256"],
        "first_run_sha256": protocol["first_run_sha256"],
        "notes": "首轮中文 CER 口径有误；复测使用实际 LyricsStage、缺失行计失败。"
        "能量补检候选曾按旧指标搁置，但未标注区间词不能直接视为幻听。"
        "分离模式后复用同一验收集做回归验证，非新盲测。",
    }
    if args.local_regression:
        latest = json.loads(args.local_regression.read_text())
        assert latest["baseline_sha256"] == hashlib.sha256(current_file.read_bytes()).hexdigest()
        result["previous_acceptance"] = result["acceptance"]
        result["acceptance"] = latest["acceptance"]
        result["current_regression"] = latest
        c = latest["gap_classifications"]
        latest_header = [
            "# 第 5 步歌词：当前回归与历史记录",
            "",
            "**当前结果**：[局部人声证据修复的完整回归](step-5-local-regression.md)。",
            f"{len(latest['songs'])} 条回归完成；新输出中确认幻听 "
            f"{c['confirmed_hallucination']} 个，待核验 {c['unresolved']} 个，"
            f"官方人声标注支持 {c['reference_supported_vocal']} 个。",
            "本次质量门槛：" + ("通过。" if latest["acceptance"]["all"] else "仍未全部通过。"),
            "该过滤可能遗漏孤立短演唱：独立真实开发歌曲中少覆盖 3.649 秒已标注人声。"
            "附歌词对齐不使用该过滤。",
            "",
            "以下各节保留修复前的历史数字和当时状态；当前验收以以上链接及 JSON 中的 "
            "`current_regression` 为准。",
            "",
        ]
        lines = latest_header + lines[2:]
    (root / "step-5-lyrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    (root / "step-5-lyrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
