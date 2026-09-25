# -*- coding: utf-8 -*-
"""Generate the method-contribution descriptive report from shard labels.

This is a coverage analysis of which method family currently holds each cell.
It is NOT a controlled ablation: family assignment follows the recorded winning
label of each cell. Regenerate after any leaderboard refresh.
"""
import json
import statistics as st
import subprocess
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHARD = HERE / "results" / "leaderboard" / "baseline.json"
OUT = HERE / "report" / "method_contribution.md"

FAMILY_ORDER = [
    "深度窗口条带(bandcomp)", "条带+精化(bandref)",
    "参数族(grow/grow_ns)", "横切(level)", "波前条带(wave)",
    "宽图分摊(wide)", "fork-join lane", "单核保底",
    "早期未记label",
]


def family_of(label):
    label = label or ""
    if "fallback" in label:
        return "单核保底"
    if "fork" in label:
        return "fork-join lane"
    if "bandref" in label:
        return "条带+精化(bandref)"
    if "bandcomp" in label:
        return "深度窗口条带(bandcomp)"
    if "wave" in label:
        return "波前条带(wave)"
    if "wide" in label:
        return "宽图分摊(wide)"
    if "level" in label:
        return "横切(level)"
    if "grow" in label or "growcnt" in label or "growmv" in label:
        return "参数族(grow/grow_ns)"
    if "np" in label or label.startswith("v1") or label.startswith("v2"):
        return "参数族(grow/grow_ns)"
    return "早期未记label"


def main():
    board = json.loads(SHARD.read_text(encoding="utf-8"))["cells"]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(HERE),
            text=True).strip()
    except Exception:                                  # noqa: BLE001
        commit = "unknown"
    hold = defaultdict(list)        # family -> [(key, speedup)]
    grid = defaultdict(lambda: defaultdict(int))   # (p,k) -> family -> count
    for key, rec in board.items():
        fam = family_of(rec.get("label"))
        hold[fam].append((key, rec["speedup"]))
        case, p, k = key.split("|")
        grid[(int(p[1:]), int(k[1:]))][fam] += 1

    lines = [
        "# 方法贡献统计（描述性覆盖分析）", "",
        f"> 生成：{time.strftime('%Y-%m-%d %H:%M:%S')} · commit `{commit}` · "
        "数据源：`results/leaderboard/baseline.json`", "",
        "**口径说明**：本表统计每个方法族当前持有（逐格 Makespan 最优）的格子，"
        "是方法覆盖分析；**不是**关闭某组件重跑的对照消融。"
        "“早期未记label”指 v3 时期合并进榜、当时未记录获胜标签的格子，"
        "其评估时间早于结构候选存在，故归于参数族时代。", "",
        "## 1. 各方法族总体持有", "",
        "| 方法族 | 持有格数 | 平均加速比 | 中位加速比 | 最佳格 |",
        "|---|---:|---:|---:|---|",
    ]
    for fam in FAMILY_ORDER:
        entries = hold.get(fam, [])
        if not entries:
            lines.append(f"| {fam} | 0 | — | — | — |")
            continue
        speeds = [s for _, s in entries]
        best_key, best_speed = max(entries, key=lambda x: x[1])
        lines.append(
            f"| {fam} | {len(entries)} | {st.mean(speeds):.3f} "
            f"| {st.median(speeds):.3f} | {best_key} ({best_speed:.2f}x) |")

    lines += ["", "## 2. 分问题×核数的方法族持有格数", ""]
    header = ["方法族"] + [f"P{p}·{k}核" for p in (1, 2, 3) for k in (2, 3, 4, 5)]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for fam in FAMILY_ORDER:
        row = [fam] + [str(grid[(p, k)].get(fam, 0))
                       for p in (1, 2, 3) for k in (2, 3, 4, 5)]
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "", "## 3. 各方法族代表性用例（该族持有格中加速比前 3）", "",
    ]
    for fam in FAMILY_ORDER:
        entries = sorted(hold.get(fam, []), key=lambda x: -x[1])[:3]
        if not entries:
            continue
        lines.append(f"- **{fam}**：" + " · ".join(
            f"{key} ({speed:.2f}x)" for key, speed in entries))

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"method contribution report -> {OUT}")
    for fam in FAMILY_ORDER:
        entries = hold.get(fam, [])
        if entries:
            print(f"{fam}: {len(entries)} 格, 平均 {st.mean(s for _, s in entries):.3f}x")


if __name__ == "__main__":
    main()
