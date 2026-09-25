# -*- coding: utf-8 -*-
"""Validate collaborative leaderboard and experiment snapshots.

The default mode reports incomplete or conflicting snapshots as warnings so it
can be used during development. ``--strict`` turns those warnings into errors.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CELL_RE = re.compile(r"^case_\d{3}\|p[123]\|k[2345]$")


def read_json(path: Path, errors: list[str]):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{path.name}: invalid JSON: {exc}")
        return None


def validate_board(errors: list[str], warnings: list[str]) -> int:
    board_dir = RESULTS / "leaderboard"
    files = sorted(board_dir.glob("*.json")) if board_dir.exists() else []
    seen = 0
    for path in files:
        data = read_json(path, errors)
        if not isinstance(data, dict):
            continue
        if not isinstance(data.get("meta"), dict):
            errors.append(f"{path.name}: missing meta object")
        cells = data.get("cells")
        if not isinstance(cells, dict):
            errors.append(f"{path.name}: missing cells object")
            continue
        seen += len(cells)
        for key, rec in cells.items():
            if not CELL_RE.match(key):
                errors.append(f"{path.name}: malformed cell key {key!r}")
                continue
            if not isinstance(rec, dict):
                errors.append(f"{path.name}:{key}: record is not an object")
                continue
            if "error" in rec:
                continue
            ms = rec.get("makespan")
            sp = rec.get("speedup")
            if not isinstance(ms, (int, float)) or ms <= 0:
                errors.append(f"{path.name}:{key}: makespan must be positive")
            if not isinstance(sp, (int, float)) or sp <= 0:
                errors.append(f"{path.name}:{key}: speedup must be positive")
    if not files:
        warnings.append("no results/leaderboard/*.json files found")
    return seen


def validate_summaries(errors: list[str], warnings: list[str]) -> int:
    files = [RESULTS / "summary.json"] + sorted(RESULTS.glob("summary_*.json"))
    files = [p for p in files if p.exists()]
    seen: dict[tuple[str, str], str] = {}
    runs = 0
    for path in files:
        data = read_json(path, errors)
        if not isinstance(data, dict):
            continue
        for case, entry in data.items():
            if not isinstance(entry, dict):
                errors.append(f"{path.name}:{case}: entry is not an object")
                continue
            for key, record in entry.get("runs", {}).items():
                runs += 1
                marker = (case, key)
                if marker in seen:
                    warnings.append(
                        f"duplicate run {case}/{key}: {seen[marker]} and {path.name}"
                    )
                else:
                    seen[marker] = path.name
                best = record.get("best") if isinstance(record, dict) else None
                if not isinstance(best, dict):
                    continue
                hit_rate = best.get("cache_hit_rate")
                if hit_rate is not None and not (0 <= hit_rate <= 1):
                    warnings.append(
                        f"{path.name}:{case}/{key}: cache_hit_rate={hit_rate!r} outside [0, 1]"
                    )
    if not files:
        warnings.append("no summary*.json files found")
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    cells = validate_board(errors, warnings)
    runs = validate_summaries(errors, warnings)
    for item in errors:
        print(f"ERROR: {item}")
    for item in warnings:
        print(f"WARNING: {item}")
    if args.strict:
        errors.extend(warnings)
    print(f"checked board cells={cells}, summary runs={runs}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
