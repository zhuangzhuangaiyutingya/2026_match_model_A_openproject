# -*- coding: utf-8 -*-
"""Compute per-case theoretical makespan lower bounds and attainable speedups.

For k cores the bound is
    LB_k = max(W_M / k, W_V / k, CP, B_DDR / BW)
where W_M/W_V are total compute cycles on matrix/vector pipes, CP is the
compute-only weighted critical path of the contracted non-COPY DAG, and B_DDR
is the original unavoidable input/output traffic. Components may overlap, so
max (rather than sum) preserves the lower-bound interpretation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from collab_io import atomic_write_json
from scheduler import BW, GraphModel

HERE = Path(__file__).resolve().parent
ATT = HERE.parent
DATA = ATT / "data"
RESULTS = HERE / "results"


def original_copy_bytes(graph):
    ops = {op["id"]: op for op in graph["ops"]}
    tensors = {tensor["id"]: tensor for tensor in graph["tensors"]}
    in_tids, out_tids = {}, {}
    for op_id in ops:
        in_tids[op_id] = []
        out_tids[op_id] = []
    for edge in graph["edges"]:
        src, dst = edge["source"], edge["target"]
        if src in ops and dst in tensors:
            out_tids[src].append(dst)
        elif src in tensors and dst in ops:
            in_tids[dst].append(src)
    total = 0
    for op_id, op in ops.items():
        if op.get("op") == "COPY_IN":
            total += sum(tensors[tid]["size"] for tid in out_tids[op_id])
        elif op.get("op") == "COPY_OUT":
            total += sum(tensors[tid]["size"] for tid in in_tids[op_id])
    return total


def compute_critical_path(gm):
    end = {}
    for op_id in gm.topo:
        start = max((end[pred] for pred in gm.preds[op_id]), default=0.0)
        end[op_id] = start + gm.eff_work[op_id]
    return max(end.values(), default=0.0)


def lower_bound(gm, num_cores):
    m_work = sum(gm.m_work.values())
    v_work = sum(gm.v_work.values())
    cp = compute_critical_path(gm)
    copy_bytes = original_copy_bytes(gm.raw)
    components = {
        "matrix_work_bound": m_work / num_cores,
        "vector_work_bound": v_work / num_cores,
        "critical_path_bound": cp,
        "ddr_bandwidth_bound": copy_bytes / BW,
    }
    value = max([1.0] + list(components.values()))
    return value, components, copy_bytes


def load_singlecore_baselines():
    from leaderboard import merged_summary

    result = {}
    for case, entry in merged_summary().items():
        makespan = entry.get("singlecore", {}).get("makespan")
        if makespan:
            result[case] = makespan
    return result


def build(cases=None):
    baselines = load_singlecore_baselines()
    case_names = cases or sorted(baselines)
    result = {
        "definition": "LB_k=max(W_M/k,W_V/k,CP,B_DDR/60); attainment=LB_k/makespan",
        "bandwidth_bytes_per_cycle": BW,
        "cases": {},
    }
    for case in case_names:
        path = DATA / f"{case}.json"
        if not path.exists() or case not in baselines:
            continue
        graph = json.loads(path.read_text(encoding="utf-8"))
        gm = GraphModel(graph)
        record = {
            "singlecore_makespan": baselines[case],
            "eligible_ops": len(gm.eligible),
            "cores": {},
        }
        for k in (2, 3, 4, 5):
            lb, components, copy_bytes = lower_bound(gm, k)
            record["cores"][str(k)] = {
                "lower_bound": round(lb, 6),
                "ideal_speedup": round(baselines[case] / lb, 6),
                "original_copy_bytes": copy_bytes,
                **{name: round(value, 6) for name, value in components.items()},
            }
        result["cases"][case] = record
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*")
    ap.add_argument(
        "--output", default=str(RESULTS / "theoretical_bounds.json")
    )
    args = ap.parse_args()
    result = build(args.cases)
    atomic_write_json(Path(args.output), result)
    print(f"wrote {args.output}: cases={len(result['cases'])}")


if __name__ == "__main__":
    main()
