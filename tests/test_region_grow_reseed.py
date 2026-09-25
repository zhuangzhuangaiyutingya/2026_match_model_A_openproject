# -*- coding: utf-8 -*-
"""Regression tests for region_grow's frontier-exhaustion reseeding.

Graphs built from independent chains used to leave ~all ops to the scatter
fallback (one chain per cluster, heap empties at chain end), which destroyed
balance and shredded tensors into cross-core copies (see case_020: 99% of ops
scattered, S=1.00 at every core count).  With reseeding, a cluster keeps
pulling new seeds until it reaches its work target.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import unittest

from scheduler import GraphModel, region_grow


COPY_TYPES = {"COPY_IN", "COPY_OUT"}


def chain_graph(n_chains, ops_per_chain, cycles=10):
    """n_chains independent chains, each ops_per_chain long."""
    ops, tensors, edges = [], [], []
    for c in range(n_chains):
        for i in range(ops_per_chain):
            op_id = c * ops_per_chain + i + 1
            ops.append({"id": op_id, "op": "ADD", "pipe": "PIPE_V",
                        "cycles": cycles})
            if i > 0:
                prev = op_id - 1
                tid = 10000 + op_id
                tensors.append({"id": tid, "size": 64, "pos": "UB"})
                edges.append({"source": prev, "target": tid})
                edges.append({"source": tid, "target": op_id})
    return {"ops": ops, "tensors": tensors, "edges": edges}


class RegionGrowReseedTests(unittest.TestCase):
    def test_chain_graph_balance_and_no_scatter(self):
        raw = chain_graph(n_chains=6, ops_per_chain=6, cycles=10)
        gm = GraphModel(raw)
        # 6 chains x 60 cycles = 360 total; k=2 target = 360*1.3/2 = 234.
        op_cluster = region_grow(gm, 2)
        works = {}
        for op, cl in op_cluster.items():
            works[cl] = works.get(cl, 0) + gm.eff_work[op]
        self.assertEqual(set(works), {0, 1}, "both clusters must be used")
        total = sum(works.values())
        self.assertLessEqual(max(works.values()), 0.7 * total,
                             "one cluster must not swallow the graph")
        self.assertGreaterEqual(min(works.values()), 0.25 * total,
                                "balance must not degenerate to a sliver")
        # spatial coherence: every chain lives in exactly one cluster
        for c in range(6):
            chain_ops = [c * 6 + i + 1 for i in range(6)]
            clusters = {op_cluster[op] for op in chain_ops}
            self.assertEqual(
                1, len(clusters),
                f"chain {c} was split across clusters {clusters}")

    def test_chain_graph_single_cluster_per_chain_no_cut_edges(self):
        raw = chain_graph(n_chains=4, ops_per_chain=5, cycles=7)
        gm = GraphModel(raw)
        op_cluster = region_grow(gm, 2)
        cut = 0
        for (u, v), tids in gm.pair_tensors.items():
            if tids and op_cluster.get(u) != op_cluster.get(v):
                cut += 1
        # 允许每簇前沿最多截断一条链（目标额触发的正常行为），
        # 但独立链图绝不允许出现 O(n) 级别的散射切边。
        self.assertLessEqual(
            cut, 2,
            "independent chains must not be scattered (cut=%d)" % cut)

    def test_reseed_terminates_on_fully_isolated_ops(self):
        ops = [{"id": i, "op": "ADD", "pipe": "PIPE_V", "cycles": 5}
               for i in range(1, 21)]
        raw = {"ops": ops, "tensors": [], "edges": []}
        gm = GraphModel(raw)
        op_cluster = region_grow(gm, 3)
        self.assertEqual(len(op_cluster), 20, "every op must be assigned")
        works = {}
        for op, cl in op_cluster.items():
            works[cl] = works.get(cl, 0) + gm.eff_work[op]
        self.assertEqual(3, len(works),
                         "all clusters must receive work via reseeding")
        total = sum(works.values())
        self.assertLessEqual(max(works.values()), 0.5 * total,
                             "no cluster may swallow the graph")


    def test_reseed_false_keeps_legacy_frontier_stop(self):
        """grow_ns 保留旧行为：前沿耗尽即止，链图退化为每簇一条链。"""
        raw = chain_graph(n_chains=6, ops_per_chain=6, cycles=10)
        gm = GraphModel(raw)
        op_cluster = region_grow(gm, 2, reseed=False)
        works = {}
        for op, cl in op_cluster.items():
            works[cl] = works.get(cl, 0) + gm.eff_work[op]
        # 旧行为：两簇各长一条链（60/360），其余 240 全部散射兜底
        grown = sum(w for w in works.values() if w <= 120)
        self.assertLessEqual(grown, 120,
                             "legacy mode must stop clusters at frontier exhaustion")
        reseeded = region_grow(gm, 2)
        works2 = {}
        for op, cl in reseeded.items():
            works2[cl] = works2.get(cl, 0) + gm.eff_work[op]
        self.assertGreater(min(works2.values()), min(works.values()),
                           "reseed mode must fill clusters beyond the legacy stop")


if __name__ == "__main__":
    unittest.main()
