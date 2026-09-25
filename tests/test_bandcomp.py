# -*- coding: utf-8 -*-
"""Contract tests for the window-component (bandcomp) partitioner."""
import unittest

from scheduler import (
    GraphModel, _task_graph_ok, bandcomp_partition, plan_to_json)


def op(op_id, pipe="PIPE_V", cycles=1):
    return {"id": op_id, "op": "ADD", "pipe": pipe, "cycles": cycles}


def tensor(tensor_id, size=16, pos="UB"):
    return {"id": tensor_id, "size": size, "pos": pos}


def edge(source, target):
    return {"source": source, "target": target}


def chain(op_id, prev_id, size=16):
    """Tensors/edges linking prev -> op through a fresh tensor."""
    tid = 10000 + op_id
    return [tensor(tid, size), edge(prev_id, tid), edge(tid, op_id)]


def layered_graph(n_levels=12, width=4):
    """Independent vertical strands: width strands x n_levels deep."""
    ops, tensors, edges = [], [], []
    for s in range(width):
        for lvl in range(n_levels):
            oid = s * n_levels + lvl + 1
            ops.append(op(oid))
            if lvl > 0:
                tensors, edges = tensors, edges
                items = chain(oid, oid - 1)
                tensors.extend(items[0:1])
                edges.extend(items[1:])
    return {"ops": ops, "tensors": tensors, "edges": edges}


def layered_graph_full(n_levels=12, width=4):
    ops, tensors, edges = [], [], []
    for s in range(width):
        for lvl in range(n_levels):
            oid = s * n_levels + lvl + 1
            ops.append(op(oid))
            if lvl > 0:
                tid = 10000 + oid
                tensors.append(tensor(tid))
                edges.append(edge(oid - 1, tid))
                edges.append(edge(tid, oid))
    return {"ops": ops, "tensors": tensors, "edges": edges}


class BandcompTests(unittest.TestCase):
    def test_partition_is_acyclic_and_covers_all_ops(self):
        raw = layered_graph_full(n_levels=12, width=4)
        gm = GraphModel(raw)
        op_group, core_tasks = bandcomp_partition(gm, 2, window=3)
        self.assertEqual(set(gm.eligible), set(op_group),
                         "every eligible op must be assigned")
        self.assertTrue(_task_graph_ok(gm, op_group),
                        "window-component partition must be acyclic")
        scheduled = [t for order in core_tasks for t in order]
        self.assertEqual(set(op_group.values()), set(scheduled))
        self.assertEqual(len(scheduled), len(set(scheduled)),
                         "each component scheduled exactly once")

    def test_independent_strands_land_on_different_cores(self):
        raw = layered_graph_full(n_levels=12, width=4)
        gm = GraphModel(raw)
        op_group, core_tasks = bandcomp_partition(gm, 2, window=12)
        # window=12 covers the whole depth: 4 strand components in one
        # window; with k=2 cores each core must get >=1 strand
        self.assertEqual(2, len([o for o in core_tasks if o]))

    def test_plan_json_roundtrip(self):
        raw = layered_graph_full(n_levels=9, width=3)
        gm = GraphModel(raw)
        op_group, core_tasks = bandcomp_partition(gm, 3, window=4)
        plan = plan_to_json(op_group, core_tasks)
        self.assertEqual(len(plan['node_to_subgraph']), len(gm.eligible))
        sgids = set(plan['node_to_subgraph'].values())
        present = {t for order in plan['core_schedules'] for t in order}
        self.assertEqual(sgids, present)


if __name__ == "__main__":
    unittest.main()
