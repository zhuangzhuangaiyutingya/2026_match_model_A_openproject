# -*- coding: utf-8 -*-
"""Contract tests for the structural scheduling helpers.

The fixtures deliberately use tiny graphs so failures point at graph semantics
rather than at a heuristic quality choice.  The small compatibility adapters
accept the natural public representations (attributes or mapping fields) while
keeping the assertions on the documented behaviour strict.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import inspect
import unittest

from scheduler import GraphModel
from structural_scheduler import (
    StructuralGraph,
    fork_join_plan,
    task_graph_acyclic,
    wavefront_plan,
    wide_plan,
)


COPY_TYPES = {"COPY_IN", "COPY_OUT"}


def op(op_id, op_type="ADD", pipe="PIPE_V", cycles=1):
    return {"id": op_id, "op": op_type, "pipe": pipe, "cycles": cycles}


def tensor(tensor_id, size=16, pos="UB"):
    return {"id": tensor_id, "size": size, "pos": pos}


def edge(source, target):
    return {"source": source, "target": target}


def graph(ops, tensors=(), edges=()):
    return {"ops": list(ops), "tensors": list(tensors), "edges": list(edges)}


def structural_graph(raw_graph):
    """Build the structural view through the repository's canonical graph model."""
    return StructuralGraph(GraphModel(raw_graph))


def public_value(obj, *names):
    for candidate in (obj, getattr(obj, "gm", None)):
        if candidate is None:
            continue
        for name in names:
            if isinstance(candidate, dict) and name in candidate:
                return candidate[name]
            if hasattr(candidate, name):
                return getattr(candidate, name)
    raise AssertionError(
        "{} exposes none of {}".format(type(obj).__name__, ", ".join(names)))


def pair_map(value):
    """Normalize {(u,v): weight} or [(u,v,weight)] style edge collections."""
    if isinstance(value, dict):
        result = {}
        for key, weight in value.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise AssertionError("edge-weight keys must be (source, target) pairs")
            result[(int(key[0]), int(key[1]))] = weight
        return result

    result = {}
    for item in value:
        if isinstance(item, dict):
            source = item.get("source", item.get("src"))
            target = item.get("target", item.get("dst"))
            weight = item.get("bytes", item.get("weight", item.get("size", 0)))
        else:
            if len(item) == 2:
                source, target = item
                weight = 0
            elif len(item) == 3:
                source, target, weight = item
            else:
                raise AssertionError("edge entries must have two or three fields")
        result[(int(source), int(target))] = weight
    return result


def dependency_pairs(structural_graph):
    try:
        raw = public_value(
            structural_graph, "dependencies", "dependency_pairs", "dep_edges")
        if isinstance(raw, dict):
            return set(pair_map(raw))
        pairs = set()
        for item in raw:
            if isinstance(item, dict):
                pairs.add((int(item.get("source", item.get("src"))),
                           int(item.get("target", item.get("dst")))))
            else:
                pairs.add((int(item[0]), int(item[1])))
        return pairs
    except AssertionError:
        pass

    try:
        succs = public_value(structural_graph, "succs", "successors")
        return {(int(source), int(target))
                for source, targets in succs.items() for target in targets}
    except AssertionError:
        pass

    return set(pair_map(public_value(
        structural_graph, "communication", "communication_bytes", "comm_bytes")))


def communication_map(structural_graph):
    try:
        return pair_map(public_value(
            structural_graph, "communication", "communication_bytes", "comm_bytes",
            "edge_bytes"))
    except AssertionError:
        pass

    result = {}
    sizes = public_value(structural_graph, "tensor_size", "tensor_sizes")
    try:
        communication_edges = public_value(structural_graph, "communication_edges")
    except AssertionError:
        communication_edges = None
    if communication_edges is not None:
        for source, target, tensor_id in set(communication_edges):
            pair = (int(source), int(target))
            result[pair] = result.get(pair, 0) + sizes[tensor_id]
    else:
        pair_tensors = public_value(structural_graph, "pair_tensors")
        for pair, tensor_ids in pair_tensors.items():
            result[(int(pair[0]), int(pair[1]))] = sum(
                sizes[tensor_id] for tensor_id in set(tensor_ids))
    for pair in dependency_pairs(structural_graph):
        result.setdefault(pair, 0)
    return result


def tensor_positions(structural_graph):
    try:
        return public_value(
            structural_graph, "tensor_pos", "tensor_positions", "tensor_location",
            "tensor_locations")
    except AssertionError:
        return structural_graph.gm.tensor_pos


def call_task_graph_acyclic(structural_graph, grouping):
    """Support either task_graph_acyclic(graph, mapping) argument order."""
    try:
        parameters = list(inspect.signature(task_graph_acyclic).parameters)
    except (TypeError, ValueError):
        parameters = []
    if parameters and any(token in parameters[0].lower()
                          for token in ("group", "mapping", "partition")):
        return task_graph_acyclic(grouping, structural_graph)
    return task_graph_acyclic(structural_graph, grouping)


def call_plan(planner, structural_graph, n_parts):
    """Call planners across the intended n_parts/n_cores-compatible interface."""
    try:
        parameters = inspect.signature(planner).parameters
    except (TypeError, ValueError):
        parameters = {}
    arguments = {}
    for name in ("n_parts", "num_parts", "parts", "n_cores", "num_cores", "k"):
        if name in parameters:
            arguments[name] = n_parts
            break
    if "problem" in parameters:
        arguments["problem"] = 1
    graph_arg = structural_graph.gm if "gm" in parameters else structural_graph
    if arguments:
        return planner(graph_arg, **arguments)
    return planner(graph_arg, n_parts)


def mapping_from_plan(plan):
    if isinstance(plan, dict):
        for name in ("node_to_subgraph", "op_to_task", "op_group", "mapping"):
            if name in plan:
                raw = plan[name]
                return {int(node): int(group) for node, group in raw.items()}
        if plan and all(not isinstance(value, (dict, list, tuple, set))
                        for value in plan.values()):
            return {int(node): int(group) for node, group in plan.items()}
    for name in ("node_to_subgraph", "op_to_task", "op_group", "mapping"):
        if hasattr(plan, name):
            raw = getattr(plan, name)
            return {int(node): int(group) for node, group in raw.items()}
    if isinstance(plan, (tuple, list)) and plan and isinstance(plan[0], dict):
        return {int(node): int(group) for node, group in plan[0].items()}
    raise AssertionError("planner result does not expose an op-to-task mapping")


def schedules_from_plan(plan):
    if isinstance(plan, dict):
        for name in ("core_schedules", "schedules", "core_tasks"):
            if name in plan:
                return plan[name]
    for name in ("core_schedules", "schedules", "core_tasks"):
        if hasattr(plan, name):
            return getattr(plan, name)
    if isinstance(plan, (tuple, list)) and len(plan) > 1:
        return plan[1]
    raise AssertionError("planner result does not expose per-core schedules")


class StructuralGraphTests(unittest.TestCase):
    def test_non_convex_a_b_a_grouping_is_rejected(self):
        raw = graph(
            [op(1), op(2), op(3)],
            [tensor(101), tensor(102)],
            [edge(1, 101), edge(101, 2), edge(2, 102), edge(102, 3)],
        )
        structural = structural_graph(raw)

        self.assertFalse(
            call_task_graph_acyclic(structural, {1: 7, 2: 8, 3: 7}),
            "contracting A -> B -> A must reveal the task-level A <-> B cycle",
        )
        self.assertTrue(
            call_task_graph_acyclic(structural, {1: 7, 2: 7, 3: 8}))

    def test_shared_tensor_two_consumers_deduplicates_bytes_and_keeps_dependencies(self):
        raw = graph(
            [op(1), op(2), op(3)],
            [tensor(101, size=64)],
            [edge(1, 101), edge(101, 2), edge(101, 3)],
        )
        structural = structural_graph(raw)

        self.assertEqual({(1, 2), (1, 3)}, dependency_pairs(structural))
        communication = communication_map(structural)
        self.assertEqual(64, communication[(1, 2)])
        self.assertEqual(64, communication[(1, 3)])
        self.assertEqual(128, sum(communication.values()))

    def test_direct_op_edge_is_a_zero_byte_dependency(self):
        structural = structural_graph(graph([op(1), op(2)], edges=[edge(1, 2)]))

        self.assertIn((1, 2), dependency_pairs(structural))
        self.assertEqual(0, communication_map(structural).get((1, 2), 0))

    def test_dependency_is_contracted_through_copy_node(self):
        raw = graph(
            [op(1), op(9, "COPY_OUT", "PIPE_MTE3"), op(2)],
            [tensor(101, size=32), tensor(102, size=32)],
            [edge(1, 101), edge(101, 9), edge(9, 102), edge(102, 2)],
        )
        structural = structural_graph(raw)

        eligible = set(public_value(structural, "eligible", "eligible_ops", "op_ids"))
        self.assertEqual({1, 2}, eligible)
        self.assertIn((1, 2), dependency_pairs(structural))

    def test_external_l1_tensor_position_is_preserved(self):
        raw = graph(
            [op(1)],
            [tensor(101, size=48, pos="L1")],
            [edge(101, 1)],
        )
        structural = structural_graph(raw)

        positions = tensor_positions(structural)
        self.assertEqual("L1", positions[101])


class StructuralPlannerTests(unittest.TestCase):
    def setUp(self):
        # Two independent width-2 wavefronts which join at op 5.
        self.raw = graph(
            [op(node) for node in range(1, 6)],
            [tensor(101), tensor(102), tensor(103), tensor(104)],
            [
                edge(1, 101), edge(101, 3),
                edge(2, 102), edge(102, 4),
                edge(3, 103), edge(103, 5),
                edge(4, 104), edge(104, 5),
            ],
        )
        self.structural = structural_graph(self.raw)

    def assert_valid_plan(self, planner, n_parts):
        plan = call_plan(planner, self.structural, n_parts)
        mapping = mapping_from_plan(plan)
        eligible = set(public_value(
            self.structural, "eligible", "eligible_ops", "op_ids"))
        self.assertEqual(eligible, set(mapping),
                         "plan must cover every eligible op exactly once")
        self.assertTrue(call_task_graph_acyclic(self.structural, mapping),
                        "the contracted task graph must be acyclic")

        schedules = schedules_from_plan(plan)
        scheduled = [task for order in schedules for task in order]
        tasks = set(mapping.values())
        self.assertEqual(tasks, set(scheduled))
        self.assertEqual(len(tasks), len(scheduled),
                         "each task must be scheduled exactly once")

        core_of = {}
        position = {}
        for core, order in enumerate(schedules):
            for index, task in enumerate(order):
                core_of[task] = core
                position[task] = index
        for source, target in dependency_pairs(self.structural):
            source_task, target_task = mapping[source], mapping[target]
            if source_task != target_task and core_of[source_task] == core_of[target_task]:
                self.assertLess(
                    position[source_task], position[target_task],
                    "same-core schedules must respect task dependencies",
                )
        return plan, mapping

    def test_n_ops_equal_n_parts_produces_one_task_per_op(self):
        _, mapping = self.assert_valid_plan(wavefront_plan, 5)
        self.assertEqual(5, len(set(mapping.values())))

    def test_all_planners_cover_eligible_ops_and_emit_acyclic_legal_schedules(self):
        for planner in (wavefront_plan, fork_join_plan, wide_plan):
            with self.subTest(planner=planner.__name__):
                self.assert_valid_plan(planner, 3)

    def test_same_core_order_is_legal_on_a_chain(self):
        raw = graph(
            [op(1), op(2), op(3), op(4)],
            [tensor(101), tensor(102), tensor(103)],
            [edge(1, 101), edge(101, 2),
             edge(2, 102), edge(102, 3),
             edge(3, 103), edge(103, 4)],
        )
        structural = structural_graph(raw)
        for planner in (wavefront_plan, fork_join_plan, wide_plan):
            with self.subTest(planner=planner.__name__):
                old = self.structural
                self.structural = structural
                try:
                    self.assert_valid_plan(planner, 1)
                finally:
                    self.structural = old


if __name__ == "__main__":
    unittest.main()
