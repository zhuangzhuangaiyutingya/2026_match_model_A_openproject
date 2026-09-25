# -*- coding: utf-8 -*-
"""Structural scheduling candidates built on top of :mod:`scheduler`.

The routines in this module deliberately use only the public data cached by
``GraphModel``.  They do not modify the baseline scheduler and return the same
minimal plan format: ``node_to_subgraph`` plus ``core_schedules``.
"""
from collections import defaultdict, deque

from scheduler import GraphModel


class StructuralGraph:
    """A dependency-complete, tensor-aware view of an existing GraphModel.

    Dependency legality always comes from ``gm.preds``/``gm.succs``.  Tensor
    communication records are a separate annotation and are deduplicated by
    ``(source op, target op, tensor id)``; direct op-op dependencies therefore
    remain visible even when they do not carry a tensor id.
    """

    def __init__(self, gm):
        if not isinstance(gm, GraphModel):
            # Supporting GraphModel-compatible test doubles is useful, but a
            # malformed object should fail at the boundary with a clear error.
            required = ('eligible', 'preds', 'succs', 'topo', 'topo_pos',
                        'depth', 'm_work', 'v_work', 'pair_tensors')
            missing = [name for name in required if not hasattr(gm, name)]
            if missing:
                raise TypeError('gm is missing GraphModel fields: {}'.format(
                    ', '.join(missing)))
        self.gm = gm
        self.nodes = tuple(gm.topo)
        self.node_set = set(self.nodes)
        self.preds = {
            op: set(p for p in gm.preds.get(op, ()) if p in self.node_set)
            for op in self.nodes
        }
        self.succs = {
            op: set(s for s in gm.succs.get(op, ()) if s in self.node_set)
            for op in self.nodes
        }
        self.depth = dict(gm.depth)
        self.topo_pos = dict(gm.topo_pos)
        self.layers = defaultdict(list)
        for op in self.nodes:
            self.layers[self.depth[op]].append(op)
        for depth in self.layers:
            self.layers[depth].sort(key=self.topo_pos.__getitem__)
        self.max_depth = max(self.layers, default=-1)

        seen = set()
        communication_edges = []
        pair_tensors = defaultdict(list)
        for u, v in sorted(gm.pair_tensors,
                           key=lambda pair: (self.topo_pos.get(pair[0], -1),
                                             self.topo_pos.get(pair[1], -1))):
            if u not in self.node_set or v not in self.node_set:
                continue
            for tid in gm.pair_tensors[(u, v)]:
                key = (u, v, tid)
                if key in seen:
                    continue
                seen.add(key)
                communication_edges.append(key)
                pair_tensors[(u, v)].append(tid)
        self.communication_edges = tuple(communication_edges)
        self.pair_tensors = {
            pair: tuple(tids) for pair, tids in pair_tensors.items()
        }

    def task_graph_acyclic(self, mapping):
        """Return whether contracting nodes according to ``mapping`` is a DAG."""
        return task_graph_acyclic(self, mapping)


def _as_structural(graph):
    return graph if isinstance(graph, StructuralGraph) else StructuralGraph(graph)


def _validate_mapping(sg, mapping):
    if not isinstance(mapping, dict):
        raise TypeError('mapping must be a dict from eligible op id to group id')
    keys = set(mapping)
    if keys != sg.node_set:
        missing = sorted(sg.node_set - keys)
        extra = sorted(keys - sg.node_set)
        raise ValueError('mapping must cover eligible ops exactly; missing={} extra={}'
                         .format(missing[:20], extra[:20]))
    for op, group in mapping.items():
        if isinstance(group, bool) or not isinstance(group, int) or group < 0:
            raise ValueError('invalid group id for op {}: {!r}'.format(op, group))


def _task_adjacency(sg, mapping):
    groups = set(mapping.values())
    preds = {group: set() for group in groups}
    succs = {group: set() for group in groups}
    # Full contracted dependencies, not merely tensor-carrying communication.
    for u in sg.nodes:
        source = mapping[u]
        for v in sg.succs[u]:
            target = mapping[v]
            if source != target and target not in succs[source]:
                succs[source].add(target)
                preds[target].add(source)
    return preds, succs


def task_graph_acyclic(graph, mapping):
    """Check acyclicity of a node-to-task mapping using every dependency edge."""
    sg = _as_structural(graph)
    try:
        _validate_mapping(sg, mapping)
    except (TypeError, ValueError):
        return False
    preds, succs = _task_adjacency(sg, mapping)
    indegree = {group: len(preds[group]) for group in preds}
    ready = deque(group for group in sorted(preds) if indegree[group] == 0)
    visited = 0
    while ready:
        group = ready.popleft()
        visited += 1
        for nxt in sorted(succs[group]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    return visited == len(preds)


def _work(gm, op):
    return float(gm.m_work.get(op, 0)), float(gm.v_work.get(op, 0))


def _lpt_lanes(gm, ops, num_lanes, initial_loads=None, affinity=None):
    """Two-dimensional LPT minimizing max(M-load, V-load) on each lane."""
    if num_lanes <= 0:
        raise ValueError('num_lanes must be positive')
    loads = ([[0.0, 0.0] for _ in range(num_lanes)]
             if initial_loads is None
             else [[float(pair[0]), float(pair[1])] for pair in initial_loads])
    if len(loads) != num_lanes:
        raise ValueError('initial_loads length must equal num_lanes')
    assignment = {}
    ordered = sorted(
        ops,
        key=lambda op: (-max(_work(gm, op)), -sum(_work(gm, op)),
                        gm.topo_pos[op]),
    )
    for op in ordered:
        m_work, v_work = _work(gm, op)
        preferred = affinity.get(op) if affinity else None
        best_lane = min(
            range(num_lanes),
            key=lambda lane: (
                max(loads[lane][0] + m_work, loads[lane][1] + v_work),
                (loads[lane][0] + m_work) + (loads[lane][1] + v_work),
                0 if preferred == lane else 1,
                max(loads[lane]), sum(loads[lane]), lane,
            ),
        )
        assignment[op] = best_lane
        loads[best_lane][0] += m_work
        loads[best_lane][1] += v_work
    return assignment, loads


def _normalize_window(window):
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError('window must be a positive integer')
    return window


def _build_depth_chunks(sg, op_core, window):
    """Use the widest legal depth chunks, refining only cyclic intervals."""
    intervals = []
    start = 0
    while start <= sg.max_depth:
        intervals.append((start, min(start + window, sg.max_depth + 1)))
        start += window

    while True:
        interval_of_depth = {}
        for index, (start, stop) in enumerate(intervals):
            for depth in range(start, stop):
                interval_of_depth[depth] = index
        keys = sorted({(interval_of_depth[sg.depth[op]], op_core[op])
                       for op in sg.nodes})
        group_of_key = {key: index for index, key in enumerate(keys)}
        mapping = {
            op: group_of_key[(interval_of_depth[sg.depth[op]], op_core[op])]
            for op in sg.nodes
        }
        if task_graph_acyclic(sg, mapping):
            return mapping, group_of_key

        cyclic = _cyclic_groups(sg, mapping)
        bad_intervals = {
            interval_of_depth[sg.depth[op]]
            for op in sg.nodes if mapping[op] in cyclic
        }
        refined = []
        changed = False
        for index, (begin, end) in enumerate(intervals):
            if index not in bad_intervals or end - begin <= 1:
                refined.append((begin, end))
                continue
            middle = begin + (end - begin) // 2
            refined.extend(((begin, middle), (middle, end)))
            changed = True
        if not changed:
            raise ValueError('internal error: unit-depth grouping is cyclic')
        intervals = refined


def _cyclic_groups(sg, mapping):
    """Return task groups left after Kahn elimination."""
    preds, succs = _task_adjacency(sg, mapping)
    indegree = {group: len(preds[group]) for group in preds}
    ready = deque(group for group in preds if indegree[group] == 0)
    while ready:
        group = ready.popleft()
        for nxt in succs[group]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    return {group for group, degree in indegree.items() if degree > 0}


def _finalize_plan(sg, mapping, core_schedules):
    """Create the standard plan and perform strict internal validation."""
    _validate_mapping(sg, mapping)
    if not task_graph_acyclic(sg, mapping):
        raise ValueError('internal error: generated task graph is cyclic')
    if not isinstance(core_schedules, list) or not core_schedules:
        raise ValueError('core_schedules must be a non-empty list')
    groups = set(mapping.values())
    scheduled = [group for order in core_schedules for group in order]
    if len(scheduled) != len(set(scheduled)) or set(scheduled) != groups:
        raise ValueError('internal error: schedules do not cover groups exactly once')
    group_core = {
        group: core for core, order in enumerate(core_schedules) for group in order
    }
    group_pos = {
        group: pos for order in core_schedules
        for pos, group in enumerate(order)
    }
    for u in sg.nodes:
        source = mapping[u]
        for v in sg.succs[u]:
            target = mapping[v]
            if (source != target and group_core[source] == group_core[target]
                    and group_pos[source] >= group_pos[target]):
                raise ValueError('internal error: same-core order violates {} -> {}'
                                 .format(source, target))
    return {
        'node_to_subgraph': {
            str(op): int(mapping[op]) for op in sg.nodes
        },
        'core_schedules': [
            [int(group) for group in order] for order in core_schedules
        ],
    }


def plan_from_op_core(graph, op_core, num_cores=None, window=1):
    """Convert an op-to-core assignment into legal depth-window micrographs.

    Splitting at depth-window boundaries prevents a lane from becoming one
    disconnected, dependency-crossing task.  The resulting task graph is a DAG
    because every group belongs to exactly one monotone depth interval.
    """
    sg = _as_structural(graph)
    window = _normalize_window(window)
    if set(op_core) != sg.node_set:
        missing = sorted(sg.node_set - set(op_core))
        extra = sorted(set(op_core) - sg.node_set)
        raise ValueError('op_core must cover eligible ops exactly; missing={} extra={}'
                         .format(missing[:20], extra[:20]))
    if num_cores is None:
        num_cores = max(op_core.values(), default=-1) + 1
    if isinstance(num_cores, bool) or not isinstance(num_cores, int) or num_cores <= 0:
        raise ValueError('num_cores must be a positive integer')
    for op, core in op_core.items():
        if (isinstance(core, bool) or not isinstance(core, int)
                or core < 0 or core >= num_cores):
            raise ValueError('invalid core for op {}: {!r}'.format(op, core))

    # Larger windows can occasionally contract an interleaved diamond into a
    # task-level cycle (lane A -> lane B and lane B -> lane A).  Refine only the
    # offending depth intervals until contraction is safe; the requested
    # op-to-core placement is never changed.  Unit-depth groups are guaranteed
    # acyclic because every dependency strictly increases GraphModel.depth.
    mapping, group_of_key = _build_depth_chunks(sg, op_core, window)

    core_schedules = [[] for _ in range(num_cores)]
    for (depth_chunk, core), group in sorted(group_of_key.items()):
        del depth_chunk
        core_schedules[core].append(group)
    return _finalize_plan(sg, mapping, core_schedules)


def wavefront_plan(gm, num_cores, problem, window=1):
    """Depth-window wavefront scheduling with two-dimensional M/V LPT lanes."""
    sg = StructuralGraph(gm)
    window = _normalize_window(window)
    _validate_problem_and_cores(problem, num_cores)
    lane_of = {}
    for start in range(0, sg.max_depth + 1, window):
        ops = []
        for depth in range(start, min(start + window, sg.max_depth + 1)):
            ops.extend(sg.layers.get(depth, ()))
        assigned, _ = _lpt_lanes(gm, ops, num_cores)
        lane_of.update(assigned)
    return plan_from_op_core(sg, lane_of, num_cores=num_cores, window=window)


def _ancestor_lane_hints(sg, num_cores):
    """Propagate root/branch ancestry without materializing ancestor sets."""
    hints = {}
    next_root_lane = 0
    for op in sg.nodes:
        pred_lanes = [hints[p] for p in sg.preds[op] if p in hints]
        if not pred_lanes:
            hints[op] = next_root_lane % num_cores
            next_root_lane += 1
        else:
            counts = defaultdict(int)
            for lane in pred_lanes:
                counts[lane] += 1
            hints[op] = min(counts, key=lambda lane: (-counts[lane], lane))
    return hints


def fork_join_plan(gm, num_cores, problem, window=8):
    """Schedule repeated narrow fork/join stages by depth and branch ancestry.

    Non-join nodes prefer the lane inherited from their dominant predecessor.
    Join nodes prefer the lane carrying the largest amount of predecessor work;
    two-dimensional LPT remains the balancing authority within each layer.
    """
    sg = StructuralGraph(gm)
    window = _normalize_window(window)
    _validate_problem_and_cores(problem, num_cores)
    ancestry_hint = _ancestor_lane_hints(sg, num_cores)
    lane_of = {}
    for depth in range(sg.max_depth + 1):
        ops = list(sg.layers.get(depth, ()))
        affinity = {}
        for op in ops:
            predecessors = sg.preds[op]
            if not predecessors:
                affinity[op] = ancestry_hint[op]
                continue
            lane_weight = defaultdict(float)
            for pred in predecessors:
                lane = lane_of.get(pred, ancestry_hint[pred])
                lane_weight[lane] += max(_work(gm, pred)) or 1.0
            affinity[op] = min(
                lane_weight,
                key=lambda lane: (-lane_weight[lane], lane),
            )
        assigned, _ = _lpt_lanes(gm, ops, num_cores, affinity=affinity)
        lane_of.update(assigned)
    return plan_from_op_core(sg, lane_of, num_cores=num_cores, window=window)


def _tensor_sig(gm, tid):
    return gm.tensor_pos[tid], gm.tensor_size[tid]


def _output_sig(gm, op):
    tids = gm.prod_of_op.get(op, ())
    signatures = {_tensor_sig(gm, tid) for tid in tids}
    return next(iter(signatures)) if len(signatures) == 1 else None


def _merge_like(gm, op, required_sig=None):
    """Identify a reduction node from dependency and tensor-shape semantics."""
    preds = gm.preds[op]
    out_sig = _output_sig(gm, op)
    if len(preds) < 2 or out_sig is None:
        return False
    if required_sig is not None and out_sig != required_sig:
        return False
    incoming = [gm.pair_tensors.get((pred, op), ()) for pred in preds]
    return (all(incoming)
            and all(_tensor_sig(gm, tid) == out_sig
                    for tids in incoming for tid in tids))


def _parse_fork_join_stage(gm, root):
    """Recover one broadcast-lanes-reduction stage backwards from its root."""
    scalar_sig = _output_sig(gm, root)
    if not _merge_like(gm, root, scalar_sig):
        return None
    reducers, leaves = set(), set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node in reducers:
            continue
        if not _merge_like(gm, node, scalar_sig):
            leaves.add(node)
            continue
        reducers.add(node)
        for pred in gm.preds[node]:
            if _merge_like(gm, pred, scalar_sig):
                stack.append(pred)
            else:
                leaves.add(pred)
    lanes = []
    for tail in sorted(leaves, key=gm.topo_pos.get):
        reverse_chain = [tail]
        current = tail
        while len(gm.preds[current]) == 1:
            pred = next(iter(gm.preds[current]))
            shared_input = any(
                len(gm.t_cons[tid]) > 1
                for tid in gm.cons_of_op.get(current, ())
                if not gm.t_prod[tid])
            if len(gm.succs[pred]) != 1 or shared_input:
                break
            if not gm.pair_tensors.get((pred, current), ()):
                break
            reverse_chain.append(pred)
            current = pred
        lanes.append(list(reversed(reverse_chain)))
    if len(lanes) < 2:
        return None
    nodes = reducers | {op for lane in lanes for op in lane}
    return {'root': root, 'reducers': reducers, 'lanes': lanes,
            'nodes': nodes, 'scalar_sig': scalar_sig}


def detect_fork_join_stages(gm, min_coverage=0.8):
    """Detect a repeated fork-join stage chain without case/id constants."""
    broadcasts = []
    for tid in gm.tensors:
        producers, consumers = gm.t_prod[tid], gm.t_cons[tid]
        if len(producers) == 1 and len(consumers) >= 2:
            producer = next(iter(producers))
            if consumers <= gm.succs[producer]:
                broadcasts.append((producer, tid, frozenset(consumers)))
    roots = {producer for producer, _, _ in broadcasts}
    roots.update(op for op in gm.eligible if not gm.succs[op])
    stages = []
    occupied = set()
    for root in sorted(roots, key=gm.topo_pos.get):
        stage = _parse_fork_join_stage(gm, root)
        if stage is None or occupied & stage['nodes']:
            continue
        stages.append(stage)
        occupied.update(stage['nodes'])
    if not stages or len(occupied) < min_coverage * len(gm.eligible):
        return []
    lane_counts = [len(stage['lanes']) for stage in stages]
    dominant = max(set(lane_counts), key=lane_counts.count)
    if sum(count == dominant for count in lane_counts) < 0.8 * len(stages):
        return []
    return stages


def template_fork_join_plan(gm, num_cores, problem, stage_window=1):
    """Keep repeated fork-join lanes on stable cores across all stages."""
    _validate_problem_and_cores(problem, num_cores)
    if isinstance(stage_window, bool) or not isinstance(stage_window, int) \
            or stage_window <= 0:
        raise ValueError('stage_window must be a positive integer')
    sg = StructuralGraph(gm)
    stages = detect_fork_join_stages(gm)
    if not stages:
        raise ValueError('no repeated fork-join stage chain detected')
    op_core = {}
    stage_index = {}
    for index, stage in enumerate(stages):
        lane_core = {}
        for lane_id, lane in enumerate(stage['lanes']):
            core = lane_id % num_cores
            lane_core[lane[-1]] = core
            for op in lane:
                op_core[op] = core
                stage_index[op] = index
        # Reduction nodes follow the majority/heaviest predecessor lane.
        for op in gm.topo:
            if op not in stage['reducers']:
                continue
            weights = defaultdict(float)
            for pred in gm.preds[op]:
                core = op_core.get(pred, lane_core.get(pred, 0))
                weights[core] += max(_work(gm, pred)) or 1.0
            core = min(weights, key=lambda c: (-weights[c], c))
            op_core[op] = core
            stage_index[op] = index
    # Any uncovered boundary operation falls back to least-loaded legal lane.
    loads = [0.0] * num_cores
    for op, core in op_core.items():
        loads[core] += max(_work(gm, op))
    for op in sg.nodes:
        if op in op_core:
            continue
        core = min(range(num_cores), key=lambda c: (loads[c], c))
        op_core[op] = core
        loads[core] += max(_work(gm, op))
        stage_index[op] = max((stage_index.get(p, 0)
                               for p in sg.preds[op]), default=0)
    # Group by stage windows and core, retaining a monotone stage order.
    keys = sorted({(stage_index[op] // stage_window, op_core[op])
                   for op in sg.nodes})
    key_group = {key: idx for idx, key in enumerate(keys)}
    mapping = {op: key_group[(stage_index[op] // stage_window, op_core[op])]
               for op in sg.nodes}
    if not task_graph_acyclic(sg, mapping):
        # Conservative fallback: one stage per core-task.
        keys = sorted({(stage_index[op], op_core[op]) for op in sg.nodes})
        key_group = {key: idx for idx, key in enumerate(keys)}
        mapping = {op: key_group[(stage_index[op], op_core[op])]
                   for op in sg.nodes}
    schedules = [[] for _ in range(num_cores)]
    for (window_id, core), group in sorted(key_group.items()):
        del window_id
        schedules[core].append(group)
    return _finalize_plan(sg, mapping, schedules)


def wide_plan(gm, num_cores, problem):
    """Schedule wide/high-parallelism graphs with per-layer M/V LPT."""
    sg = StructuralGraph(gm)
    _validate_problem_and_cores(problem, num_cores)
    lane_of = {}
    cumulative = [[0.0, 0.0] for _ in range(num_cores)]
    for depth in range(sg.max_depth + 1):
        assigned, cumulative = _lpt_lanes(
            gm, sg.layers.get(depth, ()), num_cores,
            initial_loads=cumulative,
        )
        lane_of.update(assigned)
    # A layer is the natural barrier in this graph family.
    return plan_from_op_core(sg, lane_of, num_cores=num_cores, window=1)


def _validate_problem_and_cores(problem, num_cores):
    if problem not in (1, 2, 3):
        raise ValueError('problem must be 1, 2, or 3')
    if isinstance(num_cores, bool) or not isinstance(num_cores, int) or num_cores <= 0:
        raise ValueError('num_cores must be a positive integer')


__all__ = [
    'StructuralGraph',
    'task_graph_acyclic',
    'plan_from_op_core',
    'wavefront_plan',
    'fork_join_plan',
    'detect_fork_join_stages',
    'template_fork_join_plan',
    'wide_plan',
    'cache_affine_plan',
]

def cache_affine_plan(gm, num_cores, problem, mode='strong', tolerance=0.25):
    """Wide-family plan with graph-input core affinity.

    Every shared graph input is pre-assigned to one core (size-aware round
    robin); ops then prefer the core owning most of their input bytes.  This
    collapses duplicate COPY_INs of the same tensor onto a single core, which
    both removes re-reads and (when they remain) keeps them close in time for
    the read-only L2.  ``weak`` only tie-breaks inside LPT; ``strong`` allows a
    load tolerance to follow affinity.
    """
    sg = StructuralGraph(gm)
    _validate_problem_and_cores(problem, num_cores)
    inputs = sorted(gm.input_tensors, key=lambda t: -gm.tensor_size[t])
    input_core, byte_loads = {}, [0.0] * num_cores
    for tid in inputs:
        core = min(range(num_cores), key=lambda c: (byte_loads[c], c))
        input_core[tid] = core
        byte_loads[core] += gm.tensor_size[tid]

    def affinity_of(op):
        weights = defaultdict(float)
        for tid in gm.cons_of_op.get(op, ()):
            if tid in input_core:
                weights[input_core[tid]] += gm.tensor_size[tid]
        return max(weights, key=weights.get) if weights else None

    lane_of, cumulative = {}, [[0.0, 0.0] for _ in range(num_cores)]
    for depth in range(sg.max_depth + 1):
        ops = list(sg.layers.get(depth, ()))
        if mode == 'weak':
            affinity = {op: affinity_of(op) for op in ops}
            assigned, cumulative = _lpt_lanes(
                gm, ops, num_cores, initial_loads=cumulative, affinity=affinity)
            lane_of.update(assigned)
        else:
            for op in sorted(ops, key=lambda o: -max(_work(gm, o))):
                m_work, v_work = _work(gm, op)
                loads_after = [max(cumulative[c][0] + m_work,
                                   cumulative[c][1] + v_work)
                               for c in range(num_cores)]
                best = min(loads_after)
                preferred = affinity_of(op)
                if preferred is not None and loads_after[preferred]                         <= best + tolerance * max(best, 1.0):
                    core = preferred
                else:
                    core = min(range(num_cores), key=lambda c: (loads_after[c], c))
                cumulative[core][0] += m_work
                cumulative[core][1] += v_work
                lane_of[op] = core
    return plan_from_op_core(sg, lane_of, num_cores=num_cores, window=1)
