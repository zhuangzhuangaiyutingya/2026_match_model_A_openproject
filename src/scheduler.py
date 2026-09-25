# -*- coding: utf-8 -*-
"""多核切图与调度算法（A题参赛实现）。

流程：
  1. GraphModel 收缩 COPY 节点建非 COPY 算子 DAG，缓存张量桥接边权，
     并按评估器口径精确计算"新增 DDR 搬运量"（场景 A 按 Task、场景 B 按核）。
  2. region_grow：以张量字节为通信收益的贪心簇生长（惰性堆更新），
     把强通信、共享输入的算子聚到同簇，按目标工作量均衡成簇。
  3. fm_refine：边界算子移动的局部改进，目标 = 新增搬运量（可切换
     场景 A / 场景 B 口径），带负载硬约束。
  4. 场景 A：簇数 = 核数的 m 倍，通信感知列表调度（HEFT 简化）排核；
     场景 B：簇可直接对应核（或少量超分后按场景 B 代价共调度）。

输出方案 JSON：node_to_subgraph + core_schedules。
"""
import heapq
import math
import random
from collections import defaultdict, deque

COPY_TYPES = {'COPY_IN', 'COPY_OUT'}
BW = 60.0               # DDR 字节/周期
CAP_L1 = 524288
CAP_UB = 131072
SAME_CORE_WAIT = 100    # 场景 A 同核 Task 间隔
CROSS_CORE_WAIT = 1000  # 场景 A 跨核前驱等待
CROSS_COPY_DELAY = 500  # 场景 B 跨核同步延迟


class GraphModel:
    def __init__(self, graph_json):
        self.raw = graph_json
        ops = {op['id']: op for op in graph_json['ops']}
        self.ops = ops
        tensors = {t['id']: t for t in graph_json['tensors']}
        self.tensors = tensors
        self.tensor_size = {tid: t['size'] for tid, t in tensors.items()}
        self.tensor_pos = {tid: t['pos'] for tid, t in tensors.items()}

        producers = defaultdict(set)
        consumers = defaultdict(set)
        direct_edges = []
        for e in graph_json['edges']:
            s, t = e['source'], e['target']
            if s in ops and t not in ops:
                producers[t].add(s)
            elif s not in ops and t in ops:
                consumers[s].add(t)
            elif s in ops and t in ops and s != t:
                direct_edges.append((s, t))

        self.eligible = sorted(
            i for i, o in ops.items() if o['op'] not in COPY_TYPES)
        eligible_set = set(self.eligible)

        self.has_orig_out = {
            tid: any(ops[o].get('op') == 'COPY_OUT'
                     for o in consumers.get(tid, ()) if o in ops)
            for tid in tensors}
        self.t_prod = {tid: producers.get(tid, set()) & eligible_set
                       for tid in tensors}
        self.t_cons = {tid: consumers.get(tid, set()) & eligible_set
                       for tid in tensors}

        # op -> 张量索引
        self.prod_of_op = defaultdict(list)
        self.cons_of_op = defaultdict(list)
        for tid in tensors:
            for p in self.t_prod[tid]:
                self.prod_of_op[p].append(tid)
            for c in self.t_cons[tid]:
                self.cons_of_op[c].append(tid)
        self.tensors_of_op = {}
        for op in self.eligible:
            seen = set(self.prod_of_op.get(op, ()))
            self.tensors_of_op[op] = sorted(
                seen | set(self.cons_of_op.get(op, ())))

        # 收缩 COPY：eligible 邻接 + (u,v) 桥接张量
        succs_raw = defaultdict(set)
        for s, t in direct_edges:
            succs_raw[s].add(t)
        for tid, pset in producers.items():
            for p in pset:
                for c in consumers.get(tid, ()):
                    if p != c:
                        succs_raw[p].add(c)

        def walk(node):
            out = []
            stack = list(succs_raw.get(node, ()))
            seen = set()
            while stack:
                nxt = stack.pop()
                if nxt in eligible_set:
                    out.append(nxt)
                    continue
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.extend(succs_raw.get(nxt, ()))
            return out

        preds = {i: set() for i in eligible_set}
        succs = {i: set() for i in eligible_set}
        self.pair_tensors = defaultdict(list)
        for i in self.eligible:
            for j in walk(i):
                if j != i:
                    succs[i].add(j)
                    preds[j].add(i)
        for tid in tensors:
            if self.tensor_size[tid] <= 0:
                continue
            for p in self.t_prod[tid]:
                for c in self.t_cons[tid]:
                    if p != c:
                        self.pair_tensors[(p, c)].append(tid)
        self.preds = preds
        self.succs = succs

        # 图输入张量（无 eligible 生产者但有消费者）
        self.input_tensors = sorted(
            tid for tid in tensors
            if self.t_cons[tid] and not self.t_prod[tid])

        self._build_topo()

        self.m_work = {i: (ops[i]['cycles'] if ops[i]['pipe'] == 'PIPE_M' else 0)
                       for i in self.eligible}
        self.v_work = {i: (ops[i]['cycles'] if ops[i]['pipe'] == 'PIPE_V' else 0)
                       for i in self.eligible}
        self.eff_work = {i: max(self.m_work[i], self.v_work[i])
                         for i in self.eligible}
        self.total_eff = sum(self.eff_work.values())

        # 深度、向上工作量
        self.depth = {}
        for i in self.topo:
            self.depth[i] = max(
                (self.depth[p] + 1 for p in self.preds[i]), default=0)
        self.up_work = {}
        for i in reversed(self.topo):
            best = 0.0
            for j in self.succs[i]:
                comm = max((self.tensor_size[t]
                            for t in self.pair_tensors.get((i, j), ())), default=0)
                best = max(best, self.up_work[j] + comm / BW)
            self.up_work[i] = self.eff_work[i] + best
        self.critical_path = max(
            (self.up_work[i] for i in self.eligible), default=0.0)
        self.parallelism = self.total_eff / max(self.critical_path, 1.0)

    def _build_topo(self):
        indeg = {i: len(self.preds[i]) for i in self.eligible}
        q = deque(sorted(i for i in self.eligible if indeg[i] == 0))
        topo = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v in sorted(self.succs[u]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(topo) != len(self.eligible):
            raise ValueError('contracted graph has a cycle')
        self.topo = topo
        self.topo_pos = {op: k for k, op in enumerate(topo)}

    # ---------- 评估器口径的新增搬运量 ----------
    def _tensor_added(self, tid, tset, cset):
        """单张量在给定 生产簇集合/消费簇集合 下的新增搬运字节。"""
        size = self.tensor_size[tid]
        if size <= 0:
            return 0
        if not tset:                       # 图输入
            return size * max(0, len(cset) - 1) if cset else 0
        external = cset - tset
        out_count = len(tset) if (external or not cset
                                  or self.has_orig_out[tid]) else 0
        orig = 1 if self.has_orig_out[tid] else 0
        return size * max(0, out_count - orig) + size * len(external)

    def added_traffic(self, op_group, empty_group_factory=set):
        """op -> 组(Task/核) 映射的总新增搬运字节（不含 spill）。"""
        tset_of = defaultdict(set)
        cset_of = defaultdict(set)
        for tid in self.tensors:
            for p in self.t_prod[tid]:
                tset_of[tid].add(op_group[p])
            for c in self.t_cons[tid]:
                cset_of[tid].add(op_group[c])
        total = 0
        for tid in self.tensors:
            total += self._tensor_added(tid, tset_of[tid], cset_of[tid])
        return total

    def cluster_workset(self, ops_of_cluster):
        """簇工作集：外部输入与 UB 中间量落 UB、L1 中间量落 L1。"""
        opset = set(ops_of_cluster)
        l1 = ub = 0
        for tid in self.tensors:
            produced = self.t_prod[tid] & opset
            touched = produced or (self.t_cons[tid] & opset)
            if not touched:
                continue
            size = self.tensor_size[tid]
            if produced:
                if self.tensor_pos[tid] == 'L1':
                    l1 += size
                else:
                    ub += size
            else:
                ub += size               # 外部输入（含图输入）落 UB
        return l1, ub


# ---------------- 区域生长划分 ----------------

def region_grow(gm, n_parts, size_slack=0.30, edge_mode='bytes',
                balance='max', stats=None, reseed=True):
    """贪心簇生长：种子取剩余向上工作量最大者，邻居按通信收益/工作量入堆。

    reseed=True：簇前沿耗尽（链/弱连通图）时换新种子继续装填，避免
    剩余算子落入兜底散射（case_020 类宽图实测收益 S 1.0->3.8）。
    reseed=False：保留旧行为（前沿耗尽即止，剩余走散射兜底）；部分图
    （case_011/027）散射+精化反而更优，作为独立候选保留。
    stats（可选 dict）回填诊断信息：stalls=前沿耗尽后补种的次数，
    scattered=落入兜底散射的算子数。
    """
    n = len(gm.eligible)
    if n_parts <= 1 or n <= n_parts:
        return {op: 0 for op in gm.eligible}
    target = gm.total_eff * (1.0 + size_slack) / n_parts
    op_cluster = {}
    cluster_work = [0.0] * n_parts
    cluster_m = [0.0] * n_parts
    cluster_v = [0.0] * n_parts

    def load_of(cluster):
        if balance == 'mv':
            return max(cluster_m[cluster], cluster_v[cluster])
        return cluster_work[cluster]
    unassigned = set(gm.eligible)

    # 输入张量已读簇：避免同簇重复读同一图输入
    input_read = {tid: -1 for tid in gm.input_tensors}

    def gain_of(op, cluster):
        """op 并入 cluster 的通信收益；edge_mode='count' 时按边条数计权。

        小图上跨核边的代价主要来自固定同步延迟（500 周期）而非字节数，
        此时按条数计权能找到边数更少的切分。
        """
        g = 0.0
        for j in gm.preds[op] | gm.succs[op]:
            if op_cluster.get(j) == cluster:
                if edge_mode == 'count':
                    g += 1.0 if gm.pair_tensors.get((op, j)) else 0.0
                    g += 1.0 if gm.pair_tensors.get((j, op)) else 0.0
                else:
                    g += sum(gm.tensor_size[t]
                             for t in gm.pair_tensors.get((op, j), ()))
                    g += sum(gm.tensor_size[t]
                             for t in gm.pair_tensors.get((j, op), ()))
        for tid in gm.cons_of_op.get(op, ()):
            if input_read.get(tid) == cluster:
                g += (1.0 if edge_mode == 'count' else gm.tensor_size[tid])
        return g

    # 种子序：向上工作量降序（静态，可重复用于前沿耗尽后的补种）
    seed_order = sorted(gm.eligible,
                        key=lambda o: (gm.up_work[o], -gm.topo_pos[o]),
                        reverse=True)
    seed_pos = 0
    stall_count = 0

    def grow_seed(cluster, heap):
        """从剩余算子里选新种子并入 cluster，邻居入堆。"""
        nonlocal seed_pos
        while seed_pos < len(seed_order) and seed_order[seed_pos] not in unassigned:
            seed_pos += 1
        if seed_pos >= len(seed_order):
            return False
        seed = seed_order[seed_pos]
        op_cluster[seed] = cluster
        cluster_work[cluster] += gm.eff_work[seed]
        cluster_m[cluster] += gm.m_work[seed]
        cluster_v[cluster] += gm.v_work[seed]
        unassigned.discard(seed)
        for tid in gm.cons_of_op.get(seed, ()):
            if tid in input_read and input_read[tid] == -1:
                input_read[tid] = cluster
        for j in gm.preds[seed] | gm.succs[seed]:
            if j in unassigned:
                g = gain_of(j, cluster)
                w = max(gm.eff_work[j], 1e-6)
                heapq.heappush(heap, (-g / w, gm.topo_pos[j], j))
        return True

    for cluster in range(n_parts):
        if not unassigned:
            break
        heap = []
        grow_seed(cluster, heap)
        while unassigned and load_of(cluster) < target:
            if not heap:
                if not reseed:
                    break
                # 前沿耗尽（链/弱连通图：一条链长完堆即空）。换新种子
                # 继续装填本簇；否则剩余算子会全部落入兜底散射，
                # 既破坏均衡又把大量张量切成跨核拷贝（case_020 实测
                # 99% 算子进兜底、新增搬运 20~34MB、S=1.00）。
                stall_count += 1
                grow_seed(cluster, heap)
                continue
            neg, _, op = heapq.heappop(heap)
            if op not in unassigned:
                continue
            g_now = gain_of(op, cluster)
            w = max(gm.eff_work[op], 1e-6)
            key = -g_now / w
            if key > neg + 1e-12:          # 收益已变化：惰性重推
                heapq.heappush(heap, (key, gm.topo_pos[op], op))
                continue
            op_cluster[op] = cluster
            cluster_work[cluster] += gm.eff_work[op]
            cluster_m[cluster] += gm.m_work[op]
            cluster_v[cluster] += gm.v_work[op]
            unassigned.discard(op)
            for tid in gm.cons_of_op.get(op, ()):
                if tid in input_read and input_read[tid] == -1:
                    input_read[tid] = cluster
            for j in gm.preds[op] | gm.succs[op]:
                if j in unassigned:
                    g2 = gain_of(j, cluster)
                    w2 = max(gm.eff_work[j], 1e-6)
                    heapq.heappush(heap, (-g2 / w2, gm.topo_pos[j], j))
    # 兜底：按通信最强邻簇吸收剩余
    if unassigned:
        input_owner = {}
        for tid in gm.input_tensors:
            for c in gm.t_cons[tid]:
                if c in op_cluster:
                    input_owner[tid] = op_cluster[c]
                    break
        for op in gm.topo:
            if op in op_cluster:
                continue
            best, best_gain = None, -1.0
            cand_clusters = {op_cluster[j]
                             for j in gm.preds[op] | gm.succs[op]
                             if j in op_cluster}
            if not cand_clusters:
                cand_clusters = set(range(n_parts))
            for cl in cand_clusters:
                g = gain_of(op, cl)
                if g > best_gain:
                    best_gain, best = g, cl
            op_cluster[op] = best
            cluster_work[best] += gm.eff_work[op]
            cluster_m[best] += gm.m_work[op]
            cluster_v[best] += gm.v_work[op]
            unassigned.discard(op)
            if stats is not None:
                stats['scattered'] = stats.get('scattered', 0) + 1
    if stats is not None:
        stats['stalls'] = stall_count
    return op_cluster


def horizontal_split(gm, n_parts):
    """按（深度, 拓扑序）横切的保守候选。"""
    order = sorted(gm.eligible, key=lambda o: (gm.depth[o], gm.topo_pos[o]))
    step = gm.total_eff / n_parts
    acc = 0.0
    next_bound = step
    cluster = 0
    mapping = {}
    for o in order:
        mapping[o] = cluster
        acc += gm.eff_work[o]
        if acc >= next_bound and cluster < n_parts - 1:
            cluster += 1
            next_bound += step
    return mapping


def path_partition(gm, n_parts, max_pass=6):
    """路径分区：迭代提取最长链，每条链一个簇，LPT 装簇。

    适用于规则链图（case_020/044/005 类）：每条链整条一簇 => 零切边、
    链内天然有序。任务图仍可能因链间交叉依赖成环（罕见），调用方需
    _task_graph_ok 验证。装簇按 max(M,V) LPT 并带前驱链粘性，减少
    跨核等待。返回 (op_group, core_tasks)。
    """
    remaining = set(gm.eligible)
    chains = []
    while remaining:
        # 链头：剩余算子中 depth 最小者（并列取 up_work 最大）
        head = min(remaining,
                   key=lambda o: (gm.depth[o], -gm.up_work[o], gm.topo_pos[o]))
        chain = [head]
        seen = {head}
        cur = head
        while True:
            nxt = None
            best_key = None
            for j in gm.succs[cur]:
                if j in remaining and j not in seen:
                    # 选向下工作量最大且 depth 恰好 +1 的后继（保持链状）
                    key = (-gm.up_work[j], gm.topo_pos[j])
                    if best_key is None or key < best_key:
                        best_key, nxt = key, j
            if nxt is None:
                break
            chain.append(nxt)
            seen.add(nxt)
            cur = nxt
        for o in chain:
            remaining.discard(o)
        chains.append(chain)
        if max_pass and len(chains) > max_pass * n_parts * 40:
            break

    # 链装核：max(M,V) LPT + 前驱粘性
    chain_work = []
    for chain in chains:
        m = sum(gm.m_work[o] for o in chain)
        v = sum(gm.v_work[o] for o in chain)
        chain_work.append((max(m, v), m, v, chain))
    chain_work.sort(key=lambda x: -x[0])

    op_group = {}
    core_m = [0.0] * n_parts
    core_v = [0.0] * n_parts
    core_of = {}
    core_tasks = [[] for _ in range(n_parts)]
    for idx, (w, m, v, chain) in enumerate(chain_work):
        stick = defaultdict(float)
        for o in chain:
            for j in gm.preds[o]:
                if j in core_of:
                    stick[core_of[j]] += gm.eff_work[j]
        best_core, best_key = None, None
        for core in range(n_parts):
            load = max(core_m[core], core_v[core])
            key = (load - 0.5 * min(stick.get(core, 0.0), load), core)
            if best_key is None or key < best_key:
                best_core, best_key = core, key
        cid = len(core_tasks) * 0 + idx  # 簇 id 唯一
        for o in chain:
            op_group[o] = cid
        core_m[best_core] += m
        core_v[best_core] += v
        for o in chain:
            core_of[o] = best_core
        core_tasks[best_core].append(cid)
    return op_group, core_tasks


def bandcomp_partition(gm, n_cores, window=6, sticky=0.5, seed=0):
    """深度窗口 × 连通分量划分：天然无环的并行条带。

    每个深度窗口内的弱连通分量互无边（分量定义保证），窗口间边只会
    从浅窗口指向深窗口 => 任务图按窗口分层，天然无环。分量按
    max(M,V) 工作量 LPT 装到核上（粘性：优先跟随前驱分量所在核，
    减少跨核等待）。适用于深而稀疏、张量很小的图（case_005/049/050/
    056 类：grow 补种后簇交错成环、只能回退横切串行）。
    返回 (op_group, core_tasks)。
    """
    maxd = max(gm.depth[o] for o in gm.eligible)
    by_depth = defaultdict(list)
    for o in gm.eligible:
        by_depth[gm.depth[o]].append(o)

    windows = []
    for start in range(0, maxd + 1, window):
        lo, hi = start, min(start + window - 1, maxd)
        win_ops = {o for d in range(lo, hi + 1) for o in by_depth.get(d, ())}
        # 弱连通分量
        seen = set()
        comps = []
        for o in sorted(win_ops, key=lambda x: gm.topo_pos[x]):
            if o in seen:
                continue
            comp = []
            seen.add(o)
            stack = [o]
            while stack:
                x = stack.pop()
                comp.append(x)
                for j in gm.preds[x] | gm.succs[x]:
                    if j not in seen and j in win_ops:
                        seen.add(j)
                        stack.append(j)
            comps.append(sorted(comp, key=lambda x: gm.topo_pos[x]))
        comps.sort(key=lambda c: -sum(gm.eff_work[o] for o in c))
        windows.append(comps)

    op_group = {}
    comp_core_of = {}
    core_m = [0.0] * n_cores
    core_v = [0.0] * n_cores
    core_tasks = [[] for _ in range(n_cores)]
    rng = random.Random(seed) if seed else None
    next_id = 0
    for comps in windows:
        comp_cores = {}
        for comp in comps:
            # 粘性：前驱分量（与本分量有边的已放置算子）所在核加权
            stick = defaultdict(float)
            for o in comp:
                for j in gm.preds[o]:
                    cg = op_group.get(j)
                    if cg is not None:
                        core = comp_core_of.get(cg)
                        if core is not None:
                            stick[core] += gm.eff_work[j]
            best_core, best_key = None, None
            for core in range(n_cores):
                load = max(core_m[core], core_v[core])
                stick_gain = stick.get(core, 0.0)
                tie = rng.random() if rng else 0.0
                key = (load - sticky * min(stick_gain, load), tie, core)
                if best_key is None or key < best_key:
                    best_core, best_key = core, key
            cid = next_id
            next_id += 1
            for o in comp:
                op_group[o] = cid
            core_m[best_core] += sum(gm.m_work[o] for o in comp)
            core_v[best_core] += sum(gm.v_work[o] for o in comp)
            core_tasks[best_core].append(cid)
            comp_cores[cid] = best_core
        comp_core_of.update(comp_cores)
    return op_group, core_tasks


# ---------------- 移动环守卫 ----------------

class CycleGuard:
    """增量维护任务图边计数，拒绝会造成任务环的算子搬动。

    环判定：把 op 从 g_from 移到 g_to 新增跨任务边
    (pred_task -> g_to) 与 (g_to -> succ_task)。当旧任务图中存在
    g_to 可达某 pred_task、或某 succ_task 可达 g_to 的路径时，移动成环。
    """

    def __init__(self, gm, op_group):
        self.gm = gm
        self.group = dict(op_group)
        self.edge_count = defaultdict(int)     # (ta, tb) -> 支撑该边的依赖数
        for u in gm.eligible:
            for v in gm.succs.get(u, ()):      # 完整依赖，含零字节边
                ta, tb = self.group[u], self.group[v]
                if ta != tb:
                    self.edge_count[(ta, tb)] += 1
        self.succ_adj = None
        self.pred_adj = None

    def _rebuild_adj(self):
        succ = defaultdict(set)
        pred = defaultdict(set)
        for (a, b), c in self.edge_count.items():
            if c > 0:
                succ[a].add(b)
                pred[b].add(a)
        self.succ_adj, self.pred_adj = succ, pred

    def _update_edges(self, op, g_from, g_to, sign):
        """按移动前后逐边增减计数（sign=-1 移除旧边，+1 添加新边）。"""
        gm = self.gm
        for s in gm.succs[op]:
            ts = self.group[s]
            if ts != g_from:
                self.edge_count[(g_from, ts)] += sign
            if ts != g_to:
                self.edge_count[(g_to, ts)] += sign
        for p in gm.preds[op]:
            tp = self.group[p]
            if tp != g_from:
                self.edge_count[(tp, g_from)] += sign
            if tp != g_to:
                self.edge_count[(tp, g_to)] += sign

    def can_move(self, op, g_to):
        g_from = self.group[op]
        if g_from == g_to:
            return True
        self._rebuild_adj()
        succ_tasks = {self.group[s] for s in self.gm.succs[op]}
        pred_tasks = {self.group[p] for p in self.gm.preds[op]}
        # 前向：g_to 可达的所有任务（判定 pred_task 落入则成环）
        fwd = set()
        stack = [g_to]
        while stack:
            t = stack.pop()
            for nxt in self.succ_adj.get(t, ()):
                if nxt not in fwd:
                    fwd.add(nxt)
                    stack.append(nxt)
        if pred_tasks & fwd:
            return False
        # 后向：可达 g_to 的所有任务（判定 succ_task 落入则成环）
        bwd = set()
        stack = [g_to]
        while stack:
            t = stack.pop()
            for prv in self.pred_adj.get(t, ()):
                if prv not in bwd:
                    bwd.add(prv)
                    stack.append(prv)
        if succ_tasks & bwd:
            return False
        return True

    def apply(self, op, g_to):
        g_from = self.group[op]
        if g_from == g_to:
            return
        self._update_edges(op, g_from, g_to, -1)
        self.group[op] = g_to
        self._update_edges(op, g_from, g_to, +1)


# ---------------- FM 局部改进 ----------------

def fm_refine(gm, op_group, n_groups, passes=4, balance_tol=0.30):
    """边界算子在相邻组间移动以减少新增搬运量；负载硬约束防漂移。"""
    work = defaultdict(float)
    for op, g in op_group.items():
        work[g] += gm.eff_work[op]
    total = sum(work.values())
    target = total / max(n_groups, 1)
    guard = CycleGuard(gm, op_group)

    def move_delta(op, g_from, g_to):
        """把 op 从 g_from 移到 g_to 引起的新增搬运变化。"""
        delta = 0.0
        for tid in gm.tensors_of_op.get(op, ()):
            tset = {op_group[p] for p in gm.t_prod[tid]}
            cset = {op_group[c] for c in gm.t_cons[tid]}
            before = gm._tensor_added(tid, tset, cset)
            tset2 = ((tset - {g_from}) | {g_to}) if op in gm.t_prod[tid] else tset
            cset2 = ((cset - {g_from}) | {g_to}) if op in gm.t_cons[tid] else cset
            after = gm._tensor_added(tid, tset2, cset2)
            delta += after - before
        return delta

    for _ in range(passes):
        moved = 0
        boundary = []
        for op in gm.eligible:
            neighbors = {op_group[j]
                         for j in gm.preds[op] | gm.succs[op]}
            neighbors.discard(op_group[op])
            if neighbors:
                strength = sum(
                    gm.tensor_size[t]
                    for j in gm.preds[op] | gm.succs[op]
                    for t in gm.pair_tensors.get((op, j, ), ()))
                boundary.append((-strength, gm.topo_pos[op], op, neighbors))
        boundary.sort()
        for _, _, op, neighbors in boundary:
            g_from = op_group[op]
            w = gm.eff_work[op]
            best_g, best_delta = g_from, 0.0
            for g_to in sorted(neighbors):
                if work[g_to] + w > target * (1.0 + balance_tol):
                    continue
                d = move_delta(op, g_from, g_to)
                if d < best_delta - 1e-9:
                    best_delta, best_g = d, g_to
            if best_g != g_from:
                if not guard.can_move(op, best_g):
                    continue
                op_group[op] = best_g
                guard.apply(op, best_g)
                work[g_from] -= w
                work[best_g] += w
                moved += 1
        if not moved:
            break
    return op_group


def rebalance(gm, op_group, n_groups, tol=0.30):
    """处理超载组：把溢出算子挪到最空闲的可达组（接受小通信代价）。"""
    work = defaultdict(float)
    groups = defaultdict(list)
    for op, g in op_group.items():
        work[g] += gm.eff_work[op]
        groups[g].append(op)
    target = sum(work.values()) / max(n_groups, 1)
    guard = CycleGuard(gm, op_group)
    for _ in range(6):
        over = [g for g in groups if work[g] > target * (1 + tol)]
        under = [g for g in groups if work[g] < target * (1 - tol * 0.5)]
        if not over or not under:
            break
        over.sort(key=lambda g: -work[g])
        under.sort(key=lambda g: work[g])
        moved = False
        for g_over in over:
            cands = sorted(groups[g_over], key=lambda o: -gm.eff_work[o])
            for op in cands:
                if work[g_over] - gm.eff_work[op] < target * (1 + tol * 0.8):
                    break
                # 通信代价最小且不造成任务环的目标组
                best = None
                for g in sorted(under, key=lambda g: _shift_cost(
                        gm, op, g_over, g, op_group)):
                    if guard.can_move(op, g):
                        best = g
                        break
                if best is None:
                    continue
                op_group[op] = best
                guard.apply(op, best)
                groups[g_over].remove(op)
                groups[best].append(op)
                work[g_over] -= gm.eff_work[op]
                work[best] += gm.eff_work[op]
                moved = True
        if not moved:
            break
    return op_group


def _shift_cost(gm, op, g_from, g_to, op_group):
    """移动 op 的通信代价（正数，越小越好）。"""
    cost = 0.0
    for tid in gm.tensors_of_op.get(op, ()):
        tset = {op_group[p] for p in gm.t_prod[tid]}
        cset = {op_group[c] for c in gm.t_cons[tid]}
        before = gm._tensor_added(tid, tset, cset)
        tset2 = ((tset - {g_from}) | {g_to}) if op in gm.t_prod[tid] else tset
        cset2 = ((cset - {g_from}) | {g_to}) if op in gm.t_cons[tid] else cset
        after = gm._tensor_added(tid, tset2, cset2)
        cost += after - before
    return cost


# ---------------- 列表调度 ----------------

def list_schedule(gm, op_group, n_cores, scene, rank_spill_w=1.5):
    """通信感知列表调度：返回每核执行顺序。

    scene='A'：Task 串行执行；同核前驱 +100，跨核前驱 +1000+搬运。
    scene='B'：同核通信免费，跨核 +500+搬运。
    """
    task_ops = defaultdict(list)
    for op, t in op_group.items():
        task_ops[t].append(op)
    tkeys = sorted(task_ops)
    n_tasks = len(tkeys)

    tm = {t: sum(gm.m_work[o] for o in task_ops[t]) for t in tkeys}
    tv = {t: sum(gm.v_work[o] for o in task_ops[t]) for t in tkeys}
    t_comm = defaultdict(lambda: defaultdict(float))
    for (u, v), tids in gm.pair_tensors.items():
        tu, tvt = op_group[u], op_group[v]
        if tu != tvt:
            for tid in tids:
                t_comm[tu][tvt] += gm.tensor_size[tid]

    # 任务依赖必须来自完整依赖（preds/succs，含零字节 op→op 边），
    # 否则无张量的直接依赖会被调度漏掉，同核顺序可能违反官方校验。
    task_preds = defaultdict(set)
    task_succ = defaultdict(set)
    for u in gm.eligible:
        tu = op_group[u]
        for v in gm.succs.get(u, ()):
            tvt = op_group[v]
            if tu != tvt:
                task_succ[tu].add(tvt)
                task_preds[tvt].add(tu)
    order_topo = _topo_sort(tkeys, task_preds, task_succ)

    # 向上 rank（含通信）
    rank = {}
    for t in reversed(order_topo):
        best = 0.0
        for s in task_succ[t]:
            delay = CROSS_COPY_DELAY if scene == 'B' else CROSS_CORE_WAIT
            best = max(best, rank[s] + t_comm[t][s] / BW + delay)
        l1, ub = gm.cluster_workset(task_ops[t])
        spill = (max(0.0, ub - CAP_UB * 0.9) + max(0.0, l1 - CAP_L1 * 0.9))
        rank[t] = max(tm[t], tv[t]) + rank_spill_w * spill / BW + best

    # 拓扑约束下的优先序：Kahn + 按 -rank 的堆，保证前驱必先于后继调度
    indeg = {t: len(task_preds[t]) for t in tkeys}
    ready_heap = []
    for t in tkeys:
        if indeg[t] == 0:
            heapq.heappush(ready_heap, (-rank[t], t))
    order = []
    while ready_heap:
        _, t = heapq.heappop(ready_heap)
        order.append(t)
        for s in task_succ[t]:
            indeg[s] -= 1
            if indeg[s] == 0:
                heapq.heappush(ready_heap, (-rank[s], s))
    core_avail = [0.0] * n_cores
    core_load = [0.0] * n_cores
    core_tasks = [[] for _ in range(n_cores)]
    task_end = {}
    task_core = {}
    for t in order:
        dur = max(tm[t], tv[t])
        best_core, best_finish, tie_load = None, None, None
        for core in range(n_cores):
            ready = core_avail[core]
            for p in task_preds[t]:
                if task_core[p] == core:
                    wait = 0.0 if scene == 'B' else SAME_CORE_WAIT
                    if scene == 'B':
                        # 同核仍需前序 Task 完成，但数据免搬运
                        ready_c = task_end[p]
                    else:
                        ready_c = task_end[p] + wait + t_comm[p][t] / BW
                    ready = max(ready, ready_c)
                else:
                    delay = CROSS_COPY_DELAY if scene == 'B' else CROSS_CORE_WAIT
                    ready = max(ready, task_end[p] + delay
                                + t_comm[p][t] / BW)
            finish = ready + dur
            if (best_finish is None or finish < best_finish - 1e-9
                    or (abs(finish - best_finish) <= 1e-9
                        and core_load[core] < tie_load)):
                best_core, best_finish, tie_load = core, finish, core_load[core]
        task_core[t] = best_core
        task_end[t] = best_finish
        core_avail[best_core] = best_finish
        core_load[best_core] += dur
        core_tasks[best_core].append(t)
    # 每核内按启动时刻自然有序（列表调度已保证依赖）
    return core_tasks


def _topo_sort(keys, preds, succs):
    indeg = {k: len(preds[k]) for k in keys}
    q = deque(sorted(k for k in keys if indeg[k] == 0))
    out = []
    while q:
        u = q.popleft()
        out.append(u)
        for v in sorted(succs[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if len(out) != len(keys):
        raise ValueError('task graph has a cycle')
    return out


# ---------------- 方案生成 ----------------

def normalize(op_group):
    """重编号为连续 sgid 并保持拓扑友好（旧 id 排序）。"""
    groups = defaultdict(list)
    for op, g in op_group.items():
        groups[g].append(op)
    remap = {g: k for k, g in enumerate(sorted(groups))}
    return {op: remap[g] for op, g in op_group.items()}


def _task_graph_ok(gm, op_group):
    """任务图（按组收缩）是否无环。

    必须用完整依赖（preds/succs，含零字节 op→op 边与穿过 COPY 节点的
    边）——官方 derive_multicore_plan 按完整依赖收缩判环，仅查
    pair_tensors 会漏判零字节边造成的环（官方报 'contracted subgraph
    graph contains a cycle'）。"""
    tp, ts = defaultdict(set), defaultdict(set)
    for u in gm.eligible:
        for v in gm.succs.get(u, ()):
            a, b = op_group[u], op_group[v]
            if a != b:
                ts[a].add(b)
                tp[b].add(a)
    keys = set(op_group.values())
    indeg = {t: len(tp[t]) for t in keys}
    dq = deque(t for t in keys if indeg[t] == 0)
    seen = 0
    while dq:
        t = dq.popleft()
        seen += 1
        for s in ts[t]:
            indeg[s] -= 1
            if indeg[s] == 0:
                dq.append(s)
    return seen == len(keys)


def _break_task_cycles(gm, op_group, max_moves=400):
    """就地修复非凸簇造成的任务环（补种生长的簇是多个生长岛的并集）。

    每轮扫出双向任务边对（A→B 且 B→A），把边界上"更属于对端"的算子
    搬过去：score = 邻居在对端簇数 - 邻居在本簇数，平分时搬工作量小者。
    相比整体回退横切（串行带、S≈1，case_005/049/050/056 均如此），
    通常只需搬极少量算子。消除 2-环后仍带长环时，DFS 找环、把环上
    最弱任务边的源端算子搬进目标簇。返回分组；调用方需再验证无环。
    """
    group = dict(op_group)

    def neighbors_by_cluster(op):
        cnt = defaultdict(int)
        for j in gm.preds[op] | gm.succs[op]:
            cnt[group[j]] += 1
        return cnt

    def two_cycle_pairs():
        dirs = defaultdict(list)
        for u in gm.eligible:
            for v in gm.succs.get(u, ()):      # 完整依赖，含零字节边
                a, b = group[u], group[v]
                if a != b:
                    dirs[(a, b)].append(u)
        return [(a, b) for (a, b) in dirs if (b, a) in dirs], dirs

    for _ in range(max_moves):
        pairs, dirs = two_cycle_pairs()
        if not pairs:
            break
        a, b = pairs[0]
        boundary = set(dirs[(a, b)]) | set(dirs[(b, a)])
        best_op, best_score = None, None
        for op in boundary:
            cnt = neighbors_by_cluster(op)
            own = group[op]
            other = b if own == a else a
            score = cnt.get(other, 0) - cnt.get(own, 0)
            if (best_score is None or score > best_score or
                    (score == best_score and
                     gm.eff_work[op] < gm.eff_work[best_op])):
                best_op, best_score = op, score
        if best_op is None:
            break
        own = group[best_op]
        group[best_op] = b if own == a else a

    # 长环兜底：最多 20 轮，每轮切一条环上最弱任务边
    for _ in range(20):
        ts = defaultdict(set)
        edge_ops = defaultdict(list)
        for u in gm.eligible:
            for v in gm.succs.get(u, ()):      # 完整依赖，含零字节边
                x, y = group[u], group[v]
                if x != y:
                    ts[x].add(y)
                    edge_ops[(x, y)].append(u)
        state = {}
        path = []

        def dfs(node):
            state[node] = 1
            path.append(node)
            for s in sorted(ts[node]):
                if state.get(s, 0) == 1:
                    return path[path.index(s):]
                if state.get(s, 0) == 0:
                    found = dfs(s)
                    if found:
                        return found
            path.pop()
            state[node] = 2
            return None

        cycle = None
        for t in sorted(ts):
            if state.get(t, 0) == 0:
                cycle = dfs(t)
                if cycle:
                    break
        if not cycle:
            return group
        def edge_bytes(x, y):
            total = 0
            for (u, v) in gm.pair_tensors:
                if group[u] == x and group[v] == y:
                    total += sum(gm.tensor_size[t2]
                                 for t2 in gm.pair_tensors.get((u, v), ()))
            return total
        best_edge, best_bytes = None, None
        for i in range(len(cycle)):
            x, y = cycle[i], cycle[(i + 1) % len(cycle)]
            nbytes = edge_bytes(x, y)
            if best_bytes is None or nbytes < best_bytes:
                best_edge, best_bytes = (x, y), nbytes
        x, y = best_edge
        best_op, best_score = None, None
        for op in edge_ops[(x, y)]:
            cnt = neighbors_by_cluster(op)
            score = cnt.get(y, 0) - cnt.get(x, 0)
            if best_score is None or score > best_score:
                best_op, best_score = op, score
        if best_op is None:
            return group
        group[best_op] = y
    return group


def _partition_safe(gm, n_parts, method, do_refine, balance_tol):
    """划分 + 精化，带无环验证与自动降级（精化 -> 修复环 -> 仅生长 -> 横切）。"""
    if method == 'grow':
        op_group = region_grow(gm, n_parts)
    elif method == 'grow_count':
        op_group = region_grow(gm, n_parts, edge_mode='count')
    elif method == 'grow_mv':
        op_group = region_grow(gm, n_parts, balance='mv')
    elif method == 'grow_ns':
        op_group = region_grow(gm, n_parts, reseed=False)
    elif method == 'grow_count_ns':
        op_group = region_grow(gm, n_parts, edge_mode='count', reseed=False)
    elif method == 'grow_mv_ns':
        op_group = region_grow(gm, n_parts, balance='mv', reseed=False)
    else:
        op_group = horizontal_split(gm, n_parts)
    if do_refine:
        refined = fm_refine(gm, dict(op_group), n_parts,
                            balance_tol=balance_tol)
        refined = rebalance(gm, refined, n_parts, tol=balance_tol)
        if _task_graph_ok(gm, refined):
            return refined
        for cand in (refined, op_group):
            if cand is None:
                continue
            repaired = _break_task_cycles(gm, cand)
            # 防退化：修复不得把簇数吞到一半以下（退化的散射输入会把
            # 搬运分数永远推向大簇，最终塌缩成单簇）
            if (len(set(repaired.values())) >= max(2, n_parts // 2)
                    and _task_graph_ok(gm, repaired)):
                return repaired
        if _task_graph_ok(gm, op_group):
            return op_group
        return horizontal_split(gm, n_parts)
    if _task_graph_ok(gm, op_group):
        return op_group
    repaired = _break_task_cycles(gm, op_group)
    if (len(set(repaired.values())) >= max(2, n_parts // 2)
            and _task_graph_ok(gm, repaired)):
        return repaired
    return horizontal_split(gm, n_parts)


def build_plan_scene_a(gm, n_cores, mult=2, method='grow',
                       balance_tol=0.30, do_refine=True):
    """场景 A：n_parts = n_cores*mult 个 Task + 列表调度。"""
    n_parts = max(1, min(int(n_cores * mult), len(gm.eligible)))
    op_task = _partition_safe(gm, n_parts, method, do_refine, balance_tol)
    op_task = normalize(op_task)
    core_tasks = list_schedule(gm, op_task, n_cores, scene='A')
    return plan_to_json(op_task, core_tasks)


def build_plan_scene_b(gm, n_cores, mult=1, method='grow',
                       balance_tol=0.25, do_refine=True):
    """场景 B：簇即核（mult=1）或少量超分后按场景 B 代价共调度。"""
    n_parts = max(1, min(int(n_cores * mult), len(gm.eligible)))
    op_task = _partition_safe(gm, n_parts, method, do_refine, balance_tol)
    op_task = normalize(op_task)
    core_tasks = list_schedule(gm, op_task, n_cores, scene='B')
    return plan_to_json(op_task, core_tasks)


def plan_to_json(op_group, core_tasks):
    """打包为提交格式，并保证每个簇恰好出现一次。"""
    sgids = set(op_group.values())
    schedule = [[int(s) for s in order] for order in core_tasks]
    present = {s for order in schedule for s in order}
    missing = sorted(sgids - present)
    if missing:
        if not schedule:
            schedule.append([])
        schedule[0].extend(int(s) for s in missing)
    return {
        'node_to_subgraph': {str(op): int(g) for op, g in op_group.items()},
        'core_schedules': schedule,
    }
