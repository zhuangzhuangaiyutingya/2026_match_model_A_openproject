# -*- coding: utf-8 -*-
"""实验驱动：生成候选方案 -> 官方评估器打分 -> 择优记录。

用法：
  python run_experiments.py --smoke            # 少量小用例快速验证
  python run_experiments.py --cases case_001 case_002 ...
  python run_experiments.py --full             # 全部 100 用例
结果写入 results/summary.json，支持断点续跑（已有条目跳过）。
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from collab_io import FileLock, atomic_write_json

HERE = Path(__file__).resolve().parent
ATT = HERE.parent
CODE = ATT / 'code'
DATA = ATT / 'data'
RESULTS = HERE / 'results'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))

from scheduler import GraphModel, build_plan_scene_a, build_plan_scene_b  # noqa: E402


def load_evaluators():
    from multicore_cut_evaluate_problem_1 import evaluate_scene_a
    from multicore_cut_evaluate_problem_2 import evaluate_scene_b
    from multicore_cut_evaluate_problem_3 import evaluate_problem_3
    from singlecore_evaluate import evaluate_singlecore
    return {
        1: evaluate_scene_a, 2: evaluate_scene_b,
        3: evaluate_problem_3, 'sc': evaluate_singlecore,
    }


def read_config():
    from evaluation_validation import read_evaluation_config
    from multicore_cut_evaluate_problem_1 import read_scene_a_config
    from multicore_cut_evaluate_problem_2 import read_scene_b_config
    from multicore_cut_evaluate_problem_3 import read_cache_config
    cfg_path = str(DATA / 'config.txt')
    base = read_evaluation_config(cfg_path)
    return {
        'bandwidth': base['bandwidth'], 'capacity': base['capacity'],
        'scene_a': read_scene_a_config(cfg_path),
        'scene_b': read_scene_b_config(cfg_path),
        'cache': read_cache_config(cfg_path),
    }


def eval_plan(evals, problem, graph, plan, cfg):
    if problem == 1:
        return evals[1](
            graph, plan, bandwidth=cfg['bandwidth'], capacity=cfg['capacity'],
            cross_core_wait=cfg['scene_a']['task_cross_core_wait_cycles'],
            same_core_wait=cfg['scene_a']['task_same_core_wait_cycles'])
    if problem == 2:
        return evals[2](
            graph, plan, bandwidth=cfg['bandwidth'], capacity=cfg['capacity'],
            cross_core_copy_delay=cfg['scene_b']['cross_core_copy_delay_cycles'])
    return evals[3](
        graph, plan, bandwidth=cfg['bandwidth'], capacity=cfg['capacity'],
        cross_core_copy_delay=cfg['scene_b']['cross_core_copy_delay_cycles'],
        **cfg['cache'])


def candidates_for(gm, problem, k, budget):
    """按用例规模/并行度生成候选（参数, 标签）列表。budget: 0=小 1=中 2=大。"""
    cands = []
    par = gm.parallelism
    if problem == 1:
        if budget == 0:
            # 小图评估便宜：粒度全档扫描（探针实测 mult 3/5/8 在 k=3/5
            # 上有 1.7%~7.9% 确定收益，见 case_011/case_027）
            mults = [1, 2, 3, 4, 5, 6, 8]
        elif budget == 1:
            mults = [1, 2, 4]
            if par > 8 * k:
                mults.append(6)
        else:
            # v12 大图实测：level 横切在 3 万算子级图上同样有效
            # （case_030 P2五核 +164%），故大图也加入。
            mults = [2, 3]
            for m in (2, 4):
                cands.append((dict(mult=m, method='level', do_refine=True),
                              f'level_m{m}'))
            for m in (2, 4, 6):
                cands.append((dict(mult=m, method='grow', balance_tol=0.10),
                              f'grow_m{m}_t10'))
        for m in mults:
            cands.append((dict(mult=m, method='grow'), f'grow_m{m}'))
        if budget == 0:
            # 小图额外试更严格的负载均衡（0.15），成本可控。
            for m in mults:
                cands.append((
                    dict(mult=m, method='grow', balance_tol=0.15),
                    f'grow_m{m}_t15'))
        if budget <= 1:
            # 小/中图跨核边延迟主导（见 OPTIMIZATION_NOTES 第五节）：
            # 按边条数而非字节切分，并优先少切。
            for m in (1, 2, 4):
                cands.append((dict(mult=m, method='grow_count'),
                              f'growcnt_m{m}'))
        if budget <= 1:
            # v10 网格实测：横切(level) 与 tol=0.10 在小中图上常胜出
            # （c008/c095 最高 +233%），此前只试过 level m1/m2 且不精化。
            for m in (2, 4):
                cands.append((dict(mult=m, method='level', do_refine=True),
                              f'level_m{m}'))
            for m in mults[:4]:
                cands.append((
                    dict(mult=m, method='grow', balance_tol=0.10),
                    f'grow_m{m}_t10'))
            cands.append((dict(mult=2, method='level', do_refine=False),
                          'level_m2'))
        elif budget == 1:
            # 中型 k=5 探针：mult=5/8 + tol=0.15 在 20/39 格有效，
            # 最大提升 143%（case_099）。其它核数也保留为通用候选。
            for m in (5, 8):
                cands.append((
                    dict(mult=m, method='grow', balance_tol=0.15),
                    f'grow_m{m}_t15'))
    else:
        if budget == 0:
            # P2/P3 小图同样受切分粒度影响；抽样实测补充 3/4/6 后
            # 部分格可提升 24%~260%（case_011/012/037/052）。
            mults = [1, 2, 3, 4, 6]
        elif budget == 1:
            mults = [1, 2]
        else:
            # 大图 P2/P3 此前只有 grow m1 + level + t10；补齐小图已验证的
            # 全部候选族（growcnt/growmv/np/mult 2·4），与 P1 对齐。
            mults = [1, 2, 4]
            for m in (2, 4):
                cands.append((dict(mult=m, method='level', do_refine=True),
                              f'level_m{m}'))
            for m in (2, 4, 6):
                cands.append((dict(mult=m, method='grow', balance_tol=0.10),
                              f'grow_m{m}_t10'))
            for m in (1, 2, 4):
                cands.append((dict(mult=m, method='grow_count'),
                              f'growcnt_m{m}'))
                cands.append((dict(mult=m, method='grow_mv'),
                              f'growmv_m{m}'))
        for m in mults:
            cands.append((dict(mult=m, method='grow'), f'grow_m{m}'))
        if budget == 0:
            cands.append((dict(mult=1, method='level', do_refine=False),
                          'level_m1'))
        elif budget == 1:
            # 中型 P2/P3 五核探针：m3/4/6/8 + tol=0.15 在 64/78
            # 格有效，最大提升 376%（case_060）。作为通用候选保留。
            for m in (3, 4, 6, 8):
                cands.append((
                    dict(mult=m, method='grow', balance_tol=0.15),
                    f'grow_m{m}_t15'))
            for m in (1, 2, 4):
                cands.append((dict(mult=m, method='grow_count'),
                              f'growcnt_m{m}'))
                cands.append((dict(mult=m, method='grow_mv'),
                              f'growmv_m{m}'))
        if budget >= 1:
            for frac in (0.4, 0.6):
                n_parts = max(2, int(round(k * frac)))
                if n_parts < k:
                    cands.append((dict(mult=1.0 * n_parts / k,
                                       method='grow_count'), f'growcnt_np{n_parts}'))
                    cands.append((dict(mult=1.0 * n_parts / k,
                                       method='grow_mv'), f'growmv_np{n_parts}'))
    # 低并行度用例：少用几个核往往更优（少切割、少等待）
    if par < 1.5 * k and k >= 3:
        cands.append((dict(mult=1.0 * max(2, k // 2) / k, method='grow'),
                      f'grow_half'))
    # M/V 二维均衡（OPTIMIZATION_NOTES 第七节，c040 +6.4%）
    if budget <= 1:
        for m in (1, 2, 4):
            cands.append((dict(mult=m, method='grow_mv'), f'growmv_m{m}'))
    # 子图数少于核数：mult 网格只给 n_parts=k,2k,4k...，5 核从不试 3 个子图。
    # P1 有 85 格始终未被多核方案超越，部分应由此缺口造成。
    if budget <= 1:
        for frac in (0.4, 0.6):
            n_parts = max(2, int(round(k * frac)))
            if n_parts < k:
                cands.append((dict(mult=1.0 * n_parts / k, method='grow'),
                              f'grow_np{n_parts}'))
                cands.append((dict(mult=1.0 * n_parts / k,
                                   method='grow_count'), f'growcnt_np{n_parts}'))

    # 结构候选（v9）：只在特征明显匹配时生成，避免无界搜索。
    layer_width = {}
    for op_id in gm.eligible:
        depth = gm.depth[op_id]
        layer_width[depth] = layer_width.get(depth, 0) + 1
    max_width = max(layer_width.values(), default=0)
    max_depth = max(layer_width, default=0)
    total_m = sum(gm.m_work.values())
    total_v = sum(gm.v_work.values())
    if (max_depth > 80 and max_width >= 4 * k
            and gm.parallelism > 4):
        cands.append((dict(method='wavefront', window=1),
                      'struct_wave_w1'))
        if budget <= 1:
            # v17 窗口扫描实测：w2 在部分深图上更优（case_028 k3）
            cands.append((dict(method='wavefront', window=2),
                          'struct_wave_w2'))
    if gm.parallelism > 100 and max_width >= 20 * k:
        cands.append((dict(method='wide'), 'struct_wide'))
    if (problem in (2, 3) and total_m == 0 and total_v > 0
            and max_depth > 500 and max_width <= 32):
        cands.append((dict(method='fork_template', stage_window=1),
                      'struct_fork_template'))
    # v31：grow 补种修复改变全部分区；部分图（case_011/027）旧行为
    # （前沿耗尽即止 + 散射兜底 + 精化）反而更优，两种行为并列保留。
    for m in (2, 4):
        cands.append((dict(mult=m, method='grow_ns'), f'growns_m{m}'))
        if budget <= 1:
            cands.append((dict(mult=m, method='grow_ns', balance_tol=0.10),
                          f'growns_m{m}_t10'))
    # v32：深度窗口×连通分量条带（天然无环）。深而稀疏的图上 grow
    # 补种后簇交错成环、只能回退横切串行（case_005/049/050/056 实测
    # S=1.0）；w=10 窗口下 case_049 P1 五核 S 1.00->2.00。
    if max_depth >= 30:
        for w in (6, 10, 16):
            cands.append((dict(method='bandcomp', window=w),
                          f'bandcomp_w{w}'))
        # v33：bandcomp + FM 精化再降切分流量（w14 实测 case_009 -25%、
        # case_035 -33%）；精化非法（成环）时回退原始条带。
        for w in (10, 14, 20):
            cands.append((dict(method='bandcomp_ref', window=w),
                          f'bandref_w{w}'))
        # v34：窗口细化 + 高粘性变体（小图 case_069/071 实测 sticky 1.0
        # 收益 5~7%，大图 0.5 仍优，两档并存）。
        for w in (8, 12, 18):
            cands.append((dict(method='bandcomp_ref', window=w),
                          f'bandref_w{w}'))
        for w in (10, 14):
            cands.append((dict(method='bandcomp_ref', window=w, sticky=1.0),
                          f'bandref_w{w}_s1'))
    return cands


def fallback_plan(gm, k):
    """全部算子放 1 个子图、0 号核：任何用例的保底方案。"""
    return {
        'node_to_subgraph': {str(op): 0 for op in gm.eligible},
        'core_schedules': [[0]] + [[] for _ in range(k - 1)],
    }


def build_plan(gm, problem, k, params):
    method = params.get('method')
    if method in {'wavefront', 'wide', 'fork_template'}:
        from structural_scheduler import (
            fork_join_plan, template_fork_join_plan,
            wavefront_plan, wide_plan)
        if method == 'wide':
            return wide_plan(gm, k, problem)
        if method == 'wavefront':
            return wavefront_plan(
                gm, k, problem, window=params.get('window', 1))
        return template_fork_join_plan(
            gm, k, problem, stage_window=params.get('stage_window', 1))
    if method == 'bandcomp':
        from scheduler import bandcomp_partition, plan_to_json
        op_group, core_tasks = bandcomp_partition(
            gm, k, window=params.get('window', 10),
            sticky=params.get('sticky', 0.5))
        return plan_to_json(op_group, core_tasks)
    if method == 'bandcomp_ref':
        from scheduler import (bandcomp_partition, fm_refine, list_schedule,
                               normalize, plan_to_json, rebalance,
                               _task_graph_ok)
        op_group, core_tasks = bandcomp_partition(
            gm, k, window=params.get('window', 14),
            sticky=params.get('sticky', 0.5))
        n_clusters = len(set(op_group.values()))
        scene = 'A' if problem == 1 else 'B'
        for n_parts in (n_clusters, k * 2, 8):
            try:
                refined = rebalance(gm, fm_refine(gm, dict(op_group),
                                                  n_parts,
                                                  balance_tol=0.30),
                                    n_parts, tol=0.30)
                refined = normalize(refined)
                if _task_graph_ok(gm, refined):
                    return plan_to_json(
                        refined, list_schedule(gm, refined, k, scene=scene))
            except Exception:
                continue
        return plan_to_json(op_group, core_tasks)
    if problem == 1:
        return build_plan_scene_a(gm, k, **params)
    return build_plan_scene_b(gm, k, **params)


def budget_of(n_elig):
    if n_elig <= 1500:
        return 0
    if n_elig <= 9000:
        return 1
    return 2


def run_case(case_name, evals, cfg, summary, lock):
    graph_path = DATA / f'{case_name}.json'
    graph = json.loads(graph_path.read_text(encoding='utf-8'))
    t0 = time.time()
    gm = GraphModel(graph)
    t_gm = time.time() - t0
    budget = budget_of(len(gm.eligible))
    entry = summary.setdefault(case_name, {})
    entry['meta'] = {
        'n_elig': len(gm.eligible),
        'total_work': int(gm.total_eff),
        'parallelism': round(gm.parallelism, 2),
        'budget': budget,
        'ver': 3,
    }
    entry.setdefault('runs', {})
    lock()

    # ---- 单核基线（同时是 P1 的全图单核兜底方案结果） ----
    if 'singlecore' not in entry:
        t0 = time.time()
        try:
            res = evals['sc'](graph, bandwidth=cfg['bandwidth'],
                              capacity=cfg['capacity'])
            entry['singlecore'] = {
                'makespan': res['makespan'],
                'added_copy_bytes': res['data_movement_bytes']['added_copy_bytes'],
                'secs': round(time.time() - t0, 1),
            }
        except Exception as exc:                    # noqa: BLE001
            entry['singlecore'] = {'error': str(exc)[:300]}
        lock()
    sc_ms = entry.get('singlecore', {}).get('makespan')

    plan_dir = RESULTS / 'best_plans'

    def save_plan(problem, k, plan, rec):
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / f'{case_name}_p{problem}_k{k}.json').write_text(
            json.dumps(plan), encoding='utf-8')
        return rec

    # ---- 各问题 × 核数 ----
    for problem in (1, 2, 3):
        for k in (2, 3, 4, 5):
            key = f'p{problem}_k{k}'
            if key in entry['runs']:
                continue
            best = None
            errors = []
            for params, label in candidates_for(gm, problem, k, budget):
                t0 = time.time()
                try:
                    plan = build_plan(gm, problem, k, params)
                    res = eval_plan(evals, problem, graph, plan, cfg)
                    rec = {
                        'label': label,
                        'makespan': res['makespan'],
                        'added_copy_bytes':
                            res['data_movement_bytes']['added_copy_bytes'],
                        'secs': round(time.time() - t0, 1),
                    }
                    if problem == 3:
                        cs = res.get('cache_stats', {})
                        rec['cache_hit_rate'] = round(
                            cs.get('hit_rate', 0.0), 4)
                    better = (best is None
                              or rec['makespan'] < best['makespan']
                              or (rec['makespan'] == best['makespan']
                                  and rec.get('added_copy_bytes', 0)
                                  < best.get('added_copy_bytes', 0)))
                    if better:
                        best = save_plan(problem, k, plan, rec)
                except Exception as exc:            # noqa: BLE001
                    errors.append(f'{label}: {str(exc)[:200]}')
            # 兜底：全图单核方案。P1 结果与单核基线完全一致，直接复用；
            # P2/P3 仅当候选都跑输单核基线时才花一次评估确认兜底。
            try:
                if problem == 1:
                    if sc_ms is not None and (best is None
                                              or sc_ms < best['makespan']):
                        best = save_plan(
                            problem, k, fallback_plan(gm, k),
                            {'label': 'fallback1core', 'makespan': sc_ms,
                             'added_copy_bytes':
                                 entry['singlecore'].get(
                                     'added_copy_bytes', 0), 'secs': 0.0})
                elif sc_ms is not None and (best is None
                                            or best['makespan'] > sc_ms):
                    t0 = time.time()
                    plan = fallback_plan(gm, k)
                    res = eval_plan(evals, problem, graph, plan, cfg)
                    better = (best is None
                              or res['makespan'] < best['makespan']
                              or (res['makespan'] == best['makespan']
                                  and res['data_movement_bytes']['added_copy_bytes']
                                  < best.get('added_copy_bytes', 0)))
                    if better:
                        rec = {
                            'label': 'fallback1core',
                            'makespan': res['makespan'],
                            'added_copy_bytes':
                                res['data_movement_bytes']['added_copy_bytes'],
                            'secs': round(time.time() - t0, 1),
                        }
                        if problem == 3:
                            cs = res.get('cache_stats', {})
                            rec['cache_hit_rate'] = round(
                                cs.get('hit_rate', 0.0), 4)
                        best = save_plan(problem, k, plan, rec)
            except Exception as exc:                # noqa: BLE001
                errors.append(f'fallback: {str(exc)[:200]}')
            run_rec = {'best': best}
            if errors:
                run_rec['errors'] = errors
            entry['runs'][key] = run_rec
            lock()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='*', default=[])
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--problems', nargs='*', type=int, default=[1, 2, 3])
    ap.add_argument('--cores', nargs='*', type=int, default=[2, 3, 4, 5])
    ap.add_argument('--summary-file', default='results/summary.json')
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    summary_path = HERE / args.summary_file

    if args.full:
        names = sorted(p.stem for p in DATA.glob('case_*.json'))
    elif args.smoke:
        names = ['case_001', 'case_029', 'case_032', 'case_064',
                 'case_069', 'case_071', 'case_093', 'case_100']
    else:
        names = args.cases

    # One process owns a summary file for the duration of a run. This prevents
    # two workers from reading the same snapshot and silently overwriting each
    # other's results. Use --summary-file for separate workers.
    with FileLock(summary_path):
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding='utf-8'))

        def lock():
            atomic_write_json(summary_path, summary)

        evals = load_evaluators()
        cfg = read_config()
        print(f'cases={len(names)} config={cfg}', flush=True)

        # 小用例先跑，尽快暴露问题
        def size_key(name):
            p = DATA / f'{name}.json'
            return p.stat().st_size
        names = sorted(names, key=size_key)

        t_start = time.time()
        for idx, name in enumerate(names):
            # 断点续跑：全部条目齐备且为当前算法版本则跳过
            ent = summary.get(name, {})
            if ent.get('meta', {}).get('ver') != 3 and ent.get('runs'):
                ent['runs'] = {}                       # 旧版本结果作废重跑
            def _cell_done(p, k):
                best = ent.get('runs', {}).get(f'p{p}_k{k}', {}).get('best')
                return (isinstance(best, dict)
                        and isinstance(best.get('makespan'), (int, float))
                        and best['makespan'] > 0)
            done = (ent.get('meta', {}).get('ver') == 3
                    and 'singlecore' in ent
                    and ent.get('singlecore', {}).get('makespan')
                    and all(_cell_done(p, k)
                            for p in args.problems for k in args.cores))
            if done:
                print(f'[{idx+1}/{len(names)}] {name}: skip (done)', flush=True)
                continue
            t0 = time.time()
            try:
                run_case(name, evals, cfg, summary, lock)
                sc = summary[name].get('singlecore', {})
                runs = summary[name].get('runs', {})
                msg = []
                for p in args.problems:
                    for k in args.cores:
                        r = runs.get(f'p{p}_k{k}', {}).get('best')
                        if r:
                            base = sc.get('makespan') or 1
                            msg.append(f"p{p}k{k}={r['makespan']:,}({base / r['makespan']:.2f}x)")
                print(f"[{idx+1}/{len(names)}] {name} in {time.time()-t0:.0f}s: "
                      + ' '.join(msg[:8]), flush=True)
            except Exception as exc:                    # noqa: BLE001
                traceback.print_exc()
                print(f'[{idx+1}/{len(names)}] {name} FAILED: {exc}', flush=True)
            lock()
        print(f'TOTAL {time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
