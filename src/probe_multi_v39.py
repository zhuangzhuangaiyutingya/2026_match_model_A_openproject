# -*- coding: utf-8 -*-
"""v39 多方向探测：size_slack 档位 / FM 轮数 / bandref 容差 /
rank 溢出权重 / bandcomp 装核扰动，逐方向逐格与榜单比。

用法：
  python3 probe_multi_v39.py --case case_027 --cores 5 --family slack
  families: slack | passes | btol | rankw | bseed
"""
import argparse, gzip, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'code')]
from run_experiments import eval_plan, load_evaluators, read_config
from scheduler import (GraphModel, bandcomp_partition, fm_refine, list_schedule,
                       normalize, plan_to_json, rebalance, region_grow,
                       _task_graph_ok)


def make_plan(gm, problem, k, params):
    """按方向参数组装方案（bandcomp 管道或 grow 管道）。"""
    fam = params['family']
    scene = 'A' if problem == 1 else 'B'
    if fam in ('btol', 'bseed'):
        op_group, core_tasks = bandcomp_partition(
            gm, k, window=params.get('window', 10),
            sticky=params.get('sticky', 0.5),
            seed=params.get('seed', 0))
        if fam == 'btol':
            n_clusters = len(set(op_group.values()))
            for n_parts in (n_clusters, k * 2, 8):
                try:
                    ref = rebalance(gm, fm_refine(gm, dict(op_group), n_parts,
                                                  balance_tol=params['tol']),
                                    n_parts, tol=params['tol'])
                    refn = normalize(ref)
                    if _task_graph_ok(gm, refn):
                        return plan_to_json(
                            refn, list_schedule(gm, refn, k, scene=scene))
                except Exception:
                    continue
            return plan_to_json(op_group, core_tasks)
        return plan_to_json(op_group, core_tasks)
    # grow 管道：slack / passes / rankw
    n_parts = k * params.get('mult', 2)
    op_group = region_grow(gm, n_parts, size_slack=params.get('slack', 0.30))
    try:
        ref = rebalance(gm, fm_refine(gm, dict(op_group), n_parts,
                                      balance_tol=0.30,
                                      passes=params.get('passes', 4)),
                        n_parts, tol=0.30)
        refn = normalize(ref)
        if _task_graph_ok(gm, refn):
            op_group = refn
    except Exception:
        refn = normalize(op_group)
        if _task_graph_ok(gm, refn):
            op_group = refn
    return plan_to_json(op_group, list_schedule(gm, op_group, k, scene=scene,
                                                rank_spill_w=params.get('rankw', 1.5)))


FAMILIES = {
    'slack':  [dict(family='slack', slack=s, mult=m) for s in (0.20, 0.25, 0.35, 0.40) for m in (2, 4)],
    'passes': [dict(family='passes', passes=n, mult=m) for n in (2, 8, 12) for m in (2, 4)],
    'btol':   [dict(family='btol', tol=t, window=w) for t in (0.15, 0.20, 0.40) for w in (10, 14)],
    'rankw':  [dict(family='rankw', rankw=w, mult=m) for w in (1.0, 2.0, 3.0) for m in (2, 4)],
    'bseed':  [dict(family='bseed', seed=s, window=w) for s in (1, 2, 3, 4) for w in (10, 14)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--cores', type=int, required=True)
    ap.add_argument('--family', required=True, choices=sorted(FAMILIES))
    args = ap.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    graph = json.loads((HERE.parent / 'data' / f'{args.case}.json').read_text(encoding='utf-8'))
    gm = GraphModel(graph)
    baseline = json.loads(Path(args.baseline).read_text(encoding='utf-8'))['cells']
    evals, cfg = load_evaluators(), read_config()
    winners = []
    for problem in (1, 2, 3):
        key = f'{args.case}|p{problem}|k{args.cores}'
        old = baseline[key]['makespan']
        best = None
        best_params = None
        for params in FAMILIES[args.family]:
            try:
                plan = make_plan(gm, problem, args.cores, params)
                res = eval_plan(evals, problem, graph, plan, cfg)
                tag = '-'.join(f'{k2}{v}' for k2, v in sorted(params.items())
                               if k2 != 'family')
                rec = {'label': f'v39_{args.family}_{tag}',
                       'makespan': res['makespan'], 'plan': plan,
                       'added_copy_bytes':
                           res['data_movement_bytes']['added_copy_bytes'],
                       'cache_hit_rate':
                           res.get('cache_stats', {}).get('hit_rate', 0)}
                if best is None or rec['makespan'] < best['makespan']:
                    best, best_params = rec, params
            except Exception:
                continue
        if best and best['makespan'] < old:
            winners.append({'key': key, 'old_makespan': old,
                            'makespan': best['makespan'], **best})
            print(f"{args.case} {key}: {old:,} -> {best['makespan']:,} "
                  f"(+{(old/best['makespan']-1)*100:.1f}%) {best['label']}",
                  flush=True)
        else:
            print(f'{args.case} {key}: no improvement '
                  f'(best {best["makespan"] if best else "?"} vs {old})',
                  flush=True)
        with gzip.open(args.output, 'wt', encoding='utf-8') as fh:
            json.dump({'case': args.case, 'family': args.family,
                       'winners': winners}, fh, ensure_ascii=False)


if __name__ == '__main__':
    main()
