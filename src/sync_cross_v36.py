# -*- coding: utf-8 -*-
"""v36：全量交叉评估——把每个问题的赢家方案在其他问题的评估器下重评。

方案格式通用（node_to_subgraph + core_schedules），跨问题合法且免费。
P3←P2 已有 sync_p3_from_p2；本脚本覆盖其余 5 个方向（P2→P1、P3→P1、
P1→P2、P3→P2、P1→P3），仅当新 makespan 更小时写 v36_cross_<dst>_from_<src>
标签进 ZZW 分片与 best_plans。

用法：
  python sync_cross_v36.py            # 全量
  python sync_cross_v36.py --cases case_005 case_064 ...   # 定向
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATT = HERE.parent
CODE = ATT / 'code'
DATA = ATT / 'data'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))

DIRECTIONS = [('p2', 'p1'), ('p3', 'p1'), ('p1', 'p2'),
              ('p3', 'p2'), ('p1', 'p3'), ('p2', 'p3')]
CORES = [2, 3, 4, 5]

_WORKER = {}


def _init():
    from run_experiments import load_evaluators, read_config
    _WORKER['evals'] = load_evaluators()
    _WORKER['cfg'] = read_config()


def _job(spec):
    case, src_p, dst_p, k = spec
    from run_experiments import eval_plan
    plan_path = HERE / 'results' / 'best_plans' / f'{case}_{src_p}_k{k}.json'
    if not plan_path.exists():
        return None
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    graph = json.loads((DATA / f'{case}.json').read_text(encoding='utf-8'))
    t0 = time.time()
    try:
        res = eval_plan(_WORKER['evals'], int(dst_p[1]), graph, plan,
                        _WORKER['cfg'])
    except Exception as exc:  # noqa: BLE001
        return None
    dm = res['data_movement_bytes']
    return dict(case=case, src=src_p, dst=dst_p, k=k,
                makespan=res['makespan'],
                added=dm['added_copy_bytes'],
                hit=res.get('cache_stats', {}).get('hit_rate', 0),
                secs=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='*', default=None)
    ap.add_argument('--dry', action='store_true')
    args = ap.parse_args()

    summary = {}
    for name in ('summary.json', 'summary_complete.json'):
        p = HERE / 'results' / name
        if p.exists():
            for c, e in json.loads(p.read_text(encoding='utf-8')).items():
                sc = e.get('singlecore', {})
                if isinstance(sc.get('makespan'), (int, float)):
                    summary[c] = sc['makespan']

    cases = args.cases
    if not cases:
        cases = sorted(p.stem for p in DATA.glob('case_*.json'))

    specs = [(c, s, d, k) for c in cases for s, d in DIRECTIONS for k in CORES]
    print(f'{len(specs)} cross evaluations', flush=True)

    t0 = time.time()
    results = []
    with Pool(processes=6, initializer=_init) as pool:
        for i, rec in enumerate(pool.imap_unordered(_job, specs)):
            if rec:
                results.append(rec)
            if (i + 1) % 100 == 0:
                print(f'[{i+1}/{len(specs)}] {time.time()-t0:.0f}s', flush=True)
    print(f'evaluated {len(results)} in {time.time()-t0:.0f}s', flush=True)

    # 找改进：目标格当前最优 makespan（读分片）
    shard_path = HERE / 'results' / 'leaderboard' / 'baseline.json'
    from collab_io import FileLock, atomic_write_json
    cells = json.loads(shard_path.read_text(encoding='utf-8'))['cells']

    improved = []
    for rec in results:
        key = f"{rec['case']}|{rec['dst']}|k{rec['k']}"
        cur = cells.get(key, {}).get('makespan')
        if cur is None or rec['makespan'] < cur - 1:
            improved.append((key, rec))
    # 同一目标格可能有两个来源方向同时改进，保留更优者
    best_by_key = {}
    for key, rec in improved:
        prev = best_by_key.get(key)
        if prev is None or rec['makespan'] < prev['makespan']:
            best_by_key[key] = rec
    improved = sorted(best_by_key.items())
    print(f'improving candidates: {len(improved)}', flush=True)
    if args.dry:
        for key, rec in improved:
            print(f'  {key}: {cells.get(key, {}).get("makespan")} -> '
                  f'{rec["makespan"]} (from {rec["src"]})')
        return

    with FileLock(shard_path):
        cells = json.loads(shard_path.read_text(encoding='utf-8'))['cells']
        n = 0
        for key, rec in improved:
            cur = cells.get(key, {}).get('makespan')
            if cur is not None and rec['makespan'] >= cur:
                continue
            case = rec['case']
            t1 = summary.get(case)
            if t1 is None:
                continue
            label = f"v36_cross_{rec['dst']}_from_{rec['src']}"
            rec_new = {
                'makespan': rec['makespan'],
                'added_copy_bytes': rec['added'],
                'speedup': round(t1 / rec['makespan'], 6),
                'label': label,
                'cache_hit_rate': rec['hit'],
            }
            cells[key] = rec_new
            atomic_write_json(
                HERE / 'results' / 'best_plans' /
                f"{case}_{rec['dst']}_k{rec['k']}.json",
                json.loads((HERE / 'results' / 'best_plans' /
                            f"{case}_{rec['src']}_k{rec['k']}.json")
                           .read_text(encoding='utf-8')))
            n += 1
        atomic_write_json(shard_path, {
            **json.loads(shard_path.read_text(encoding='utf-8')),
            'cells': cells})
    print(f'updated {n} cells', flush=True)


if __name__ == '__main__':
    main()
