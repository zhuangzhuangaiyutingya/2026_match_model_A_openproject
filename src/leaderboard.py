# -*- coding: utf-8 -*-
"""内榜：候选算法注册 -> 官方评估器打分 -> 榜单更新。

用法：
  1. 写一个候选模块 candidates/cand_<name>.py，暴露：
       def generate_plan(gm, problem, k, graph=None): -> plan dict
     gm 是 scheduler.GraphModel；返回标准方案 JSON 结构。
     可选暴露 CAND_META dict（作者、说明）。
  2. python leaderboard.py --candidate <name> [--cases case_001 ... | --all]
     只评估该候选，结果写入 results/leaderboard/<name>.json。
  3. python leaderboard.py --show          # 打印当前榜单
  4. python leaderboard.py --show --cells  # 附逐格冠军表

规则：
  - 评估口径 = 官方评估器（不可改），单核基线复用 summary 合并结果。
  - 主指标 = 几何平均加速比（相对单核基线，全部已评用例）。
  - 每格 (case, problem, K) 记录当前最优 makespan 及持有候选。
"""
import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

from collab_io import atomic_write_json, atomic_write_text

HERE = Path(__file__).resolve().parent
ATT = HERE.parent
CODE = ATT / 'code'
DATA = ATT / 'data'
RESULTS = HERE / 'results'
CANDS = HERE / 'candidates'
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(HERE))

from scheduler import GraphModel  # noqa: E402


def merged_summary():
    """读取规范化完整汇总；未生成时兼容合并历史分片。"""
    complete = RESULTS / 'summary_complete.json'
    if complete.exists():
        return json.loads(complete.read_text(encoding='utf-8'))
    merged = {}
    for path in [RESULTS / 'summary.json'] + sorted(
            RESULTS.glob('summary_*.json')):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        for case, ent in data.items():
            cur = merged.setdefault(case, {})
            if 'singlecore' not in cur and 'singlecore' in ent:
                cur['singlecore'] = ent['singlecore']
            runs = cur.setdefault('runs', {})
            for key, rec in ent.get('runs', {}).items():
                runs.setdefault(key, rec)
    return merged


def load_baseline():
    """单核基线 {(case): makespan}。"""
    summary = merged_summary()
    base = {}
    for case, ent in summary.items():
        ms = ent.get('singlecore', {}).get('makespan')
        if ms:
            base[case] = ms
    return base


def eval_plan_funcs():
    from run_experiments import load_evaluators, read_config, eval_plan
    return load_evaluators(), read_config(), eval_plan


def load_board():
    """载入全部候选分片：results/leaderboard/<name>.json 每候选一文件。

    多人协作时各自只写自己名字的分片，天然无合并冲突；
    旧单体 leaderboard.json 自动迁移为分片后停用。
    """
    board = {}
    legacy = RESULTS / 'leaderboard.json'
    if legacy.exists() and not (RESULTS / 'leaderboard').exists():
        legacy_data = json.loads(legacy.read_text(encoding='utf-8'))
        (RESULTS / 'leaderboard').mkdir(parents=True, exist_ok=True)
        for name, rec in legacy_data.items():
            atomic_write_json(RESULTS / 'leaderboard' / f'{name}.json', rec)
            board[name] = rec
        legacy.rename(RESULTS / 'leaderboard_legacy.bak.json')
        return board
    for f in sorted((RESULTS / 'leaderboard').glob('*.json')) \
            if (RESULTS / 'leaderboard').exists() else []:
        board[f.stem] = json.loads(f.read_text(encoding='utf-8'))
    return board


def save_candidate(name, rec):
    """候选结果写自己的分片文件。"""
    shard_dir = RESULTS / 'leaderboard'
    shard_dir.mkdir(exist_ok=True)
    atomic_write_json(shard_dir / f'{name}.json', rec)


def evaluate_candidate(name, cases, verbose=True):
    mod = importlib.import_module(f'candidates.cand_{name}')
    gen = getattr(mod, 'generate_plan', None)
    if gen is None:
        raise SystemExit(f'candidates/cand_{name}.py 缺少 generate_plan')
    meta = getattr(mod, 'CAND_META', {'author': '?', 'desc': ''})

    evals, cfg, eval_plan = eval_plan_funcs()
    base = load_baseline()
    bounds = load_bounds()
    if not cases:
        cases = sorted(base)          # 只评已有单核基线的用例
    cand_rec = load_board().get(name, {'meta': meta, 'cells': {}})
    cand_rec['meta'] = meta
    cells = cand_rec['cells']

    t_all = time.time()
    for idx, case in enumerate(cases):
        graph = json.loads(
            (DATA / f'{case}.json').read_text(encoding='utf-8'))
        gm = GraphModel(graph)
        for problem in (1, 2, 3):
            for k in (2, 3, 4, 5):
                key = f'{case}|p{problem}|k{k}'
                try:
                    plan = gen(gm, problem, k)
                    res = eval_plan(evals, problem, graph, plan, cfg)
                    ms = res['makespan']
                    bound_rec = bounds[case]['cores'][str(k)]
                    cells[key] = {
                        'makespan': ms,
                        'added_copy_bytes':
                            res['data_movement_bytes']['added_copy_bytes'],
                        'speedup': round(base[case] / ms, 6),
                        'ideal_speedup': bound_rec['ideal_speedup'],
                        'attainment': round(
                            bound_rec['lower_bound'] / ms, 6),
                    }
                except Exception as exc:              # noqa: BLE001
                    cells[key] = {'error': str(exc)[:200]}
        if verbose and (idx + 1) % 10 == 0:
            ok = sum(1 for v in cells.values() if 'makespan' in v)
            print(f'  [{idx+1}/{len(cases)}] cells={ok}', flush=True)
        save_candidate(name, cand_rec)
    cand_rec['evaluated_at'] = time.time()
    cand_rec['secs'] = round(time.time() - t_all, 1)
    save_candidate(name, cand_rec)
    if verbose:
        show(load_board())
    return load_board()


def geo_mean(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else 0.0


def show(board, cells_too=False):
    base = load_baseline()
    all_cases = set(base)
    stats = []
    for name, rec in board.items():
        good = {k: v for k, v in rec['cells'].items()
                if 'makespan' in v and v.get('speedup', 0) > 0}
        cases_hit = {k.split('|')[0] for k in good} & all_cases
        # 单方法榜：对全部有效格子的加速比直接取几何平均。
        cell_speedups = [value['speedup'] for value in good.values()]
        stats.append({
            'name': name,
            'author': rec.get('meta', {}).get('author', '?'),
            'cells': len(good),
            'geo_speedup': round(geo_mean(cell_speedups), 4)
            if cell_speedups else 0,
            'cases': len(cases_hit),
        })
    stats.sort(key=lambda s: -s['geo_speedup'])
    print(f"\n{'rank':<5}{'candidate':<20}{'author':<10}"
          f"{'geo-speedup':<12}{'cells':>8}{'cases':>8}")
    for i, s in enumerate(stats, 1):
        print(f"{i:<5}{s['name']:<20}{str(s['author']):<10}"
              f"{s['geo_speedup']:<12}{s['cells']:>8}{s['cases']:>8}")
    # 合流榜：逐格最优
    if cells_too and stats:
        owners = {}
        for name, rec in board.items():
            for k, v in rec['cells'].items():
                if 'makespan' not in v:
                    continue
                if k not in owners or v['makespan'] < owners[k][1]:
                    owners[k] = (name, v['makespan'])
        from collections import Counter
        win = Counter(n for n, _ in owners.values())
        print('\n逐格冠军分布：')
        for n, c in win.most_common():
            print(f'  {n}: {c} 格')
        merged_speedups = [
            base[key.split('|')[0]] / makespan
            for key, (_, makespan) in owners.items()
            if key.split('|')[0] in base and makespan > 0
        ]
        coverage = len(merged_speedups)
        suffix = '' if coverage == 1200 else '（覆盖不完整）'
        print(
            f'合流逐格几何平均加速比: {geo_mean(merged_speedups):.4f} '
            f'[{coverage}/1200]{suffix}'
        )
    return stats


STATUS_RULES = (
    ('💎', 'A ≥ 0.90，且绝对加速比达到对应核数的优秀档'),
    ('🔵', 'A ≥ 0.85（但未达到 💎 条件）'),
    ('🟢', '0.70 ≤ A < 0.85'),
    ('🟡', '0.50 ≤ A < 0.70'),
    ('🟠', 'S ≥ 1.00，且 A < 0.50'),
    ('🔴', 'S < 1.00（劣于单核，最差状态，判定时优先于其余规则）'),
)
EXCELLENT_SPEEDUP = {2: 1.65, 3: 2.20, 4: 2.80, 5: 3.30}


def load_bounds():
    path = RESULTS / 'theoretical_bounds.json'
    if not path.exists():
        raise FileNotFoundError(
            f'{path} is required to compute attainment A; '
            'run: python theoretical_bounds.py')
    data = json.loads(path.read_text(encoding='utf-8'))
    cases = data.get('cases', {})
    if not cases:
        raise ValueError(f'{path}: missing cases lower-bound data')
    return cases


def cell_quality(speedup, attainment, num_cores):
    """Return the mutually exclusive status icon for one leaderboard cell."""
    if speedup < 1.0:
        return '🔴'
    if attainment >= 0.90 and speedup >= EXCELLENT_SPEEDUP[num_cores]:
        return '💎'
    if attainment >= 0.85:
        return '🔵'
    if attainment >= 0.70:
        return '🟢'
    if attainment >= 0.50:
        return '🟡'
    return '🟠'


def owner_map(board):
    """{cell_key: (holder_name, makespan)} selecting the minimum makespan."""
    owners = {}
    for name, rec in board.items():
        for key, value in rec['cells'].items():
            if 'makespan' not in value:
                continue
            if key not in owners or value['makespan'] < owners[key][1]:
                owners[key] = (name, value['makespan'])
    return owners


def write_md(board):
    """Generate the official-metric-first GitHub dashboard."""
    from dashboard import write_dashboard
    return write_dashboard(
        path=HERE / 'LEADERBOARD.md', board=board,
        summary=merged_summary(), baselines=load_baseline(),
        bounds=load_bounds(), owners=owner_map(board),
        status_rules=STATUS_RULES, quality_function=cell_quality)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidate')
    ap.add_argument('--cases', nargs='*', default=[])
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--cells', action='store_true')
    ap.add_argument('--write-md', action='store_true')
    args = ap.parse_args()
    CANDS.mkdir(exist_ok=True)
    if args.candidate:
        evaluate_candidate(args.candidate, args.cases)
    board = load_board()
    if args.show:
        show(board, args.cells)
    if args.write_md or args.candidate:
        out = write_md(board)
        print(f'LEADERBOARD.md -> {out}')


if __name__ == '__main__':
    main()
