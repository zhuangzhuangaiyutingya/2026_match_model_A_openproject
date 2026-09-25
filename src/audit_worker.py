# -*- coding: utf-8 -*-
"""官方 CLI 逐格复放审计：单图 worker，串行跑本图的 12 格。

对每一格用赛题官方命令行入口独立进程重评 results/best_plans/ 中的方案：

  python3 multicore_cut_evaluate_problem_{p}.py <case.json> <plan.json>
      --config data/config.txt -o results/audit/{case}_p{p}_k{k}.json

若存在 results/leaderboard/baseline.json（评测记录），则同时核对
makespan；否则仅留档。逐图汇总写 results/audit/{case}_audit.json。

依赖：赛题附件 data/ 与 code/（获取方式见 README）。

用法：python3 src/audit_worker.py --case case_001
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'data'
CODE = ROOT / 'code'
AUDIT = ROOT / 'results' / 'audit'
SHARD = ROOT / 'results' / 'leaderboard' / 'baseline.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    args = ap.parse_args()
    case = args.case
    AUDIT.mkdir(parents=True, exist_ok=True)

    if not DATA.exists() or not CODE.exists():
        raise SystemExit('missing problem attachment: put the official '
                         'data/ and code/ next to src/ (see README)')

    shard_cells = {}
    if SHARD.exists():
        shard_cells = json.loads(SHARD.read_text(encoding='utf-8'))['cells']

    cells_out = {}
    for p in (1, 2, 3):
        for k in (2, 3, 4, 5):
            key = f'{case}|p{p}|k{k}'
            shard_ms = shard_cells.get(key, {}).get('makespan')
            plan_path = ROOT / 'results' / 'best_plans' / f'{case}_p{p}_k{k}.json'
            out_path = AUDIT / f'{case}_p{p}_k{k}.json'
            t0 = time.time()
            if not plan_path.exists():
                cells_out[key] = {'error': 'plan missing'}
                print(f'{key}: plan missing', flush=True)
                continue
            cmd = [sys.executable,
                   str(CODE / f'multicore_cut_evaluate_problem_{p}.py'),
                   str(DATA / f'{case}.json'), str(plan_path),
                   '--config', str(DATA / 'config.txt'),
                   '-o', str(out_path)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=3600)
                if r.returncode != 0:
                    cells_out[key] = {'error': (r.stderr or '')[-200:]}
                    print(f'{key}: CLI rc={r.returncode}', flush=True)
                    continue
                cli = json.loads(out_path.read_text(encoding='utf-8'))
                cli_ms = cli['makespan']
                rec = {
                    'cli_makespan': cli_ms,
                    'added_copy_bytes':
                        cli['data_movement_bytes']['added_copy_bytes'],
                    'cache_hit_rate':
                        cli.get('cache_stats', {}).get('hit_rate', 0),
                    'secs': round(time.time() - t0, 1),
                }
                if shard_ms is not None:
                    rec['shard_makespan'] = shard_ms
                    rec['match'] = bool(abs(cli_ms - shard_ms) < 0.5)
                cells_out[key] = rec
                tail = f'{"OK" if rec.get("match") else "MISMATCH"}' \
                    if shard_ms is not None else 'archived'
                print(f'{key}: cli={cli_ms} {tail} '
                      f'({time.time()-t0:.0f}s)', flush=True)
            except subprocess.TimeoutExpired:
                cells_out[key] = {'error': 'timeout 3600s'}
                print(f'{key}: TIMEOUT', flush=True)
            except Exception as exc:  # noqa: BLE001
                cells_out[key] = {'error': repr(exc)[:200]}
                print(f'{key}: {exc!r}', flush=True)

    (AUDIT / f'{case}_audit.json').write_text(
        json.dumps({'case': case, 'cells': cells_out}, ensure_ascii=False,
                   indent=1),
        encoding='utf-8')
    n_ok = sum(1 for c in cells_out.values() if 'error' not in c)
    n_match = sum(1 for c in cells_out.values() if c.get('match'))
    print(f'{case}: done ok={n_ok}/12 match={n_match}/12', flush=True)


if __name__ == '__main__':
    main()
