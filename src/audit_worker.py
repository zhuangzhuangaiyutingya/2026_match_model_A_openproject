# -*- coding: utf-8 -*-
"""官方 CLI 全量审计：单图 worker，串行跑本图的 12 格。

每格用官方 CLI 入口重评：
  python3 multicore_cut_evaluate_problem_{p}.py <case.json> <plan.json>
      --config data/config.txt -o audit/{case}_p{p}_k{k}.json
然后与分片 makespan 核对，追加写 audit/{case}_audit.json：
  {case, cells: {key: {cli_makespan, shard_makespan, match,
                       added_copy_bytes, cache_hit_rate}}}

用法：python3 audit_worker.py --case case_001
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent / 'code'
DATA = HERE.parent / 'data'
AUDIT = HERE / 'results' / 'audit'
SHARD = HERE / 'results' / 'leaderboard' / 'baseline.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    args = ap.parse_args()
    case = args.case
    AUDIT.mkdir(parents=True, exist_ok=True)

    shard = json.loads(SHARD.read_text(encoding='utf-8'))['cells']
    cells_out = {}
    for p in (1, 2, 3):
        for k in (2, 3, 4, 5):
            key = f'{case}|p{p}|k{k}'
            shard_ms = shard.get(key, {}).get('makespan')
            plan_path = HERE / 'results' / 'best_plans' / f'{case}_p{p}_k{k}.json'
            out_path = AUDIT / f'{case}_p{p}_k{k}.json'
            t0 = time.time()
            if not plan_path.exists():
                cells_out[key] = {'error': 'plan missing'}
                print(f'{key}: plan missing', flush=True)
                continue
            cmd = [sys.executable, str(CODE / f'multicore_cut_evaluate_problem_{p}.py'),
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
                match = (shard_ms is not None
                         and abs(cli_ms - shard_ms) < 0.5)
                cells_out[key] = {
                    'cli_makespan': cli_ms,
                    'shard_makespan': shard_ms,
                    'match': bool(match),
                    'added_copy_bytes': cli['data_movement_bytes']['added_copy_bytes'],
                    'cache_hit_rate': cli.get('cache_stats', {}).get('hit_rate', 0),
                    'secs': round(time.time() - t0, 1),
                }
                print(f'{key}: cli={cli_ms} shard={shard_ms} '
                      f'{"OK" if match else "MISMATCH"} '
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
    n_match = sum(1 for c in cells_out.values() if c.get('match'))
    n_err = sum(1 for c in cells_out.values() if 'error' in c)
    print(f'{case}: done match={n_match}/12 errors={n_err}', flush=True)


if __name__ == '__main__':
    main()
