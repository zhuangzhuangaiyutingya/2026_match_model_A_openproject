# -*- coding: utf-8 -*-
"""批量审计：对所有用例并行运行 audit_worker。

用法：
  python src/run_audit.py                # 100 图全部
  python src/run_audit.py --cases case_001 case_002
  python src/run_audit.py --jobs 8       # 并行进程数（默认 6）

结果：results/audit/{case}_audit.json 逐图留档；
     results/audit/audit_summary.json 汇总（与提交时口径一致）。
"""
import argparse
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AUDIT = ROOT / 'results' / 'audit'


def run_case(case):
    t0 = time.time()
    r = subprocess.run([sys.executable, str(HERE / 'audit_worker.py'),
                        '--case', case],
                       capture_output=True, text=True, timeout=7200)
    tail = (r.stdout or '').strip().splitlines()[-1:] or ['']
    return case, r.returncode, tail[0], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='*')
    ap.add_argument('--jobs', type=int, default=6)
    args = ap.parse_args()
    cases = args.cases or sorted(
        p.stem for p in (ROOT / 'results' / 'best_plans').glob('case_*_p1_k2.json'))
    cases = sorted(set(c.rsplit('_p1_k2', 1)[0] for c in cases)) if \
        all('_p1_k2' in c for c in cases) else sorted(cases)
    print(f'{len(cases)} cases, {args.jobs} workers', flush=True)
    t0 = time.time()
    with Pool(processes=args.jobs) as pool:
        for case, rc, tail, dt in pool.imap_unordered(run_case, cases):
            print(f'[{case}] rc={rc} {tail} ({dt:.0f}s)', flush=True)
    print(f'all done in {(time.time()-t0)/60:.0f} min', flush=True)
    _summarize()


def _summarize():
    total = match = err = 0
    mismatch_cells = {}
    for p in sorted(AUDIT.glob('case_*_audit.json')):
        d = json.loads(p.read_text(encoding='utf-8'))
        for key, rec in d['cells'].items():
            total += 1
            if 'error' in rec:
                err += 1
                continue
            if rec.get('match') is False:
                mismatch_cells[key] = {
                    'cli': rec['cli_makespan'],
                    'recorded': rec.get('shard_makespan'),
                }
            elif rec.get('match'):
                match += 1
    summary = {
        'definition': '官方 CLI 逐格独立进程复放（config.txt 固定参数）',
        'total_cells': total,
        'match': match,
        'mismatch': len(mismatch_cells),
        'errors': err,
        'mismatch_note': ('P2/P3 官方评估器跨进程非确定：同一方案不同进程的 '
                 'makespan 与新增搬运量存在 0.1%~1.6% 的运行间微差'
                 '（进程内确定）。'),
        'mismatch_cells': mismatch_cells,
    }
    out = AUDIT / 'audit_summary.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    print(f'summary: total={total} match={match} mismatch={len(mismatch_cells)} '
          f'errors={err} -> {out}', flush=True)


if __name__ == '__main__':
    main()
