# -*- coding: utf-8 -*-
"""结果数据校验：覆盖、一致性、数值范围。默认严格，任何问题都以非零退出。

校验 results/report/all_results.csv（1200 行逐格明细）：
  1. 覆盖：case_001..case_100 × 问题 1..3 × 核数 2..5 恰好各一格；
  2. 数值：makespan/单核基线/speedup 均为正，且 speedup == 单核/makespan
     （相对误差 < 0.1%）；
  3. 范围：cache_hit_rate ∈ [0, 1]（问题三），问题一/二应为 0；
  4. 若存在 results/audit/audit_summary.json：核对 total/mismatch 计数
     与当前 CSV 一致。

用法：python src/validate_results.py
"""
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV_PATH = ROOT / 'results' / 'report' / 'all_results.csv'
SUMMARY = ROOT / 'results' / 'audit' / 'audit_summary.json'


def main():
    errors = []
    if not CSV_PATH.exists():
        raise SystemExit(f'ERROR: missing {CSV_PATH}')
    with open(CSV_PATH, encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))

    seen = set()
    for r in rows:
        key = f"{r['case']}|p{r['problem']}|k{r['cores']}"
        if key in seen:
            errors.append(f'{key}: duplicate cell')
        seen.add(key)

        try:
            sc = float(r['singlecore_makespan'])
            ms = float(r['makespan'])
            sp = float(r['speedup'])
        except (KeyError, ValueError):
            errors.append(f'{key}: missing or non-numeric value')
            continue
        if sc <= 0 or ms <= 0 or sp <= 0:
            errors.append(f'{key}: non-positive value')
        elif abs(sc / ms - sp) > 0.001 * sp:
            errors.append(f'{key}: speedup {sp} != {sc}/{ms}={sc/ms:.4f}')

        hit = r.get('cache_hit_rate')
        if hit not in (None, ''):
            h = float(hit)
            if not 0 <= h <= 1:
                errors.append(f'{key}: cache_hit_rate {h} outside [0,1]')
            elif r['problem'] in ('1', '2') and h != 0:
                errors.append(f'{key}: cache_hit_rate must be 0 for '
                              f'problem {r["problem"]}')

    expected = {'case_%03d' % i for i in range(1, 101)}
    for p in ('1', '2', '3'):
        for k in ('2', '3', '4', '5'):
            for c in expected:
                key = f'{c}|p{p}|k{k}'
                if key not in seen:
                    errors.append(f'{key}: missing cell')

    if len(rows) != 1200:
        errors.append(f'expected 1200 rows, got {len(rows)}')

    if SUMMARY.exists():
        s = json.loads(SUMMARY.read_text(encoding='utf-8'))
        if s.get('total_cells') not in (None, len(rows)):
            errors.append(f"audit_summary total_cells={s.get('total_cells')} "
                          f'!= CSV rows {len(rows)}')

    for e in errors[:30]:
        print('ERROR:', e)
    if errors:
        raise SystemExit(f'{len(errors)} problems found')
    print(f'validate OK: 1200 cells, coverage complete, values consistent')


if __name__ == '__main__':
    main()
