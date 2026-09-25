# -*- coding: utf-8 -*-
"""从逐格明细 CSV 生成评估结果总表（results/LEADERBOARD.md）。

口径：与 README / BENCHMARK_1200.md / 官方成绩一致，全部使用
「100 个用例加速比的算术平均」，不引入其他平均方式。

数据源：results/report/all_results.csv（每行一格，1200 行）。

用法：
  python src/leaderboard.py                # 写 results/LEADERBOARD.md
  python src/leaderboard.py --check        # 只做一致性与覆盖检查
"""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'results'
CSV_PATH = RESULTS / 'report' / 'all_results.csv'
OUT_PATH = RESULTS / 'LEADERBOARD.md'

CORES = ('2', '3', '4', '5')
PROBLEMS = (('1', '问题一（场景 A）'), ('2', '问题二（场景 B）'),
            ('3', '问题三（场景 B + Cache）'))


def load_rows():
    if not CSV_PATH.exists():
        raise SystemExit(f'missing {CSV_PATH}; see README for how to '
                         'produce it (run_experiments + make_report)')
    with open(CSV_PATH, encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    seen = set()
    for r in rows:
        key = (r['case'], r['problem'], r['cores'])
        if key in seen:
            raise SystemExit(f'duplicate cell in CSV: {key}')
        seen.add(key)
    if len(rows) != 1200:
        raise SystemExit(f'expected 1200 cells, got {len(rows)}')
    return rows


def build_markdown(rows):
    by_pk = defaultdict(list)
    for r in rows:
        by_pk[(r['problem'], r['cores'])].append(r)

    def mean_speedup(p, k):
        return statistics.mean(float(r['speedup'])
                               for r in by_pk[(str(p), str(k))])

    cache = {}
    for k in CORES:
        p2 = {r['case']: float(r['makespan']) for r in by_pk[('2', k)]}
        cache[k] = statistics.mean(
            p2[r['case']] / float(r['makespan']) for r in by_pk[('3', k)])

    adds = {}
    for p in ('1', '2', '3'):
        mb = [float(r['added_copy_bytes']) / 1048576
              for k in CORES for r in by_pk[(p, k)]]
        adds[p] = (statistics.mean(mb), statistics.median(mb), max(mb))

    L = []
    A = L.append
    A('# 评估结果总表\n')
    A('> 评估：赛题官方评估器（config.txt 固定参数）· Makespan 越小越好 · '
      '更完整的口径与解读见 [BENCHMARK_1200.md](BENCHMARK_1200.md)\n')
    A('## 三问题平均加速比（官方成绩口径）\n')
    A(r'100 个用例加速比的算术平均：$\overline{S}_{p,k}='
      r'\frac{1}{100}\sum_i T_{i,single}/T_{i,p,k}$。')
    A('')
    A('| 问题 | 2 核 | 3 核 | 4 核 | 5 核 |')
    A('|---|---|---|---|---|')
    for p, name in PROBLEMS:
        A('| %s | %s |' % (name, ' | '.join(
            '%.3f' % mean_speedup(p, k) for k in CORES)))
    A('')
    A('## 问题三 Cache 相对无 L2 的收益\n')
    A(r'同一核数下 $S^{L2}_k=T_{P2,k}/T_{P3,k}$，大于 1 表示 Cache 有收益。')
    A('')
    A('| 2 核 | 3 核 | 4 核 | 5 核 |')
    A('|---|---|---|---|---|')
    A('| %s |' % ' | '.join('%.3f' % cache[k] for k in CORES))
    A('')
    A('## 次要指标：新增搬运量\n')
    A('切分与核内溢出额外产生的拷贝量（MiB），同 Makespan 下的次要参考。\n')
    A('| 问题 | 平均 | 中位 | 最大 |')
    A('|---|---|---|---|')
    for p, name in PROBLEMS:
        A('| %s | %.2f | %.2f | %.1f |' % (name, adds[p][0], adds[p][1],
                                           adds[p][2]))
    A('')
    A('逐格明细见 [report/all_results.csv](report/all_results.csv)；'
      '口径定义、达标分布与解读见 [BENCHMARK_1200.md](BENCHMARK_1200.md)。'
      '全部数字可由该 CSV 直接重算。')
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='只校验 CSV 覆盖与数值，不写文件')
    args = ap.parse_args()
    rows = load_rows()

    bad = []
    for r in rows:
        try:
            sc, ms = float(r['singlecore_makespan']), float(r['makespan'])
            sp = float(r['speedup'])
        except (KeyError, ValueError):
            bad.append(f"{r['case']}|p{r['problem']}|k{r['cores']}: bad number")
            continue
        if ms <= 0 or sc <= 0 or sp <= 0:
            bad.append(f"{r['case']}|p{r['problem']}|k{r['cores']}: "
                       'non-positive value')
        elif abs(sc / ms - sp) > 0.001 * sp:
            bad.append(f"{r['case']}|p{r['problem']}|k{r['cores']}: "
                       f"speedup {sp} != {sc}/{ms} = {sc/ms:.4f}")
    if bad:
        for b in bad[:20]:
            print('ERROR:', b)
        raise SystemExit(f'{len(bad)} inconsistent rows')
    print('CSV check OK: 1200 cells, speedup = single/makespan consistent')

    if not args.check:
        OUT_PATH.write_text(build_markdown(rows), encoding='utf-8')
        print(f'wrote {OUT_PATH}')


if __name__ == '__main__':
    main()
