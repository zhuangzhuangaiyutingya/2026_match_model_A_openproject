# -*- coding: utf-8 -*-
"""生成论文用结果表与曲线。

多核结果一律取自 results/leaderboard/*.json 候选分片（官方评估器逐格产出，
与 LEADERBOARD.md 同源）；单核基线与用例元信息取自实验汇总快照。
无分片时回退读取 summary*.json（旧数据源，仅作过渡）。
"""
import csv
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
REPORT = HERE / 'report'
REPORT.mkdir(exist_ok=True)
CORES = [1, 2, 3, 4, 5]


def _load_legacy_summary():
    complete = RESULTS / 'summary_complete.json'
    if complete.exists():
        return json.loads(complete.read_text(encoding='utf-8'))
    files = [RESULTS / 'summary.json'] + sorted(
        RESULTS.glob('summary_*.json'))
    merged = {}
    for path in files:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        for case, ent in data.items():
            cur = merged.get(case)
            if cur is None:
                merged[case] = ent
                continue
            # 取并集：单核基线取已有值，runs 按 key 合并
            if 'singlecore' not in cur and 'singlecore' in ent:
                cur['singlecore'] = ent['singlecore']
            runs = cur.setdefault('runs', {})
            for key, rec in ent.get('runs', {}).items():
                runs.setdefault(key, rec)
    return merged


def _load_baselines():
    """单核基线与用例元信息：不随多核算法优化变化。"""
    complete = RESULTS / 'summary_complete.json'
    if complete.exists():
        data = json.loads(complete.read_text(encoding='utf-8'))
        return {case: {'meta': ent.get('meta', {}),
                       'singlecore': ent.get('singlecore', {})}
                for case, ent in data.items()}
    baselines = {}
    for path in [RESULTS / 'summary.json'] + sorted(
            RESULTS.glob('summary_*.json')):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        for case, ent in data.items():
            cur = baselines.setdefault(case, {'meta': {}, 'singlecore': {}})
            if not cur['singlecore'] and ent.get('singlecore'):
                cur['singlecore'] = ent['singlecore']
            if not cur['meta'] and ent.get('meta'):
                cur['meta'] = ent['meta']
    return baselines


def load():
    """多核结果从榜单分片合成（逐格取最小 Makespan），与 LEADERBOARD 同源。"""
    shard_dir = RESULTS / 'leaderboard'
    shards = sorted(shard_dir.glob('*.json')) if shard_dir.exists() else []
    if not shards:
        return _load_legacy_summary()
    baselines = _load_baselines()
    board = {}
    for shard in shards:
        data = json.loads(shard.read_text(encoding='utf-8'))
        for key, rec in data.get('cells', {}).items():
            makespan = rec.get('makespan')
            if not makespan:
                continue
            case, problem, cores = key.split('|')     # case_001 | p1 | k2
            run_key = f'{problem}_{cores}'
            entry = board.setdefault(case, {
                'meta': baselines.get(case, {}).get('meta', {}),
                'singlecore': baselines.get(case, {}).get('singlecore', {}),
                'runs': {},
            })
            best = entry['runs'].get(run_key, {}).get('best')
            if best and best['makespan'] <= makespan:
                continue
            entry['runs'][run_key] = {'best': {
                'label': rec.get('label', ''),
                'makespan': makespan,
                'added_copy_bytes': rec.get('added_copy_bytes', 0),
                'cache_hit_rate': rec.get('cache_hit_rate', ''),
            }}
    return board


def collect(summary):
    rows = []
    for case in sorted(summary):
        ent = summary[case]
        sc = ent.get('singlecore', {})
        sc_ms = sc.get('makespan')
        if not sc_ms:
            continue
        for p in (1, 2, 3):
            for k in (2, 3, 4, 5):
                best = ent.get('runs', {}).get(f'p{p}_k{k}', {}).get('best')
                if not best:
                    continue
                rows.append({
                    'case': case, 'problem': p, 'cores': k,
                    'singlecore_makespan': sc_ms,
                    'makespan': best['makespan'],
                    'speedup': round(sc_ms / best['makespan'], 4),
                    'added_copy_bytes': best.get('added_copy_bytes', 0),
                    'label': best.get('label', ''),
                    'cache_hit_rate': best.get('cache_hit_rate', ''),
                    'parallelism': ent.get('meta', {}).get('parallelism', ''),
                    'n_elig': ent.get('meta', {}).get('n_elig', ''),
                })
    return rows


def write_csv(rows):
    path = REPORT / 'all_results.csv'
    fields = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def avg_speedups(rows):
    """{(problem, k): mean speedup}。"""
    agg = {}
    for r in rows:
        agg.setdefault((r['problem'], r['cores']), []).append(r['speedup'])
    return {key: st.mean(v) for key, v in agg.items()}


def plot_speedup_curves(rows):
    agg = avg_speedups(rows)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = {1: '#1f77b4', 2: '#d62728', 3: '#2ca02c'}
    for p in (1, 2, 3):
        xs = [1] + [k for k in (2, 3, 4, 5) if (p, k) in agg]
        ys = [1.0] + [agg[(p, k)] for k in xs[1:]]
        ax.plot(xs, ys, marker='o', color=colors[p],
                label=f'Problem {p}')
    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Average speedup (vs single-core)')
    ax.set_xticks(CORES)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = REPORT / 'speedup_curves.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_p2_p3_compare(rows):
    """问题3要求：无 L2（=问题2配置）与只读 Cache 的对比。"""
    agg = avg_speedups(rows)
    by = {(r['case'], r['problem'], r['cores']): r['makespan'] for r in rows}
    rel = {}   # k -> mean(P2 makespan / P3 makespan)
    for k in (2, 3, 4, 5):
        vals = []
        for (case, p, kk), ms in by.items():
            if p == 2 and kk == k and (case, 3, k) in by:
                vals.append(ms / by[(case, 3, k)])
        if vals:
            rel[k] = st.mean(vals)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for p, name, color in ((2, 'Problem 2 (no L2)', '#d62728'),
                           (3, 'Problem 3 (read-only L2)', '#2ca02c')):
        xs = [1] + [k for k in (2, 3, 4, 5) if (p, k) in agg]
        ys = [1.0] + [agg[(p, k)] for k in xs[1:]]
        ax.plot(xs, ys, marker='o', color=color, label=name)
    for k, v in rel.items():
        # 最右侧标注向左放置，防止导出或论文缩放时贴边裁切。
        dx = -34 if k == max(rel) else 4
        ax.annotate(f'x{v:.3f}', (k, agg[(3, k)]),
                    textcoords='offset points', xytext=(dx, -14),
                    fontsize=8, color='#2ca02c')
    ax.set_xlabel('Number of cores')
    ax.set_ylabel('Average speedup (vs single-core)')
    ax.set_xticks(CORES)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = REPORT / 'p2_vs_p3.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, rel


def write_md(rows, rel):
    agg = avg_speedups(rows)
    lines = ['# A题实验结果汇总', '']
    lines.append('## 平均加速比（相对单核基线，100 用例）')
    lines.append('')
    lines.append('| 问题 | 2核 | 3核 | 4核 | 5核 |')
    lines.append('|---|---|---|---|---|')
    for p in (1, 2, 3):
        cells = [f'{agg[(p, k)]:.3f}' if (p, k) in agg else '-'
                 for k in (2, 3, 4, 5)]
        lines.append(f'| 问题{p} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('## 问题3 只读 Cache 相对 无L2 的加速比')
    lines.append('')
    lines.append('| 核数 | 2 | 3 | 4 | 5 |')
    lines.append('|---|---|---|---|---|')
    cells = [f'{rel.get(k, float("nan")):.4f}' if k in rel else '-'
             for k in (2, 3, 4, 5)]
    lines.append('| Cache/无L2 | ' + ' | '.join(cells) + ' |')
    lines.append('')
    # 加速比分布
    lines.append('## 加速比分布（4 核）')
    lines.append('')
    for p in (1, 2, 3):
        vals = sorted(r['speedup'] for r in rows
                      if r['problem'] == p and r['cores'] == 4)
        if vals:
            lines.append(
                f'- 问题{p}: min={vals[0]:.2f} 中位={st.median(vals):.2f} '
                f'max={vals[-1]:.2f}；≥1.0 的用例数 '
                f'{sum(v >= 0.999 for v in vals)}/{len(vals)}')
    lines.append('')
    out = REPORT / 'summary_report.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    return out


def main():
    summary = load()
    rows = collect(summary)
    if not rows:
        print('no rows yet')
        return
    csv_path = write_csv(rows)
    png1 = plot_speedup_curves(rows)
    png2, rel = plot_p2_p3_compare(rows)
    md = write_md(rows, rel)
    agg = avg_speedups(rows)
    print(f'rows={len(rows)} csv={csv_path.name} png={png1.name},{png2.name} md={md.name}')
    for p in (1, 2, 3):
        print(f'P{p} avg speedup: '
              + ' '.join(f'k{k}={agg.get((p, k), 0):.3f}' for k in (2, 3, 4, 5)))


if __name__ == '__main__':
    main()
