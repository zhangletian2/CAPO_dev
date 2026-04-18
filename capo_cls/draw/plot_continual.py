"""
Combine results from all 6 continual TTA ablation settings and generate:
  1. A combined streaming accuracy curve (Figure 1)
  2. A summary table printed to stdout
  3. A LaTeX table file for the paper

Usage:
    python plot_continual.py [result_dir]
    # default result_dir = logs/continual_ablation
"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


SETTINGS = [
    ('S1_RQ_RT_RS',         'S1: Reset-Q / Reset-S'),
    ('S2_KQ_RT_RS',         'S2: Keep-Q  / Reset-S'),
    ('S3a_RQ_RT_KSp_RSo',   'S3a: Reset-Q / Keep-S-prompt,Reset-S-opt'),
    ('S3b_RQ_RT_KSp_KSo',   'S3b: Reset-Q / Keep-S-prompt,Keep-S-opt'),
    ('S4a_KQ_RT_KSp_RSo',   'S4a: Keep-Q  / Keep-S-prompt,Reset-S-opt'),
    ('S4b_KQ_RT_KSp_KSo',   'S4b: Keep-Q  / Keep-S-prompt,Keep-S-opt'),
]

LEGEND_6 = {
    'S1: Reset-Q / Reset-S':                       'RQ / RS',
    'S2: Keep-Q  / Reset-S':                       'KQ / RS',
    'S3a: Reset-Q / Keep-S-prompt,Reset-S-opt':    'RQ / KS / RO',
    'S3b: Reset-Q / Keep-S-prompt,Keep-S-opt':     'RQ / KS / KO',
    'S4a: Keep-Q  / Keep-S-prompt,Reset-S-opt':    'KQ / KS / RO',
    'S4b: Keep-Q  / Keep-S-prompt,Keep-S-opt':     'KQ / KS / KO',
}

LEGEND_4 = {
    'S1: Reset-Q / Reset-S':                       'RQ / RS',
    'S2: Keep-Q  / Reset-S':                       'KQ / RS',
    'S3a: Reset-Q / Keep-S-prompt,Reset-S-opt':    'RQ / KS',
    'S4a: Keep-Q  / Keep-S-prompt,Reset-S-opt':    'KQ / KS',
}

SUBSET_4_KEYS = set(LEGEND_4.keys())


def recompute_metrics(records, stage_boundaries, window=100):
    """Recompute metrics from raw records with unique keys per transition."""
    N = len(records)
    corrects = [r['correct'] for r in records]

    avg_acc = float(np.mean(corrects) * 100)

    stage_accs = {}
    for sb_idx, (start, domain, stage_i) in enumerate(stage_boundaries):
        end = stage_boundaries[sb_idx + 1][0] if sb_idx + 1 < len(stage_boundaries) else N
        seg = corrects[start:end]
        stage_accs[f'Stage{stage_i+1}({domain})'] = float(np.mean(seg) * 100) if seg else 0.0

    switch_accs = {}
    transition_count = {}
    for sb_idx in range(1, len(stage_boundaries)):
        start = stage_boundaries[sb_idx][0]
        prev_domain = stage_boundaries[sb_idx - 1][1]
        domain = stage_boundaries[sb_idx][1]
        end = min(start + window, N)
        seg = corrects[start:end]
        pair = f'{prev_domain}->{domain}'
        cnt = transition_count.get(pair, 0) + 1
        transition_count[pair] = cnt
        switch_accs[f'Acc@Switch_{pair}_{cnt}'] = float(np.mean(seg) * 100) if seg else 0.0

    last_start = stage_boundaries[-1][0]
    revisit_seg = corrects[last_start:min(last_start + window, N)]
    revisit_acc = float(np.mean(revisit_seg) * 100) if revisit_seg else 0.0

    first_end = stage_boundaries[1][0] if len(stage_boundaries) > 1 else N
    first_tail_start = max(first_end - window, stage_boundaries[0][0])
    first_tail_acc = float(np.mean(corrects[first_tail_start:first_end]) * 100)
    forgetting_gap = float(first_tail_acc - revisit_acc)

    uplink_ratios = {}
    transition_count2 = {}
    for sb_idx in range(1, len(stage_boundaries)):
        start = stage_boundaries[sb_idx][0]
        prev_domain = stage_boundaries[sb_idx - 1][1]
        domain = stage_boundaries[sb_idx][1]
        end = min(start + window, N)
        seg_records = records[start:end]
        ratio = float(np.mean([r['is_uplink'] for r in seg_records]) * 100) if seg_records else 0.0
        pair = f'{prev_domain}->{domain}'
        cnt = transition_count2.get(pair, 0) + 1
        transition_count2[pair] = cnt
        uplink_ratios[f'Uplink%@{pair}_{cnt}'] = ratio

    return {
        'AvgAcc': avg_acc,
        'StageAcc': stage_accs,
        'SwitchAcc': switch_accs,
        'RevisitAcc@100': revisit_acc,
        'ForgettingGap': forgetting_gap,
        'BoundaryUplinkRatio': uplink_ratios,
    }


def load_results(result_dir):
    results = {}
    for tag, label in SETTINGS:
        path = os.path.join(result_dir, f'records_{tag}.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            data['metrics'] = recompute_metrics(data['records'], data['boundaries'])
            results[label] = data
            print(f"  Loaded {tag} (metrics recomputed)")
        else:
            print(f"  Missing {tag} — skipped")
    return results


def plot_combined(results, save_path, legend_map, subset_keys=None,
                  window=50, stream_label=None):
    fig, ax = plt.subplots(figsize=(14, 6))

    colors_6 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    styles_6 = ['-', '--', '-.', ':', '-', '--']
    colors_4 = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    styles_4 = ['-', '--', '-.', ':']

    items = [(l, d) for l, d in results.items()
             if subset_keys is None or l in subset_keys]
    use4 = len(items) <= 4
    colors = colors_4 if use4 else colors_6
    linestyles = styles_4 if use4 else styles_6

    first_boundaries = None
    N = 0

    for idx, (label, data) in enumerate(items):
        records = data['records']
        boundaries = data['boundaries']
        corrects = [r['correct'] for r in records]
        N = max(N, len(corrects))

        smooth = []
        for i in range(len(corrects)):
            s = max(0, i - window + 1)
            smooth.append(np.mean(corrects[s:i+1]) * 100)

        avg_acc = data['metrics']['AvgAcc']
        short_label = legend_map.get(label, label)
        ax.plot(range(len(corrects)), smooth,
                label=f'{short_label} (Avg={avg_acc:.1f}%)',
                color=colors[idx % len(colors)],
                linestyle=linestyles[idx % len(linestyles)],
                linewidth=1.8, alpha=0.85)

        if first_boundaries is None:
            first_boundaries = boundaries

    ax.set_ylim(0, 100)

    if first_boundaries:
        for sb_idx, (start, domain, stage_i) in enumerate(first_boundaries):
            if sb_idx > 0:
                prev_domain = first_boundaries[sb_idx - 1][1]
                ax.axvline(x=start, color='gray', linestyle='--',
                           linewidth=2, alpha=0.5)
                ax.text(start + N * 0.005, 96,
                        f'{prev_domain}\u2192{domain}', fontsize=13,
                        color='dimgray', fontweight='bold', va='top')

        for sb_idx, (start, domain, stage_i) in enumerate(first_boundaries):
            if sb_idx + 1 < len(first_boundaries):
                end = first_boundaries[sb_idx + 1][0]
            else:
                end = N
            mid = (start + end) // 2
            ax.text(mid, 3, domain, fontsize=16, ha='center',
                    fontweight='bold', color='dimgray', alpha=0.5)

    ax.set_xlabel('Sample Index', fontsize=14)
    ax.set_ylabel(f'Top-1 Accuracy (%)', fontsize=14)
    ax.set_title(f'Continual stream evaluation', fontsize=16)
    ax.legend(fontsize=13, loc='lower right')
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, N)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")


def print_table(results):
    first_data = list(results.values())[0]
    switch_keys = sorted(first_data['metrics']['SwitchAcc'].keys())
    switch_headers = [k.replace('Acc@Switch_', '') for k in switch_keys]

    col_w = max(12, max((len(h) + 2 for h in switch_headers), default=12))
    line_w = 28 + 9 + col_w * len(switch_headers) + 11 + 9

    print(f"\n{'=' * line_w}")
    header = f"{'Setting':<28} {'AvgAcc':>8}"
    for h in switch_headers:
        header += f" {'@' + h:>{col_w}}"
    header += f" {'Revisit':>10} {'F-Gap':>8}"
    print(header)
    print(f"{'-' * line_w}")

    for label, data in results.items():
        m = data['metrics']
        line = f"{label:<28} {m['AvgAcc']:>8.2f}"
        for sk in switch_keys:
            v = m['SwitchAcc'].get(sk, float('nan'))
            line += f" {v:>{col_w}.2f}"
        line += f" {m['RevisitAcc@100']:>10.2f} {m['ForgettingGap']:>+8.2f}"
        print(line)

        for sk, sv in m['StageAcc'].items():
            print(f"    {sk}: {sv:.2f}%")

    print(f"{'=' * line_w}")

    has_uplink = any(data['metrics'].get('BoundaryUplinkRatio') for data in results.values())
    if has_uplink:
        print("\nBoundary Uplink Ratios (first 100 samples after switch):")
        for label, data in results.items():
            for k, v in data['metrics'].get('BoundaryUplinkRatio', {}).items():
                print(f"  {label}: {k} = {v:.1f}%")


def save_latex_table(results, save_path):
    first_data = list(results.values())[0]
    switch_keys = sorted(first_data['metrics']['SwitchAcc'].keys())
    n_switches = len(switch_keys)

    switch_col_headers = []
    for k in switch_keys:
        short_k = k.replace('Acc@Switch_', '').replace('->', r'{\to}')
        switch_col_headers.append(f"Sw$_{{{short_k}}}$")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Continual TTA ablation: effect of resetting vs.\ keeping "
                 r"the entropy queue~(Q), student prompt and student optimizer "
                 r"across domain shifts.  Teacher always resets.}")
    lines.append(r"\label{tab:continual_ablation}")
    lines.append(r"\small")
    col_spec = "l ccc " + "c" * (n_switches + 2)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    switch_header_str = " & ".join(switch_col_headers)
    lines.append(f"Setting & Queue & S-Prompt & S-Opt & AvgAcc & "
                 f"{switch_header_str} & F-Gap \\\\")
    lines.append(r"\midrule")

    mode_map = {
        'S1':  ('Reset', 'Reset', 'Reset'),
        'S2':  ('Keep',  'Reset', 'Reset'),
        'S3a': ('Reset', 'Keep',  'Reset'),
        'S3b': ('Reset', 'Keep',  'Keep'),
        'S4a': ('Keep',  'Keep',  'Reset'),
        'S4b': ('Keep',  'Keep',  'Keep'),
    }

    for label, data in results.items():
        m = data['metrics']
        short = label.split(':')[0].strip()
        q_mode, sp_mode, so_mode = mode_map.get(short, ('?', '?', '?'))

        switch_vals_str = " & ".join(
            f"{m['SwitchAcc'].get(k, 0):.1f}" for k in switch_keys
        )
        line = (f"{short} & {q_mode} & {sp_mode} & {so_mode} & "
                f"{m['AvgAcc']:.1f} & {switch_vals_str} & "
                f"{m['ForgettingGap']:+.1f} \\\\")
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"LaTeX table saved to {save_path}")


if __name__ == '__main__':
    result_dir = sys.argv[1] if len(sys.argv) > 1 else 'logs/continual_ablation'

    print(f"Loading results from {result_dir}/")
    results = load_results(result_dir)

    if not results:
        print(f"No results found in {result_dir}")
        sys.exit(1)

    print_table(results)

    stages = list(results.values())[0]['boundaries']
    domains = [d for _, d, _ in stages]
    stream_label = r'$' + r' \!\to\! '.join(domains) + r'$'

    plot_path_6 = os.path.join(result_dir, 'combined_stream_6s.png')
    plot_combined(results, plot_path_6, legend_map=LEGEND_6,
                  stream_label=stream_label)

    plot_path_4 = os.path.join(result_dir, 'combined_stream_4s.png')
    plot_combined(results, plot_path_4, legend_map=LEGEND_4,
                  subset_keys=SUBSET_4_KEYS, stream_label=stream_label)

    latex_path = os.path.join(result_dir, 'table.tex')
    save_latex_table(results, latex_path)
