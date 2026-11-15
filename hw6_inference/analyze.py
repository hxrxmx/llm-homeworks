import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent / 'results'


def read_run(folder):
    profile = folder / 'profile'
    data = json.loads((profile / 'profile_export_aiperf.json').read_text())
    phases = json.loads((profile / 'phase_manifest.json').read_text())['phases']
    phase = next(p for p in phases if p['phase_kind'] == 'profiling')
    config = next(p for p in data['input_config']['phases'] if p['kind'] == 'profiling')
    server = json.loads((profile / 'server_metrics_export.json').read_text())['metrics']
    if not data['is_complete'] or phase['was_cancelled']:
        raise ValueError(f'{folder}: incomplete benchmark')

    def avg(metric):
        return data.get(metric, {}).get('avg')

    def server_stat(metric, stat):
        series = server.get('vllm:' + metric, {}).get('series', [])
        return series[0]['stats'].get(stat) if series else None

    latency = data.get('request_latency', {})
    n = phase['successful_request_count']
    errors = phase['error_request_count']
    rpm = round(config['rate'] * 60)
    duration = config.get('duration')
    draft = server_stat('spec_decode_num_draft_tokens', 'total')
    accepted = server_stat('spec_decode_num_accepted_tokens', 'total')
    p90 = latency.get('p90')
    return {
        'rpm': rpm, 'ok': n, 'errors': errors,
        'mean_s': latency.get('avg', float('nan')) / 1000,
        'p90_s': p90 / 1000 if p90 is not None else float('nan'),
        'ttft_mean_ms': avg('time_to_first_token'), 'itl_mean_ms': avg('inter_token_latency'),
        'completion_tokens_mean': avg('usage_completion_tokens'),
        'request_throughput_rps': avg('request_throughput'),
        'output_throughput_tps': avg('output_token_throughput'),
        'send_duration_s': duration,
        'duration_with_drain_s': (phase['end_ns'] - phase['start_ns']) / 1e9,
        'queue_mean_s': server_stat('request_queue_time_seconds', 'avg'),
        'waiting_max': server_stat('num_requests_waiting', 'max'),
        'draft_tokens': draft, 'accepted_tokens': accepted,
        'acceptance_pct': 100 * accepted / draft if draft and accepted is not None else None,
        'sla': bool(p90 is not None and p90 < 15000 and not errors and n >= 5 * rpm * .99)
        if duration is not None or folder.parent.name == 'load' else None,
    }


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def analyze():
    chosen = json.loads((root / 'selection.json').read_text())
    tune = []
    for config in ['baseline', 'chunk1024', 'chunk4096', 'seq32', 'v1', 'ngram', 'ngram_gpu']:
        path = root / 'tune' / config
        if (path / 'profile' / 'profile_export_aiperf.json').exists():
            tune.append({'config': config, **read_run(path)})
    for row in tune:
        row['speedup'] = tune[0]['mean_s'] / row['mean_s']
    save_csv(root / 'tuning.csv', tune)

    spec = []
    for rpm in chosen['spec_rpm']:
        without, with_spec = [
            read_run(root / 'tune' / c if rpm == 120 else root / 'spec' / c / str(rpm))
            for c in ['v1', chosen['spec']]
        ]
        speedup = None
        if not (without['errors'] or with_spec['errors']):
            speedup = without['p90_s'] / with_spec['p90_s']
        spec.append({'rpm': rpm,
                     **{'off_' + k: v for k, v in without.items() if k != 'rpm'},
                     **{'on_' + k: v for k, v in with_spec.items() if k != 'rpm'},
                     'p90_speedup': speedup})
    save_csv(root / 'speculation.csv', spec)

    load = sorted([read_run(p) for p in (root / 'load').iterdir() if p.is_dir() and p.name.isdigit()],
                  key=lambda x: x['rpm'])
    save_csv(root / 'load.csv', load)

    fig, (ax, speed_ax) = plt.subplots(1, 2, figsize=(7.0, 2.55),
                                      gridspec_kw={'width_ratios': [3, 2]})
    for prefix, label, color in [('off', 'без спекдека', 'C0'), ('on', chosen['spec'], 'C1')]:
        ax.plot([r['rpm'] for r in spec], [r[f'{prefix}_p90_s'] for r in spec],
                color=color, label=label)
        for row in spec:
            face = 'white' if row[f'{prefix}_errors'] else color
            ax.plot(row['rpm'], row[f'{prefix}_p90_s'], 'o', color=color, markerfacecolor=face)
    ax.set(xlabel='нагрузка, RPM', ylabel='P90 E2E, с', xticks=chosen['spec_rpm'])
    ax.legend(loc='lower right', fontsize=8)
    complete = [r for r in spec if r['p90_speedup'] is not None]
    speed_ax.plot([r['rpm'] for r in complete], [r['p90_speedup'] for r in complete], 'o-')
    speed_ax.axhline(1, color='gray', linestyle='--', linewidth=1)
    speed_ax.set(xlabel='нагрузка, RPM', ylabel='ускорение по P90',
                 xticks=[r['rpm'] for r in complete])
    fig.tight_layout()
    fig.savefig(root / 'speculation.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    ax.plot([r['rpm'] for r in load], [r['p90_s'] for r in load], 'o-')
    ax.axhline(15, color='gray', linestyle='--', linewidth=1, label='граница SLA, 15 с')
    for rpm, color, label in [(chosen['max_rpm'], 'green', 'прошёл'),
                              (chosen['first_fail_rpm'], 'red', 'не прошёл')]:
        if rpm:
            row = next(r for r in load if r['rpm'] == rpm)
            ax.scatter(rpm, row['p90_s'], color=color, zorder=3, label=f'{rpm} RPM, {label}')
    ax.set(xlabel='нагрузка, RPM', ylabel='P90 E2E, с', xticks=[r['rpm'] for r in load])
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'load.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    analyze()
