import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


os.chdir(Path(__file__).resolve().parent)
results = Path(os.environ.get('RESULTS', 'results'))
url = os.environ.get('URL', 'http://127.0.0.1:8000')
configs = ['baseline', 'chunk1024', 'chunk4096', 'seq32']
dataset_size = 640
dataset_sha256 = '67c48379e5ae2608f7ca08650fd346facdc01f766a3e8eecea6459d7cec00ebc'


def read_run(folder):
    profile = folder / 'profile'
    if not (profile / '.aiperf_results_ready.json').exists():
        return
    run = json.loads((profile / 'profile_export_aiperf.json').read_text())
    if run.get('is_complete') and not run.get('was_cancelled'):
        return run


def measure(folder, rpm=120, load=False):
    folder.parent.mkdir(parents=True, exist_ok=True)
    run = read_run(folder)
    if run is None:
        if folder.exists():
            folder.rename(folder.with_name(folder.name + f'.incomplete-{int(time.time())}'))
        cmd = ['bash', 'load_test.sh', str(rpm), str(folder)] if load else [
            'bash', 'benchmark.sh', str(folder), str(rpm), str(dataset_size)
        ]
        print('measure', folder, rpm, flush=True)
        send_seconds = 300 if load else math.ceil(dataset_size * 60 / rpm)
        with folder.with_suffix('.log').open('w') as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(send_seconds + 240)
            finally:
                stop(proc)
        run = read_run(folder)
        if code or run is None:
            raise RuntimeError(f'{folder}: benchmark incomplete (exit {code})')
    latency = run.get('request_latency') or {}
    ok = int(run.get('request_count', {}).get('avg', 0))
    errors = int(run.get('error_request_count', {}).get('avg', 0))
    print(folder, f'{ok} ok / {errors} errors', 'mean ms', latency.get('avg'), 'p90 ms', latency.get('p90'), flush=True)
    return run


def start_server(config, tune):
    log = results / 'servers' / f'{config}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        f.write(f'\n{time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime())} bash configs.sh {config} {tune}\n')
        f.flush()
        return subprocess.Popen(['bash', 'configs.sh', config, tune], stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True)


def wait_server(proc):
    end = time.monotonic() + 540
    while proc.poll() is None and time.monotonic() < end:
        try:
            with urlopen(url + '/health', timeout=2) as response:
                if response.status == 200:
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(3)
    raise RuntimeError(f'vLLM startup failed, exit={proc.poll()}')


def stop(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def tune_run(config, tune):
    path = results / 'tune' / config
    if read_run(path) is not None:
        return measure(path)
    proc = start_server(config, tune)
    try:
        wait_server(proc)
        return measure(path)
    finally:
        stop(proc)


def mean_latency(run):
    if run.get('request_count', {}).get('avg') != dataset_size or run.get('error_summary'):
        return float('inf')
    return run.get('request_latency', {}).get('avg', float('inf'))


def passes(run, rpm):
    ok = run.get('request_count', {}).get('avg', 0)
    p90 = (run.get('request_latency') or {}).get('p90', float('inf'))
    return p90 < 15000 and ok >= rpm * 5 * .99 and not run.get('error_summary')


def load_limit():
    lo, hi = 0, None
    for rpm in [120, 240, 420, 660, 960, 1440, 1920, 2400]:
        run = measure(results / 'load' / str(rpm), rpm, load=True)
        if not passes(run, rpm):
            hi = rpm
            break
        lo = rpm

    while hi is not None and hi - lo > 30:
        rpm = ((lo + hi) // 60) * 30
        if rpm <= lo:
            rpm = lo + 30
        run = measure(results / 'load' / str(rpm), rpm, load=True)
        if passes(run, rpm):
            lo = rpm
        else:
            hi = rpm
    return lo, hi


def main():
    if hashlib.sha256(Path('data/requests.jsonl').read_bytes()).hexdigest() != dataset_sha256:
        raise ValueError('data/requests.jsonl does not match the assignment archive')
    results.mkdir(exist_ok=True)
    scores = {c: mean_latency(tune_run(c, c)) for c in configs}
    tune = min(scores, key=scores.get)
    for c in ['v1', 'ngram', 'ngram_gpu']:
        try:
            scores[c] = mean_latency(tune_run(c, tune))
        except RuntimeError as exc:
            print(c, exc, flush=True)
            scores[c] = float('inf')
    spec = min(['ngram', 'ngram_gpu'], key=scores.get)
    if scores['v1'] == float('inf') or scores[spec] == float('inf'):
        raise RuntimeError('Speculative decoding pair did not finish')
    best = min(scores, key=scores.get)
    best_tune = best if best in configs else tune
    cases = {'tune': tune, 'spec': spec, 'best': best, 'best_tune': best_tune}
    (results / 'selection.json').write_text(json.dumps(cases, indent=2) + '\n')
    print('selected', cases, flush=True)

    rates = [60, 240, 360, 480, 720]
    for config in ['v1', spec]:
        proc = start_server(config, tune)
        try:
            wait_server(proc)
            for rpm in rates:
                measure(results / 'spec' / config / str(rpm), rpm)
            if best == config:
                lo, hi = load_limit()
        finally:
            stop(proc)

    if best not in ['v1', spec]:
        proc = start_server(best, best_tune)
        try:
            wait_server(proc)
            lo, hi = load_limit()
        finally:
            stop(proc)
    cases.update(max_rpm=lo, first_fail_rpm=hi, spec_rpm=[60, 120] + rates[1:])
    (results / 'selection.json').write_text(json.dumps(cases, indent=2) + '\n')
    print('Done', cases, flush=True)


if __name__ == '__main__':
    main()
