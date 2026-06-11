#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

COLORS = {
    'hg':   'tab:blue',
    'hgs':  'tab:purple',
    'hgc':  'tab:cyan',
    'hgcs': 'tab:red',
    'g':    'tab:gray',
}
LABELS = {
    'hg':   'naive (no scatter)',
    'hgs':  'naive w/ scatter',
    'hgc':  'cuda (no scatter)',
    'hgcs': 'cuda w/ scatter',
    'g':    'graph',
}

LOSSLOG_DIR = 'losslogs'


def read_log(path):
    try:
        d = np.loadtxt(path, usecols=(0, 1))
        if d.ndim == 1:
            d = d[None]
        return d[:, 0], d[:, 1]
    except Exception:
        return None, None


def ema(y, alpha=0.02):
    s = np.empty_like(y, dtype=float)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = alpha * y[i] + (1 - alpha) * s[i - 1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-name', required=True)
    ap.add_argument('--task', type=int, required=True)
    ap.add_argument('--impls', default='hgs,hgcs',
                    help='Comma-separated impl tags to plot')
    ap.add_argument('--repl', default='1',
                    help='Comma-separated replicate indices')
    ap.add_argument('--ema', type=float, default=0.02,
                    help='EMA smoothing factor (0=no smoothing, 1=no memory)')
    ap.add_argument('--no-logy', action='store_true',
                    help='Linear y-axis instead of log')
    ap.add_argument('--interval', type=float, default=3.0,
                    help='Refresh interval in seconds')
    args = ap.parse_args()

    impls = args.impls.split(',')
    repls = [int(r) for r in args.repl.split(',')]
    use_logy = not args.no_logy

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(f'Task {args.task}  —  {args.log_name}')
    if use_logy:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    mean_lines = {}
    for impl in impls:
        line, = ax.plot([], [],
                        color=COLORS.get(impl),
                        label=LABELS.get(impl, impl),
                        linewidth=2.0)
        mean_lines[impl] = line
    ax.legend()

    def update(_frame):
        for coll in ax.collections[:]:
            coll.remove()

        any_data = False
        for impl in impls:
            series = []
            for r in repls:
                path = f'{LOSSLOG_DIR}/losslog_{impl}_t{args.task}_{args.log_name}_r{r}.txt'
                xs, ys = read_log(path)
                if xs is not None and len(xs) > 1:
                    series.append((xs, ema(ys, args.ema)))

            if not series:
                continue

            any_data = True
            min_len = min(len(s[0]) for s in series)
            xs = series[0][0][:min_len]
            ys_stack = np.array([s[1][:min_len] for s in series])

            if len(series) == 1:
                mean_lines[impl].set_data(xs, ys_stack[0])
            else:
                if use_logy:
                    log_ys = np.log(np.clip(ys_stack, 1e-12, None))
                    mean_log = np.mean(log_ys, axis=0)
                    std_log = np.std(log_ys, axis=0)
                    mean_y = np.exp(mean_log)
                    lo = np.exp(mean_log - std_log)
                    hi = np.exp(mean_log + std_log)
                else:
                    mean_y = np.mean(ys_stack, axis=0)
                    std_y = np.std(ys_stack, axis=0)
                    lo = mean_y - std_y
                    hi = mean_y + std_y

                mean_lines[impl].set_data(xs, mean_y)
                ax.fill_between(xs, lo, hi, alpha=0.2, color=COLORS.get(impl, 'black'))

        if any_data:
            ax.relim()
            ax.autoscale_view()
        return list(mean_lines.values())

    _ani = animation.FuncAnimation(
        fig, update,
        interval=args.interval * 1000,
        blit=False,
        cache_frame_data=False,
    )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
