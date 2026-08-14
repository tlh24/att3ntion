import sys
from collections import defaultdict


def pytest_terminal_summary(terminalreporter):
    """Bucket kernel-correctness failures by shape. A kernel bug clusters on one
    axis -- every D=64 config, everything past N=256 -- and the flat per-test
    failure list hides that.

    The config table is read out of the already-imported test module rather than
    imported here, which would trip its module-level CUDA skip at collection.
    """
    if not terminalreporter.stats.get('failed'):
        return
    configs = {}
    for name, module in list(sys.modules.items()):
        # test modules only -- a bare getattr sweep would hit torch.classes,
        # whose __getattr__ tries to instantiate the name as a TorchScript class
        if name.rpartition('.')[2].startswith('test_'):
            configs.update(getattr(module, 'CONFIGS_BY_NAME', None) or {})
    if not configs:
        return

    axes = {'N': lambda c: c.N, 'D': lambda c: c.D, 'B*H': lambda c: c.B * c.H}
    tally = {axis: defaultdict(lambda: [0, 0]) for axis in axes}
    for outcome in ('passed', 'failed'):
        for report in terminalreporter.stats.get(outcome, []):
            if report.when != 'call':
                continue
            ids = report.nodeid.partition('[')[2].rstrip(']').split('-')
            config = next((configs[i] for i in ids if i in configs), None)
            if config is None:
                continue
            for axis, key in axes.items():
                counts = tally[axis][key(config)]
                counts[0] += outcome == 'passed'
                counts[1] += 1

    terminalreporter.write_sep('=', 'FAILURES BY SHAPE')
    for axis in axes:
        line = '  '.join(f"{key}: {passed}/{total}"
                         for key, (passed, total) in sorted(tally[axis].items())
                         if passed < total)
        if line:
            terminalreporter.write_line(f"  {axis:>3} {line}")
