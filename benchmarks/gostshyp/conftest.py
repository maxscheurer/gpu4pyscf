"""conftest for GOSTSHYP Amylose benchmarks."""
import os

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print and save summary table after all tests complete."""
    try:
        import pandas as pd
        from test_amylose_benchmark import _results
    except ImportError:
        return

    if not _results:
        return

    df = pd.DataFrame(_results)
    df = df.sort_values('natom')

    # Save raw CSV
    csv_path = os.path.join(BENCH_DIR, 'amylose_benchmark_results.csv')
    df.to_csv(csv_path, index=False)

    # Print formatted table
    display = df.copy()
    fmt = {
        'e_tot':          '{:.6f}',
        'e_gostshyp':     '{:.6f}',
        'min_F':          '{:.2e}',
        'min_A':          '{:.2e}',
        'total_A':        '{:.1f}',
        'max_grad':       '{:.6f}',
        'rms_grad':       '{:.6f}',
        'max_grad_gost':  '{:.6f}',
        'overlap_cutoff': '{:.1e}',
    }
    for col, f in fmt.items():
        if col in display.columns:
            display[col] = display[col].apply(f.format)

    terminalreporter.write_line('')
    terminalreporter.write_sep('=', 'GOSTSHYP Amylose Benchmark Summary')
    terminalreporter.write_line('PBE/def2-SV(P), 50 GPa, vdW/OCC')
    terminalreporter.write_line('')
    with pd.option_context('display.max_columns', None,
                           'display.width', 200,
                           'display.max_colwidth', 20):
        terminalreporter.write_line(display.to_string(index=False))
    terminalreporter.write_line('')
    terminalreporter.write_line(f'Results saved to {csv_path}')
    terminalreporter.write_sep('=', '')
