"""
run_all.py  -  run the jump_test pipeline and print the runtime of every step
=============================================================================

    python run_all.py                # tests -> calibration (all days) -> analysis -> figures   (about 20 min on 2 cores)
    python run_all.py --jobs 2       # calibrate the two stocks in parallel
    python run_all.py --quick-tests  # smaller Monte Carlo in the tests
    python run_all.py --skip-tests   # go straight to the calibration
    python run_all.py --overnight    # also overnight.py (about 1 h on 5 worker processes, see README)

Run seb_merton_analys.py in the folder above first: the tests compare against its results.
"""

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick-tests", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--overnight", action="store_true")
    ap.add_argument("--force", action="store_true", help="continue even if a step fails")
    a = ap.parse_args()

    steps = [] if a.skip_tests else [("tests.py", ["--quick"] if a.quick_tests else [])]
    steps += [("calibrate.py", ["--jobs", str(a.jobs)]), ("analyze.py", [])]
    if a.overnight:
        steps.append(("overnight.py", ["--jobs", str(a.jobs)] if a.jobs > 1 else []))
    steps += [("plot_skew_powerlaw.py", []), ("make_vol_surface_3d.py", [])]

    timings, t_all = [], time.time()
    for script, args in steps:
        print(f"\n{'=' * 100}\n>>> {script} {' '.join(args)}\n{'=' * 100}", flush=True)
        t0 = time.time()
        rc = subprocess.call([sys.executable, os.path.join(HERE, script)] + args, cwd=HERE)
        timings.append((f"{script} {' '.join(args)}", rc, time.time() - t0))
        if rc != 0 and not a.force:
            print(f"\n{script} failed (exit code {rc}). Stopping; use --force to continue anyway.")
            break

    print(f"\n{'=' * 100}\nRUNTIMES\n{'=' * 100}")
    for name, rc, dt in timings:
        print(f"  {name:50s} {dt:8.0f} s   {'ok' if rc == 0 else 'FAILED'}")
    print(f"  {'TOTAL':50s} {time.time() - t_all:8.0f} s")
    sys.exit(max((rc for _, rc, _ in timings), default=0))


if __name__ == "__main__":
    main()
