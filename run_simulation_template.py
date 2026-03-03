"""
Template for running a simulation.
Example: run_simulation.py — fit a Gaussian copula to each (250, 50) window.

Usage:
    python run_calculation_1.py --data-path /path/to/data.npy
    python run_calculation_1.py --data-path /path/to/data.npy --chunk-size 50
"""

import argparse
import logging
import os
import tempfile
from functools import partial
from pathlib import Path
from tools.Helpers import save_scalars, save_objects, save_params
from tools.Runner import Runner
from simulations.GaussianCopula import fit_gaussian_copula


def parse_args():
    parser = argparse.ArgumentParser(description="Fit Gaussian copula to each window")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to input .npy file, shape (N, 250, 50)")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Windows per chunk (default: 100)")
    parser.add_argument("--object-stride", type=int, default=250,
                        help="Save fitted copula every N windows (default: 250)")
    parser.add_argument("--n-workers", type=int, default=os.cpu_count())
    parser.add_argument("--verbose", action="store_true",
                        help="Pass verbose=True to copula fitter")
    parser.add_argument("--temp-dir", type=str, default=None)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    # Build the callable — plain function or partial if args are needed.
    # Both are picklable since fit_gaussian_copula is a module-level function.
    if args.verbose:
        calc_fn = partial(fit_gaussian_copula, verbose=True)
    else:
        calc_fn = fit_gaussian_copula   # no partial needed, use directly

    # Instantiate the runner
    runner = Runner(
        calculation_fn=calc_fn,
        data_path=args.data_path,
        chunk_size=args.chunk_size,
        object_stride=args.object_stride,
        n_workers=args.n_workers,
        temp_dir="temp",
    )

    runner.run()

    # Collect the results
    scalars = runner.collect_scalars()
    print(f"Scalars collected: {len(scalars)} windows")
    print(f"Sample — window 0: log_lik={scalars[0][0]:.4f}  aic={scalars[0][1]:.4f}")

    objects = runner.collect_objects()
    print(f"Objects collected: {len(objects)} fitted copulas (every {args.object_stride} windows)")

    # Save results to simulation folder
    #save_scalars(scalars)
    #save_objects(objects)
    #save_params()

    # Clean temporary data
    runner.cleanup()


if __name__ == "__main__":
    main()