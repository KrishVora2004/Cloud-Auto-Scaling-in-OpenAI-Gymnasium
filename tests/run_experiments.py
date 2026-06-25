# --save flag is optional, for when you want a saved artifact to inspect
# later or re-plot without re-running agents -- it is no longer required
# for visualization to work.

# Usage:
#   python -m tests.run_experiments
#   python -m tests.run_experiments --save
#   python -m tests.run_experiments --scenarios spike noisy --seeds 1 2 3

import argparse
import json

from tests.compare import run_comparison
from tests.scenarios import SCENARIOS
from tests.plot_dashboard import plot_all_metrics
from evaluation.load_models import ModelLoadError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=SCENARIOS,
                         help="Scenarios to run (default: all of tests.scenarios.SCENARIOS)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 42, 100],
                         help="Seeds to average over per scenario")
    parser.add_argument("--steps", type=int, default=500,
                         help="Episode length (workload steps) per run")
    parser.add_argument("--save", action="store_true",
                         help="Also write results to results.json (optional artifact)")

    args = parser.parse_args()

    print("\nRunning experiments...\n")

    try:
        results = run_comparison(
            scenarios=args.scenarios,
            seeds=tuple(args.seeds),
            steps=args.steps,
        )
    except ModelLoadError:
        # load_models.py already printed "Failed to load models" -- nothing further to add, just stop the pipeline cleanly
        return

    print("\nExperiments complete. Plotting dashboard...\n")

    
    plot_all_metrics(results)

    if args.save:
        with open("results.json", "w") as f:
            json.dump(results, f, indent=4)
        print("\nResults also saved to results.json\n")


if __name__ == "__main__":
    main()