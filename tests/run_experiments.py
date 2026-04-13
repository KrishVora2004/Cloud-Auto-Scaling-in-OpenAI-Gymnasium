import sys
import os
import json

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.compare import run_comparison


def main():
    print("\nRunning experiments...\n")

    results = run_comparison()

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to results.json\n")


if __name__ == "__main__":
    main()