import argparse
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession import build

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2 of 4: measure the competition's tracker and train the three boosted models "
                    "on the DFL matches (about 15 min with 5 processes).")
    parser.add_argument("--jobs", type=int, default=5, help="processes to use (speed only; the result does not depend on it)")
    args = parser.parse_args()
    build.step_boosting(args.jobs)
