import argparse
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession import build

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 4 of 4: label the competition match -> output/submission.csv "
                    "(about 2 min; numpy only).")
    args = parser.parse_args()
    build.step_predict()
