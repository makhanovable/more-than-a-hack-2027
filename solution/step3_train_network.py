import argparse
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession import build

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 3 of 4: train the sequence networks on the DFL matches and choose the decoder "
                    "settings on the Metrica matches (about 20 min; needs PyTorch).")
    parser.add_argument("--jobs", type=int, default=5, help="processes to use (speed only; the result does not depend on it)")
    args = parser.parse_args()
    build.step_network(args.jobs)
