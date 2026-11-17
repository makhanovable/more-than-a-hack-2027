import argparse
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from possession import build

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1 of 4: download the open external data (2.8 GB) and check every file.")
    args = parser.parse_args()
    sys.exit(build.step_download())
