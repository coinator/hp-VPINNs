import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--results_dir', default=None)

args = parser.parse_args()

results_dir = args.results_dir

if results_dir:
    os.makedirs(results_dir, exist_ok=True)
