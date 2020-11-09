import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--results_dir')

args = parser.parse_args()

results_dir = args.results_dir

os.makedirs(results_dir, exist_ok=True)
