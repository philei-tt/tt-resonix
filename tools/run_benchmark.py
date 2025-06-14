#!/usr/bin/env python3
import argparse
import json5 as json
import json as json_std
import subprocess
import re
import statistics
import shutil
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark runner")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--executable", required=True, help="Path to executable")
    parser.add_argument(
        "--n", type=int, default=5, help="Number of runs per stencil size"
    )
    return parser.parse_args()


def extract_throughput(output):
    match = re.search(r"Throughput:\s*([\d.]+)\s*Mpts/s", output)
    if match:
        return float(match.group(1))
    return None


def run_simulation(executable, config_path):
    try:
        result = subprocess.run(
            [executable, config_path], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Execution failed:\n{e.stderr}", file=sys.stderr)
        raise


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        original_config = json.load(f)

    try:
        for m in [1, 2, 4, 8]:
            print(f"\nRunning benchmarks for m = {m}...")
            throughputs = []
            for i in range(args.n):
                # Update config
                config = original_config.copy()
                config["derivative"]["m"] = m
                with open(args.config, "w") as f:
                    json_std.dump(config, f, indent=2)

                # Run the simulation
                output = run_simulation(args.executable, args.config)
                throughput = extract_throughput(output)
                if throughput is None:
                    raise RuntimeError("Throughput not found in output")
                print(f"  Run {i+1}: {throughput:.2f} Mpts/s")
                throughputs.append(throughput)

            print(f"Results for m = {m}:")
            print(f"  Min:  {min(throughputs):.2f} Mpts/s")
            print(f"  Max:  {max(throughputs):.2f} Mpts/s")
            print(f"  Mean: {statistics.mean(throughputs):.2f} Mpts/s")

    except Exception as e:
        print(f"\nError occurred: {e}", file=sys.stderr)

    finally:
        with open(args.config, "w") as f:
            json_std.dump(original_config, f, indent=2)
        print("\nConfiguration restored to original.")


if __name__ == "__main__":
    main()
