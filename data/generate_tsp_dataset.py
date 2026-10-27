#!/usr/bin/env python3
"""
Generate synthetic 2D Euclidean Symmetric TSP instances with optimal route sequences.
Uses the high-performance LKH (Lin-Kernighan-Helsgaun) solver to find the optimal tours.
"""

import argparse
import concurrent.futures
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LKH_BIN = os.path.join(SCRIPT_DIR, "bin", "lkh")
if not os.path.isfile(LKH_BIN):
    LKH_BIN = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "bin", "lkh"))


def solve_instance(instance_id, coords, scale=1_000_000):
    """
    Solve a 2D Euclidean TSP instance with LKH.
    coords: list of [x, y] floats in [0, 1]
    Returns: (tour_indices, tour_length)
    """
    n = len(coords)
    pid = os.getpid()
    unique_tag = f"{pid}_{instance_id}_{time.time_ns()}"
    tmp_tsp = os.path.join(tempfile.gettempdir(), f"tsp_{unique_tag}.tsp")
    tmp_par = os.path.join(tempfile.gettempdir(), f"tsp_{unique_tag}.par")
    tmp_tour = os.path.join(tempfile.gettempdir(), f"tsp_{unique_tag}.tour")

    try:
        # Write TSPLIB problem specification
        with open(tmp_tsp, "w") as f:
            f.write(f"NAME : inst_{instance_id}\n")
            f.write("TYPE : TSP\n")
            f.write(f"DIMENSION : {n}\n")
            f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")
            for i, (x, y) in enumerate(coords, 1):
                f.write(f"{i} {int(round(x * scale))} {int(round(y * scale))}\n")
            f.write("EOF\n")

        # Write LKH parameter file
        with open(tmp_par, "w") as f:
            f.write(f"PROBLEM_FILE = {tmp_tsp}\n")
            f.write(f"OUTPUT_TOUR_FILE = {tmp_tour}\n")
            f.write("RUNS = 1\n")
            f.write("TRACE_LEVEL = 0\n")

        # Run LKH
        subprocess.run(
            [LKH_BIN, tmp_par],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        # Parse tour (convert 1-indexed to 0-indexed)
        tour = []
        with open(tmp_tour, "r") as f:
            reading = False
            for line in f:
                line = line.strip()
                if line == "TOUR_SECTION":
                    reading = True
                    continue
                if reading:
                    if line in ("-1", "EOF") or not line:
                        break
                    tour.append(int(line) - 1)

        # Calculate exact float tour length in [0, 1] space
        tour_length = 0.0
        for i in range(len(tour)):
            n1 = coords[tour[i]]
            n2 = coords[tour[(i + 1) % len(tour)]]
            tour_length += math.hypot(n1[0] - n2[0], n1[1] - n2[1])

        return {
            "id": instance_id,
            "num_nodes": n,
            "coordinates": coords,
            "tour": tour,
            "tour_length": round(tour_length, 6),
        }

    finally:
        for p in (tmp_tsp, tmp_par, tmp_tour):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def generate_single_instance(args_tuple):
    instance_id, num_nodes, seed = args_tuple
    rng = random.Random(seed)
    coords = [[round(rng.random(), 6), round(rng.random(), 6)] for _ in range(num_nodes)]
    return solve_instance(instance_id, coords)


def main():
    parser = argparse.ArgumentParser(description="Generate solved 2D Symmetric TSP instances for ML models.")
    parser.add_argument("--num_instances", type=int, default=1000, help="Number of TSP instances to generate (default: 1000)")
    parser.add_argument("--min_nodes", type=int, default=10, help="Minimum number of cities per instance (default: 10)")
    parser.add_argument("--max_nodes", type=int, default=100, help="Maximum number of cities per instance (default: 100)")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory for generated files (default: data)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42)")
    parser.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="Parallel solver workers")

    args = parser.parse_args()

    if not os.path.isfile(LKH_BIN):
        sys.exit(f"Error: LKH solver binary not found at '{LKH_BIN}'.")

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"=== Generating {args.num_instances} Symmetric TSP Instances ===")
    print(f"Node size range : [{args.min_nodes}, {args.max_nodes}]")
    print(f"Parallel workers: {args.num_workers}")
    print(f"Random seed     : {args.seed}")

    tasks = []
    for i in range(args.num_instances):
        num_nodes = rng.randint(args.min_nodes, args.max_nodes)
        sub_seed = rng.randint(1, 10_000_000)
        tasks.append((i, num_nodes, sub_seed))

    start_time = time.time()
    results = []
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(generate_single_instance, t): t[0] for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % 100 == 0 or completed == args.num_instances:
                elapsed = time.time() - start_time
                print(f"Progress: {completed}/{args.num_instances} instances solved ({elapsed:.2f}s elapsed)")

    # Sort results by instance ID
    results.sort(key=lambda x: x["id"])

    # 1. Save unified JSON format
    json_path = os.path.join(args.output_dir, f"tsp_dataset_{args.num_instances}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # 2. Save standard ML line-based text format (Google Pointer Network / GNN convention)
    txt_path = os.path.join(args.output_dir, f"tsp_dataset_{args.num_instances}.txt")
    with open(txt_path, "w") as f:
        for item in results:
            coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in item["coordinates"])
            tour_str = " ".join(str(idx) for idx in item["tour"])
            f.write(f"{coord_str} output {tour_str}\n")

    total_time = time.time() - start_time
    avg_nodes = sum(r["num_nodes"] for r in results) / len(results)
    avg_length = sum(r["tour_length"] for r in results) / len(results)

    print("\n=== Dataset Successfully Generated ===")
    print(f"Total instances : {len(results)}")
    print(f"Average nodes   : {avg_nodes:.1f}")
    print(f"Average length  : {avg_length:.4f}")
    print(f"Total time      : {total_time:.2f}s ({len(results)/total_time:.1f} instances/sec)")
    print(f"Saved JSON file : {json_path}")
    print(f"Saved TXT file  : {txt_path}")


if __name__ == "__main__":
    main()
