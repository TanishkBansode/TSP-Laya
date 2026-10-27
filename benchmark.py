#!/usr/bin/env python3
"""
Multi-Instance Batch Benchmark for Laya Decision Model.
Evaluates Laya on held-out test instances from tsp_dataset_1000.json:
- Stratified by instance size (20-50, 50-100, 100-300, 300+ cities)
- Compares:
  1. LKH (Globally Optimal Ground Truth)
  2. Nearest Neighbor (Greedy Baseline)
  3. Laya Decision Model (Trained Policy)
  4. Post-refinement with 2-Opt (NN vs Laya)
- Outputs:
  * Mean Optimality Gap
  * Win Rate vs Nearest Neighbor
  * Scaling Breakdown by node tier
  * JSON summary export (benchmark_results_batch50.json)

Usage:
  model/.venv/bin/python benchmark.py
  model/.venv/bin/python benchmark.py --num 50
"""

import argparse
import json
import math
import os
import random
import sys
import time

DATASET_PATH = "data/tsp_dataset_1000.json"
MODEL_PATH = "model/laya_tsp_model/checkpoint_latest"
K = 8
SEED = 42
VALID_FRAC = 0.1


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def tour_length(coords, tour):
    total = 0.0
    n = len(tour)
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        total += dist(coords[u], coords[v])
    return total


def solve_nearest_neighbor(coords):
    n = len(coords)
    visited = [False] * n
    tour = [0]
    visited[0] = True

    for _ in range(n - 1):
        curr = tour[-1]
        best_d = float("inf")
        best_c = -1
        for j in range(n):
            if not visited[j]:
                d = dist(coords[curr], coords[j])
                if d < best_d:
                    best_d = d
                    best_c = j
        tour.append(best_c)
        visited[best_c] = True
    return tour


def two_opt(coords, tour, max_passes=50):
    best_tour = list(tour)
    n = len(tour)
    improved = True
    pass_cnt = 0

    while improved and pass_cnt < max_passes:
        improved = False
        pass_cnt += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                u1, v1 = best_tour[i], best_tour[i + 1]
                u2, v2 = best_tour[j], best_tour[(j + 1) % n]
                curr_d = dist(coords[u1], coords[v1]) + dist(coords[u2], coords[v2])
                new_d = dist(coords[u1], coords[u2]) + dist(coords[v1], coords[v2])
                if new_d < curr_d - 1e-6:
                    best_tour[i + 1 : j + 1] = reversed(best_tour[i + 1 : j + 1])
                    improved = True
                    break
            if improved:
                break
    return best_tour


def solve_with_laya(agent, coords):
    n = len(coords)
    visited = set([0])
    tour = [0]
    depot = coords[0]

    for step in range(n - 1):
        curr = tour[-1]
        current_coord = coords[curr]
        unvisited = [c for c in range(n) if c not in visited]

        if len(unvisited) == 1:
            tour.append(unvisited[0])
            visited.add(unvisited[0])
            break

        unvisited_coords = [coords[c] for c in unvisited]
        ctrd_x = sum(c[0] for c in unvisited_coords) / len(unvisited)
        ctrd_y = sum(c[1] for c in unvisited_coords) / len(unvisited)
        ctrd = (ctrd_x, ctrd_y)

        if step > 0:
            prev_coord = coords[tour[-2]]
            vin_x = current_coord[0] - prev_coord[0]
            vin_y = current_coord[1] - prev_coord[1]
            vin_norm = math.sqrt(vin_x * vin_x + vin_y * vin_y)
        else:
            prev_coord = None
            vin_x, vin_y, vin_norm = 0.0, 0.0, 0.0

        unvisited_dists = [(dist(current_coord, coords[c]), c) for c in unvisited]
        unvisited_dists.sort(key=lambda x: x[0])
        city_ranks = {c: r + 1 for r, (_, c) in enumerate(unvisited_dists)}

        cand_cities = [c for _, c in unvisited_dists[:K]]
        cand_dists = [dist(current_coord, coords[c]) for c in cand_cities]
        mean_cand_d = max(sum(cand_dists) / len(cand_dists), 1e-6)

        criteria = {}
        for slot_idx in range(K):
            if slot_idx < len(cand_cities):
                cand = cand_cities[slot_idx]
                c_coord = coords[cand]
                step_d = dist(current_coord, c_coord)
                ndist = step_d / mean_cand_d
                rank = city_ranks.get(cand, K + 1)
                to_ctr = dist(c_coord, ctrd)
                to_dep = dist(c_coord, depot)

                vout_x = c_coord[0] - current_coord[0]
                vout_y = c_coord[1] - current_coord[1]
                vout_norm = math.sqrt(vout_x * vout_x + vout_y * vout_y)
                if vin_norm > 1e-7 and vout_norm > 1e-7:
                    cos_val = (vin_x * vout_x + vin_y * vout_y) / (vin_norm * vout_norm)
                    cos_val = max(-1.0, min(1.0, cos_val))
                else:
                    cos_val = 0.0

                pool = [coords[o] for o in unvisited if o != cand]
                lookahead = min(dist(c_coord, p) for p in pool) if pool else 0.0

                criteria[f"slot_{slot_idx}"] = (
                    f"rank:{rank} ndist:{ndist:.2f} cos:{cos_val:.2f} dist:{step_d:.3f} "
                    f"ctr:{to_ctr:.3f} lookahead:{lookahead:.3f} depot:{to_dep:.3f}"
                )
            else:
                criteria[f"slot_{slot_idx}"] = (
                    "rank:99 ndist:99.00 cos:0.00 dist:99.000 ctr:99.000 lookahead:99.000 depot:99.000"
                )

        visited_count = len(tour)
        rem_count = len(unvisited)
        pct = (visited_count / n) * 100.0

        if prev_coord is not None:
            coord_line = (
                f"Current: ({current_coord[0]:.3f}, {current_coord[1]:.3f}) | "
                f"Prev: ({prev_coord[0]:.3f}, {prev_coord[1]:.3f}) | "
                f"Depot: ({depot[0]:.3f}, {depot[1]:.3f})"
            )
        else:
            coord_line = (
                f"Current: ({current_coord[0]:.3f}, {current_coord[1]:.3f}) | "
                f"Depot: ({depot[0]:.3f}, {depot[1]:.3f})"
            )

        state = (
            f"TSP Step {visited_count}/{n} ({pct:.1f}% visited, {rem_count} remaining)\n"
            f"{coord_line}\n"
            f"Cluster Centroid: ({ctrd[0]:.3f}, {ctrd[1]:.3f})"
        )

        questions = {
            "next_city": {
                "type": "choice",
                "instructions": "Select the globally optimal candidate slot to extend the TSP tour:",
                "criteria": criteria,
            }
        }

        res = agent.predict(state, questions)
        probs = res["answers"]["next_city"].get("probabilities", {})

        valid_slots = [
            (slot, p) for slot, p in probs.items()
            if int(slot.replace("slot_", "")) < len(cand_cities)
        ]

        if valid_slots:
            best_slot = max(valid_slots, key=lambda x: x[1])[0]
            slot_idx = int(best_slot.replace("slot_", ""))
            picked_city = cand_cities[slot_idx]
        else:
            picked_city = cand_cities[0]

        tour.append(picked_city)
        visited.add(picked_city)

        if (step + 1) % 50 == 0 or (step + 1) == n - 1:
            sys.stdout.write(f"\r    Solved {step + 1}/{n} cities...")
            sys.stdout.flush()

    sys.stdout.write("\r" + " " * 40 + "\r")
    sys.stdout.flush()
    return tour


def select_stratified_instances(dataset, total_instances=50):
    rng = random.Random(SEED)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - VALID_FRAC))
    valid_indices = indices[split:]

    b_small = []    # 20 - 50
    b_medium = []   # 51 - 100
    b_large = []    # 101 - 300
    b_xlarge = []   # 301 - 500

    for idx in valid_indices:
        n = len(dataset[idx]["coordinates"])
        if 20 <= n <= 50:
            b_small.append(idx)
        elif 50 < n <= 100:
            b_medium.append(idx)
        elif 100 < n <= 300:
            b_large.append(idx)
        elif n > 300:
            b_xlarge.append(idx)

    k_small = len(b_small)
    k_med = min(len(b_medium), 11)
    k_large = min(len(b_large), 18)
    k_xlarge = total_instances - (k_small + k_med + k_large)

    selected = []
    selected.extend(sorted(b_small[:k_small], key=lambda i: len(dataset[i]["coordinates"])))
    selected.extend(sorted(b_medium[:k_med], key=lambda i: len(dataset[i]["coordinates"])))
    selected.extend(sorted(b_large[:k_large], key=lambda i: len(dataset[i]["coordinates"])))
    selected.extend(sorted(b_xlarge[:k_xlarge], key=lambda i: len(dataset[i]["coordinates"])))

    return selected


def run_benchmark(num_instances=50, model_path=MODEL_PATH):
    print("=" * 96)
    print(f"        TSP-LAYA BENCHMARK ({num_instances} HELD-OUT TEST INSTANCES)")
    print("=" * 96)

    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        sys.exit(1)

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    selected_indices = select_stratified_instances(dataset, num_instances)
    print(f"Selected {len(selected_indices)} stratified test instances (20 to 500 nodes).")
    print(f"Loading Laya Decision Agent from: {model_path} ...")
    from laya.agent import Agent
    agent = Agent(model_path)
    print(f"Agent loaded on device: {agent.device} ({agent.dtype})\n")

    results = []
    print(f"{'#':<3} | {'Inst':<5} | {'Nodes':<5} | {'LKH Opt':<8} | {'Laya Raw':<8} | {'Laya Gap':<9} | {'NN Raw':<8} | {'NN Gap':<9} | {'Laya+2O':<8} | {'L+2O Gap':<9} | {'NN+2O Gap':<9} | {'Win?'}")
    print("-" * 108)

    t_bench_start = time.time()
    for row_idx, inst_id in enumerate(selected_indices, 1):
        inst = dataset[inst_id]
        coords = inst["coordinates"]
        lkh_tour = inst["tour"]
        n = len(coords)

        lkh_len = tour_length(coords, lkh_tour)

        # Nearest Neighbor
        t_nn0 = time.time()
        nn_tour = solve_nearest_neighbor(coords)
        nn_len = tour_length(coords, nn_tour)
        nn_opt_tour = two_opt(coords, nn_tour)
        nn_opt_len = tour_length(coords, nn_opt_tour)
        t_nn = time.time() - t_nn0

        # Laya
        t_l0 = time.time()
        laya_tour = solve_with_laya(agent, coords)
        laya_len = tour_length(coords, laya_tour)
        laya_opt_tour = two_opt(coords, laya_tour)
        laya_opt_len = tour_length(coords, laya_opt_tour)
        t_laya = time.time() - t_l0

        laya_gap = (laya_len - lkh_len) / lkh_len * 100.0
        nn_gap = (nn_len - lkh_len) / lkh_len * 100.0
        laya_opt_gap = (laya_opt_len - lkh_len) / lkh_len * 100.0
        nn_opt_gap = (nn_opt_len - lkh_len) / lkh_len * 100.0

        raw_win = laya_len < nn_len
        opt_win = laya_opt_len < nn_opt_len
        win_str = "WIN " if raw_win else "loss"

        res_entry = {
            "index": row_idx,
            "instance_id": inst_id,
            "nodes": n,
            "lkh_len": lkh_len,
            "laya_len": laya_len,
            "laya_gap": laya_gap,
            "nn_len": nn_len,
            "nn_gap": nn_gap,
            "laya_opt_len": laya_opt_len,
            "laya_opt_gap": laya_opt_gap,
            "nn_opt_len": nn_opt_len,
            "nn_opt_gap": nn_opt_gap,
            "raw_win": raw_win,
            "opt_win": opt_win,
            "laya_time": t_laya,
            "nn_time": t_nn,
        }
        results.append(res_entry)

        print(f"{row_idx:<3d} | #{inst_id:<4d} | {n:<5d} | {lkh_len:<8.3f} | {laya_len:<8.3f} | {laya_gap:+8.2f}% | {nn_len:<8.3f} | {nn_gap:+8.2f}% | {laya_opt_len:<8.3f} | {laya_opt_gap:+8.2f}% | {nn_opt_gap:+8.2f}% | {win_str}")

    total_time = time.time() - t_bench_start

    # Aggregations
    mean_lkh_len = sum(r["lkh_len"] for r in results) / len(results)
    mean_laya_raw_len = sum(r["laya_len"] for r in results) / len(results)
    mean_nn_raw_len = sum(r["nn_len"] for r in results) / len(results)
    mean_laya_opt_len = sum(r["laya_opt_len"] for r in results) / len(results)
    mean_nn_opt_len = sum(r["nn_opt_len"] for r in results) / len(results)

    mean_laya_raw_gap = sum(r["laya_gap"] for r in results) / len(results)
    mean_nn_raw_gap = sum(r["nn_gap"] for r in results) / len(results)
    mean_laya_opt_gap = sum(r["laya_opt_gap"] for r in results) / len(results)
    mean_nn_opt_gap = sum(r["nn_opt_gap"] for r in results) / len(results)

    total_raw_wins = sum(1 for r in results if r["raw_win"])
    total_opt_wins = sum(1 for r in results if r["opt_win"])
    raw_win_rate = total_raw_wins / len(results) * 100.0
    opt_win_rate = total_opt_wins / len(results) * 100.0

    print("=" * 116)
    print("                    AGGREGATE BENCHMARK RESULTS")
    print("=" * 116)
    print(f"Total Evaluated Instances:           {len(results)}")
    print(f"Total Benchmark Runtime:             {total_time:.1f}s ({total_time / len(results):.2f}s per instance)")
    print(f"Mean LKH Optimal Tour Length:        {mean_lkh_len:.3f}")
    print(f"Mean Tour Length (Raw Laya):         {mean_laya_raw_len:.3f} ({mean_laya_raw_gap:+.2f}% gap)")
    print(f"Mean Tour Length (Raw NN):           {mean_nn_raw_len:.3f} ({mean_nn_raw_gap:+.2f}% gap)")
    print(f"Laya Raw Improvement vs NN:          {mean_nn_raw_gap - mean_laya_raw_gap:+.2f}% absolute margin")
    print(f"Laya Raw Win Rate vs NN:             {total_raw_wins}/{len(results)} ({raw_win_rate:.1f}%)")
    print(f"Mean Tour Length (Raw Laya + 2-Opt): {mean_laya_opt_len:.3f} ({mean_laya_opt_gap:+.2f}% gap)")
    print(f"Mean Tour Length (NN + 2-Opt):       {mean_nn_opt_len:.3f} ({mean_nn_opt_gap:+.2f}% gap)")
    print(f"Laya+2Opt Win Rate vs NN+2Opt:       {total_opt_wins}/{len(results)} ({opt_win_rate:.1f}%)")
    print("-" * 116)

    # Scaling Breakdown by bucket
    buckets = [
        ("Small (20-50 nodes)", lambda n: 20 <= n <= 50),
        ("Medium (51-100 nodes)", lambda n: 50 < n <= 100),
        ("Large (101-300 nodes)", lambda n: 100 < n <= 300),
        ("X-Large (301-500 nodes)", lambda n: n > 300),
    ]

    bucket_stats = []
    for label, pred in buckets:
        sub = [r for r in results if pred(r["nodes"])]
        if not sub:
            continue
        sub_lkh = sum(r["lkh_len"] for r in sub) / len(sub)
        sub_laya_len = sum(r["laya_len"] for r in sub) / len(sub)
        sub_laya = sum(r["laya_gap"] for r in sub) / len(sub)
        sub_nn_len = sum(r["nn_len"] for r in sub) / len(sub)
        sub_nn = sum(r["nn_gap"] for r in sub) / len(sub)
        sub_laya_opt_len = sum(r["laya_opt_len"] for r in sub) / len(sub)
        sub_laya_opt = sum(r["laya_opt_gap"] for r in sub) / len(sub)
        sub_nn_opt_len = sum(r["nn_opt_len"] for r in sub) / len(sub)
        sub_nn_opt = sum(r["nn_opt_gap"] for r in sub) / len(sub)
        sub_raw_wins = sum(1 for r in sub if r["raw_win"])
        raw_win_pct = sub_raw_wins / len(sub) * 100.0
        sub_opt_wins = sum(1 for r in sub if r["opt_win"])
        opt_win_pct = sub_opt_wins / len(sub) * 100.0
        bucket_stats.append({
            "bucket": label,
            "count": len(sub),
            "mean_lkh_len": sub_lkh,
            "mean_laya_len": sub_laya_len,
            "mean_laya_gap": sub_laya,
            "mean_nn_len": sub_nn_len,
            "mean_nn_gap": sub_nn,
            "mean_laya_opt_len": sub_laya_opt_len,
            "mean_laya_opt_gap": sub_laya_opt,
            "mean_nn_opt_len": sub_nn_opt_len,
            "mean_nn_opt_gap": sub_nn_opt,
            "raw_wins": sub_raw_wins,
            "raw_win_rate": raw_win_pct,
            "opt_wins": sub_opt_wins,
            "opt_win_rate": opt_win_pct,
        })

    # Table 1: Raw Construction Policies (No 2-Opt)
    print("\n" + "=" * 116)
    print("                 TABLE 1: RAW CONCURRENT POLICIES (NO 2-OPT POST-PROCESSING)")
    print("=" * 116)
    print(f"{'Node Bucket':<24} | {'Count':<5} | {'Mean LKH':<8} | {'Raw Laya':<8} | {'Laya Gap':<9} | {'Raw NN':<8} | {'NN Gap':<9} | {'Laya Win Rate'}")
    print("-" * 116)
    for b in bucket_stats:
        print(f"{b['bucket']:<24} | {b['count']:<5d} | {b['mean_lkh_len']:<8.3f} | {b['mean_laya_len']:<8.3f} | {b['mean_laya_gap']:+8.2f}% | {b['mean_nn_len']:<8.3f} | {b['mean_nn_gap']:+8.2f}% | {b['raw_win_rate']:4.1f}% ({b['raw_wins']}/{b['count']})")
    print("-" * 116)
    print(f"{'OVERALL (20-500 nodes)':<24} | {len(results):<5d} | {mean_lkh_len:<8.3f} | {mean_laya_raw_len:<8.3f} | {mean_laya_raw_gap:+8.2f}% | {mean_nn_raw_len:<8.3f} | {mean_nn_raw_gap:+8.2f}% | {raw_win_rate:4.1f}% ({total_raw_wins}/{len(results)})")
    print("=" * 116)

    # Table 2: 2-Opt Post-Search Refinement
    print("\n" + "=" * 116)
    print("                 TABLE 2: POST-PROCESSED POLICIES (+ 2-OPT LOCAL SEARCH)")
    print("=" * 116)
    print(f"{'Node Bucket':<24} | {'Count':<5} | {'Mean LKH':<8} | {'Laya+2Opt':<9} | {'L+2Opt Gap':<10} | {'NN+2Opt':<8} | {'NN+2O Gap':<10} | {'2-Opt Win Rate'}")
    print("-" * 116)
    for b in bucket_stats:
        print(f"{b['bucket']:<24} | {b['count']:<5d} | {b['mean_lkh_len']:<8.3f} | {b['mean_laya_opt_len']:<9.3f} | {b['mean_laya_opt_gap']:+9.2f}% | {b['mean_nn_opt_len']:<8.3f} | {b['mean_nn_opt_gap']:+9.2f}% | {b['opt_win_rate']:4.1f}% ({b['opt_wins']}/{b['count']})")
    print("-" * 116)
    print(f"{'OVERALL (20-500 nodes)':<24} | {len(results):<5d} | {mean_lkh_len:<8.3f} | {mean_laya_opt_len:<9.3f} | {mean_laya_opt_gap:+9.2f}% | {mean_nn_opt_len:<8.3f} | {mean_nn_opt_gap:+9.2f}% | {opt_win_rate:4.1f}% ({total_opt_wins}/{len(results)})")
    print("=" * 116)

    # Save to JSON
    report = {
        "timestamp": time.time(),
        "total_instances": len(results),
        "total_time_sec": total_time,
        "mean_lkh_len": mean_lkh_len,
        "mean_laya_raw_len": mean_laya_raw_len,
        "mean_nn_raw_len": mean_nn_raw_len,
        "mean_laya_opt_len": mean_laya_opt_len,
        "mean_nn_opt_len": mean_nn_opt_len,
        "mean_laya_raw_gap": mean_laya_raw_gap,
        "mean_nn_raw_gap": mean_nn_raw_gap,
        "mean_laya_opt_gap": mean_laya_opt_gap,
        "mean_nn_opt_gap": mean_nn_opt_gap,
        "raw_win_rate": raw_win_rate,
        "opt_win_rate": opt_win_rate,
        "bucket_stats": bucket_stats,
        "instance_results": results,
    }
    with open("benchmark_results_batch50.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved detailed benchmark data to benchmark_results_batch50.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Instance Batch Benchmark for Laya Decision Model")
    parser.add_argument("--num", type=int, default=50, help="Number of held-out instances to benchmark (default: 50)")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to checkpoint (default: checkpoint_latest)")
    args = parser.parse_args()
    run_benchmark(num_instances=args.num, model_path=args.model)
