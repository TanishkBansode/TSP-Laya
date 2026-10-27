#!/usr/bin/env python3
"""
Evaluate Trained Laya Decision Model on TSP Instances.
Directly loads the latest trained checkpoint (checkpoint_latest) and compares:
  1. Nearest Neighbor (Greedy Baseline)
  2. Laya Decision Model (Trained Policy)
  3. LKH (Globally Optimal Tour)
"""

import argparse
import json
import math
import os
import sys
from laya.agent import Agent

DATASET_PATH = "data/tsp_dataset_1000.json"
MODEL_PATH = "model/laya_tsp_model/checkpoint_latest"
K = 8


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def tour_length(coords, tour):
    return sum(dist(coords[tour[i]], coords[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))


def solve_nearest_neighbor(coords):
    unvisited = set(range(1, len(coords)))
    tour = [0]
    while unvisited:
        curr = coords[tour[-1]]
        nxt = min(unvisited, key=lambda c: dist(curr, coords[c]))
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour


def solve_with_laya(agent, coords):
    n = len(coords)
    visited = [0]
    unvisited = set(range(1, n))
    depot = coords[0]

    for step in range(n - 1):
        curr = coords[visited[-1]]
        if len(unvisited) == 1:
            visited.append(unvisited.pop())
            break

        unvisited_list = list(unvisited)
        ctrd = (
            sum(coords[c][0] for c in unvisited_list) / len(unvisited_list),
            sum(coords[c][1] for c in unvisited_list) / len(unvisited_list),
        )

        if step > 0:
            prev = coords[visited[-2]]
            vin = (curr[0] - prev[0], curr[1] - prev[1])
            vin_norm = math.hypot(*vin)
        else:
            prev = None
            vin, vin_norm = (0.0, 0.0), 0.0

        ranked = sorted(unvisited_list, key=lambda c: dist(curr, coords[c]))
        city_ranks = {c: r + 1 for r, c in enumerate(ranked)}
        cand_cities = ranked[:K]
        cand_dists = [dist(curr, coords[c]) for c in cand_cities]
        mean_d = max(sum(cand_dists) / len(cand_dists), 1e-6)

        criteria = {}
        for slot_idx in range(K):
            if slot_idx < len(cand_cities):
                cand = cand_cities[slot_idx]
                cc = coords[cand]
                step_d = dist(curr, cc)
                vout = (cc[0] - curr[0], cc[1] - curr[1])
                vout_norm = math.hypot(*vout)
                cos_val = (vin[0] * vout[0] + vin[1] * vout[1]) / (vin_norm * vout_norm) if vin_norm > 1e-7 and vout_norm > 1e-7 else 0.0
                cos_val = max(-1.0, min(1.0, cos_val))
                pool = [coords[o] for o in unvisited_list if o != cand]
                lookahead = min(dist(cc, p) for p in pool) if pool else 0.0
                criteria[f"slot_{slot_idx}"] = (
                    f"rank:{city_ranks[cand]} ndist:{step_d / mean_d:.2f} cos:{cos_val:.2f} dist:{step_d:.3f} "
                    f"ctr:{dist(cc, ctrd):.3f} lookahead:{lookahead:.3f} depot:{dist(cc, depot):.3f}"
                )
            else:
                criteria[f"slot_{slot_idx}"] = "rank:99 ndist:99.00 cos:0.00 dist:99.000 ctr:99.000 lookahead:99.000 depot:99.000"

        coord_line = f"Current: ({curr[0]:.3f}, {curr[1]:.3f}) | " + (f"Prev: ({prev[0]:.3f}, {prev[1]:.3f}) | " if prev else "") + f"Depot: ({depot[0]:.3f}, {depot[1]:.3f})"
        pct = (len(visited) / n) * 100.0
        state = f"TSP Step {len(visited)}/{n} ({pct:.1f}% visited, {len(unvisited)} remaining)\n{coord_line}\nCluster Centroid: ({ctrd[0]:.3f}, {ctrd[1]:.3f})"

        questions = {
            "next_city": {
                "type": "choice",
                "instructions": "Select the globally optimal candidate slot to extend the TSP tour:",
                "criteria": criteria,
            }
        }
        res = agent.predict(state, questions)
        probs = res["answers"]["next_city"].get("probabilities", {})

        valid_slots = [(slot, p) for slot, p in probs.items() if int(slot.replace("slot_", "")) < len(cand_cities)]
        picked_city = cand_cities[int(max(valid_slots, key=lambda x: x[1])[0].replace("slot_", ""))] if valid_slots else cand_cities[0]

        visited.append(picked_city)
        unvisited.remove(picked_city)

        if (step + 1) % 50 == 0 or (step + 1) == n - 1:
            sys.stdout.write(f"\r  Solved {step + 1}/{n} cities ...")
            sys.stdout.flush()

    print()
    return visited


def two_opt(coords, tour):
    best_tour = list(tour)
    n = len(tour)
    improved = True
    while improved:
        improved = False
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
    return best_tour, tour_length(coords, best_tour)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Laya latest checkpoint on TSP")
    parser.add_argument("instance_pos", nargs="?", type=int, default=None, help="Dataset Instance ID (0-999)")
    parser.add_argument("--instance", "-i", type=int, default=None, help="Dataset Instance ID (0-999)")
    args = parser.parse_args()

    inst_id = args.instance if args.instance is not None else (args.instance_pos if args.instance_pos is not None else 950)

    if not os.path.exists(DATASET_PATH):
        print(f"Error: {DATASET_PATH} not found.")
        sys.exit(1)

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    inst = dataset[inst_id if inst_id < len(dataset) else 0]
    coords = inst["coordinates"]
    lkh_tour = inst["tour"]
    lkh_len = tour_length(coords, lkh_tour)
    n = len(coords)

    print("==================================================================")
    print(f" Evaluating TSP Solver on Instance #{inst_id} ({n} cities)")
    print("==================================================================")

    # 1. Optimal LKH Tour
    print(f"\n[1] LKH Optimal Tour Length:        {lkh_len:.4f} (Benchmark Target)")

    # 2. Nearest Neighbor Baseline
    nn_tour = solve_nearest_neighbor(coords)
    nn_len = tour_length(coords, nn_tour)
    _, nn_opt_len = two_opt(coords, nn_tour)
    nn_gap = (nn_len - lkh_len) / lkh_len * 100
    nn_opt_gap = (nn_opt_len - lkh_len) / lkh_len * 100
    print(f"[2] Nearest Neighbor Raw:          {nn_len:.4f} (+{nn_gap:.2f}% gap)")
    print(f"    Nearest Neighbor + 2-Opt:      {nn_opt_len:.4f} (+{nn_opt_gap:.2f}% gap)")

    # 3. Laya Decision Model Policy (Always runs latest checkpoint)
    print(f"\n[3] Loading Laya Decision Model: {MODEL_PATH} ...")
    agent = Agent(MODEL_PATH)

    print("  Solving tour with Laya policy...")
    laya_tour = solve_with_laya(agent, coords)
    laya_len = tour_length(coords, laya_tour)
    _, laya_opt_len = two_opt(coords, laya_tour)
    laya_gap = (laya_len - lkh_len) / lkh_len * 100
    laya_opt_gap = (laya_opt_len - lkh_len) / lkh_len * 100

    print("------------------------------------------------------------------")
    print("RESULTS SUMMARY:")
    print(f"  * LKH Global Optimum:           {lkh_len:.4f}  (100.0%)")
    print(f"  * Nearest Neighbor (Raw):       {nn_len:.4f}  (+{nn_gap:.2f}% gap)")
    print(f"  * Nearest Neighbor + 2-Opt:     {nn_opt_len:.4f}  (+{nn_opt_gap:.2f}% gap)")
    print(f"  * Laya Decision Model (Raw):    {laya_len:.4f}  (+{laya_gap:.2f}% gap)")
    print(f"  * Laya Decision Model + 2-Opt:  {laya_opt_len:.4f}  (+{laya_opt_gap:.2f}% gap)")
    print("==================================================================")


if __name__ == "__main__":
    main()
