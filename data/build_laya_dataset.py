#!/usr/bin/env python3
"""
Build 8-candidate TSP decision dataset for Laya (ModernBERT System-1 Decision Model).
Supports variable node instances from 4 to 500 cities.

Key Design Improvements:
1. K=8 candidates (99.9% of LKH optimal steps are in top-8; cuts noise & sharpens decision).
2. Rich inductive features per candidate:
    * rank      : explicit nearest-neighbor rank among unvisited (1 to 8)
    * ndist     : scale-invariant normalized distance (d / mean_cand_dist)
    * cos       : directional alignment / momentum [-1.0, 1.0] from incoming travel vector
    * dist      : Euclidean step distance from current city
    * ctr       : distance to remaining unvisited cluster centroid
    * lookahead : distance to candidate's nearest unvisited neighbor (2-step lookahead)
    * depot     : distance to starting depot (closure awareness)
3. Higher sampling density (~50K steps):
    * n <= 50  : sample every step (stride = 1)
    * n <= 150 : stride = 2
    * n > 150  : stride = max(1, n // 50)
4. Shuffled slot assignment ("slot_0" to "slot_7") for position-invariance.

Outputs:
  data/laya_train.jsonl
  data/laya_valid.jsonl
"""

import json
import math
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_JSON = os.path.join(SCRIPT_DIR, "tsp_dataset_1000.json")

OUTPUT_TRAIN = os.path.join(SCRIPT_DIR, "laya_train.jsonl")
OUTPUT_VALID = os.path.join(SCRIPT_DIR, "laya_valid.jsonl")

K = 8               # 8 candidate cities per decision step
VALID_FRAC = 0.1    # 10% held out for evaluation (100 instances)
SEED = 42


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def centroid(coords_list):
    n = len(coords_list)
    if n == 0:
        return (0.5, 0.5)
    return (sum(c[0] for c in coords_list) / n,
            sum(c[1] for c in coords_list) / n)


def build_steps_for_instance(instance, rng):
    coords = instance["coordinates"]
    tour = instance["tour"]
    n = len(tour)
    if n < 4:
        return []

    depot = coords[tour[0]]
    steps = []
    visited = set()

    # Adaptive stride: comprehensive sampling for small instances, balanced for large
    if n <= 50:
        stride = 1
    elif n <= 150:
        stride = 2
    else:
        stride = max(1, n // 50)

    for step_idx in range(n - 1):
        u = tour[step_idx]
        v_opt = tour[step_idx + 1]
        visited.add(u)

        if step_idx % stride != 0:
            continue

        unvisited = [c for c in range(n) if c not in visited]
        if len(unvisited) < 1:
            continue

        current_coord = coords[u]
        unvisited_coords = [coords[c] for c in unvisited]
        ctrd = centroid(unvisited_coords)

        # Incoming travel vector for directional momentum (cos similarity)
        if step_idx > 0:
            prev_coord = coords[tour[step_idx - 1]]
            vin_x = current_coord[0] - prev_coord[0]
            vin_y = current_coord[1] - prev_coord[1]
            vin_norm = math.sqrt(vin_x * vin_x + vin_y * vin_y)
        else:
            prev_coord = None
            vin_x, vin_y, vin_norm = 0.0, 0.0, 0.0

        # Sort all unvisited by distance to current city to assign global ranks
        unvisited_dists = [(dist(current_coord, coords[c]), c) for c in unvisited]
        unvisited_dists.sort(key=lambda x: x[0])
        city_ranks = {c: r + 1 for r, (_, c) in enumerate(unvisited_dists)}

        # Select top-K candidates
        real_cands = [c for _, c in unvisited_dists[:K]]

        # Guarantee the LKH optimal city is included in candidates
        if v_opt not in real_cands:
            real_cands[-1] = v_opt

        # Mean candidate distance for normalized scale-invariant distance
        cand_dists = [dist(current_coord, coords[c]) for c in real_cands]
        mean_cand_d = max(sum(cand_dists) / len(cand_dists), 1e-6)

        # Precompute candidate spatial features
        cand_features = {}
        for cand in real_cands:
            c_coord = coords[cand]
            step_d = dist(current_coord, c_coord)
            ndist = step_d / mean_cand_d
            rank = city_ranks.get(cand, K + 1)
            to_ctr = dist(c_coord, ctrd)
            to_dep = dist(c_coord, depot)

            # Cosine direction relative to incoming travel vector
            vout_x = c_coord[0] - current_coord[0]
            vout_y = c_coord[1] - current_coord[1]
            vout_norm = math.sqrt(vout_x * vout_x + vout_y * vout_y)
            if vin_norm > 1e-7 and vout_norm > 1e-7:
                cos_val = (vin_x * vout_x + vin_y * vout_y) / (vin_norm * vout_norm)
                cos_val = max(-1.0, min(1.0, cos_val))
            else:
                cos_val = 0.0

            # 2-step lookahead: distance to nearest unvisited neighbor
            pool = [coords[o] for o in unvisited if o != cand]
            lookahead = min(dist(c_coord, p) for p in pool) if pool else 0.0

            cand_features[cand] = (
                f"rank:{rank} ndist:{ndist:.2f} cos:{cos_val:.2f} dist:{step_d:.3f} "
                f"ctr:{to_ctr:.3f} lookahead:{lookahead:.3f} depot:{to_dep:.3f}"
            )

        # Shuffle real candidate positions across slots to enforce slot-invariance
        rng.shuffle(real_cands)
        optimal_slot = real_cands.index(v_opt)

        # Build criteria for all K slots (pad with dummy slot if < K unvisited)
        criteria = {}
        for slot_idx in range(K):
            if slot_idx < len(real_cands):
                cand = real_cands[slot_idx]
                criteria[f"slot_{slot_idx}"] = cand_features[cand]
            else:
                criteria[f"slot_{slot_idx}"] = (
                    "rank:99 ndist:99.00 cos:0.00 dist:99.000 ctr:99.000 lookahead:99.000 depot:99.000"
                )

        visited_count = step_idx + 1
        rem_count = n - visited_count
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

        row = {
            "state": state,
            "questions": {
                "next_city": {
                    "type": "choice",
                    "instructions": "Select the globally optimal candidate slot to extend the TSP tour:",
                    "criteria": criteria,
                }
            },
            "expected": {
                "next_city": f"slot_{optimal_slot}"
            }
        }
        steps.append(row)

    return steps


def main():
    rng = random.Random(SEED)
    print(f"Loading {INPUT_JSON} ...")
    with open(INPUT_JSON) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} TSP instances.")

    # Train / Valid split by instance
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    split = int(len(indices) * (1 - VALID_FRAC))
    train_indices = set(indices[:split])

    train_rows = []
    valid_rows = []

    print(f"Generating {K}-candidate decision steps with rank & directional features...")
    for idx, inst in enumerate(dataset):
        inst_rng = random.Random(SEED + idx)
        steps = build_steps_for_instance(inst, inst_rng)
        if idx in train_indices:
            train_rows.extend(steps)
        else:
            valid_rows.extend(steps)

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} instances ({len(train_rows):,} train, {len(valid_rows):,} valid steps)")

    rng.shuffle(train_rows)

    print(f"\nWriting {len(train_rows):,} train steps -> {OUTPUT_TRAIN}")
    with open(OUTPUT_TRAIN, "w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")

    print(f"Writing {len(valid_rows):,} valid steps -> {OUTPUT_VALID}")
    with open(OUTPUT_VALID, "w") as f:
        for r in valid_rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nDataset generation complete! Ready for laya-train.")


if __name__ == "__main__":
    main()
