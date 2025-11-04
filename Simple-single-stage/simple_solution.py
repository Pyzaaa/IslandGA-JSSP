#!/usr/bin/env python3
"""
Simple heuristic solvers for P||C_max (identical parallel machines scheduling).
Includes:
  - Random Assignment
  - LPT (Longest Processing Time first)
"""

import json
import random
import time
from typing import List, Tuple


# -------------------------
# Helper functions
# -------------------------
def evaluate_assignment(assignment: List[int], processing_times: List[int], m: int) -> Tuple[float, List[float]]:
    """Compute the makespan and machine loads for a given assignment."""
    loads = [0.0] * m
    for j, machine in enumerate(assignment):
        loads[machine] += processing_times[j]
    makespan = max(loads)
    return makespan, loads


def lower_bound(processing_times: List[int], m: int) -> float:
    """Simple lower bound for makespan: max(job length, total work / m)."""
    return max(max(processing_times), sum(processing_times) / m)


def gap_to_lb(makespan: float, lb: float) -> float:
    """Compute % gap from lower bound."""
    return (makespan - lb) / lb * 100 if lb > 0 else 0


# -------------------------
# Heuristic 1: Random Assignment
# -------------------------
def random_assignment(processing_times: List[int], m: int) -> Tuple[float, List[float]]:
    """Assign each job to a random machine."""
    assignment = [random.randrange(m) for _ in range(len(processing_times))]
    return evaluate_assignment(assignment, processing_times, m)


# -------------------------
# Heuristic 2: LPT (Longest Processing Time first)
# -------------------------
def lpt_heuristic(processing_times: List[int], m: int) -> Tuple[float, List[float]]:
    """
    Sort jobs descending by processing time, assign each job to the least loaded machine.
    """
    n = len(processing_times)
    loads = [0.0] * m
    assignment = [-1] * n
    jobs_sorted = sorted(range(n), key=lambda j: processing_times[j], reverse=True)

    for j in jobs_sorted:
        k = min(range(m), key=lambda x: loads[x])  # least loaded machine
        assignment[j] = k
        loads[k] += processing_times[j]

    makespan = max(loads)
    return makespan, loads


# -------------------------
# Main solver function
# -------------------------
def run_simple_solvers(dataset_path="dataset.json"):
    """Load dataset and run both heuristics for each instance."""
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    results = []
    for inst in dataset:
        n, m = inst["jobs"], inst["machines"]
        p_times = inst["processing_times"]
        lb = lower_bound(p_times, m)

        print(f"\nSolving {inst['id']}  (n={n}, m={m})")
        print(f"Lower bound: {lb:.2f}")

        # --- Random Assignment ---
        start = time.time()
        makespan_rand, loads_rand = random_assignment(p_times, m)
        t_rand = time.time() - start
        print(f"Random assignment: makespan={makespan_rand:.2f}, gap={gap_to_lb(makespan_rand, lb):.2f}%, time={t_rand:.4f}s")

        # --- LPT Heuristic ---
        start = time.time()
        makespan_lpt, loads_lpt = lpt_heuristic(p_times, m)
        t_lpt = time.time() - start
        print(f"LPT heuristic:     makespan={makespan_lpt:.2f}, gap={gap_to_lb(makespan_lpt, lb):.2f}%, time={t_lpt:.4f}s")

        results.append({
            "instance": inst["id"],
            "random": {"makespan": makespan_rand, "gap": gap_to_lb(makespan_rand, lb), "time": t_rand},
            "lpt": {"makespan": makespan_lpt, "gap": gap_to_lb(makespan_lpt, lb), "time": t_lpt}
        })

    return results


# -------------------------
# Run directly (no CLI)
# -------------------------
if __name__ == "__main__":
    run_simple_solvers("p_parallel_deadlines_dataset.json")
