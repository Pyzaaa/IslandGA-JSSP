#!/usr/bin/env python3
"""
Find the optimal (best possible) flow-shop scheduling solution
by evaluating all possible job permutations (brute-force search).

Now includes:
- Estimated remaining time
- Count of how many permutations achieve the best makespan
"""

import itertools
from typing import List
import math
import time

# ------------------ Utilities ------------------

def read_input_from_file(path: str) -> List[List[int]]:
    """Read flow-shop data from text file."""
    with open(path, 'r') as f:
        parts = f.read().strip().split()
    if not parts:
        raise ValueError("Empty input file.")
    n = int(parts[0])
    m = int(parts[1])
    values = parts[2:]
    if len(values) != n * m:
        raise ValueError(f"Expected {n*m} numbers, got {len(values)}.")
    times = []
    idx = 0
    for _ in range(n):
        row = [int(values[idx + j]) for j in range(m)]
        idx += m
        times.append(row)
    return times

def makespan(order: List[int], times: List[List[int]]) -> int:
    """Compute makespan (Cmax) for a given job order."""
    num_machines = len(times[0])
    machine_ready = [0] * num_machines
    for job_idx in order:
        prev_completion = 0
        for m in range(num_machines):
            start = max(machine_ready[m], prev_completion)
            completion = start + times[job_idx][m]
            machine_ready[m] = completion
            prev_completion = completion
    return machine_ready[-1]


# ------------------ Optimal Search ------------------

def brute_force_optimal(times: List[List[int]]):
    """Evaluate all permutations and find the one with the minimal makespan."""
    num_jobs = len(times)
    total_permutations = math.factorial(num_jobs)
    best_order = None
    best_makespan = math.inf
    num_best = 0
    count = 0

    print(f"Searching all {total_permutations} permutations...")
    start_time = time.time()

    for order in itertools.permutations(range(num_jobs)):
        cmax = makespan(order, times)
        count += 1

        if cmax < best_makespan:
            best_makespan = cmax
            best_order = order
            num_best = 1
        elif cmax == best_makespan:
            num_best += 1

        # Progress update every 10,000 permutations (adjust as desired)
        if count % 10000 == 0:
            elapsed = time.time() - start_time
            avg_time_per_perm = elapsed / count
            remaining = (total_permutations - count) * avg_time_per_perm
            eta_min = remaining / 60
            print(
                f"Checked {count:,}/{total_permutations:,} "
                f"({count / total_permutations * 100:.2f}%) | "
                f"Best = {best_makespan} | "
                f"Elapsed = {elapsed:.1f}s | "
                f"ETA ≈ {eta_min:.1f} min"
            )

    total_time = time.time() - start_time
    return list(best_order), best_makespan, count, num_best, total_time


def print_schedule(order: List[int], times: List[List[int]]):
    """Prints a simple schedule table with completion times for each job on each machine."""
    n = len(order)
    m = len(times[0])
    comp = [[0]*m for _ in range(n)]
    machine_ready = [0]*m
    for seq_pos, job in enumerate(order):
        prev = 0
        for mm in range(m):
            start = max(machine_ready[mm], prev)
            completion = start + times[job][mm]
            comp[seq_pos][mm] = completion
            machine_ready[mm] = completion
            prev = completion
    print("\nSchedule (completion times):")
    header = "Pos\tJob\t" + "\t".join([f"M{j+1}" for j in range(m)])
    print(header)
    for pos in range(n):
        print(f"{pos+1}\t{order[pos]+1}\t" + "\t".join(str(comp[pos][mm]) for mm in range(m)))
    print("Makespan (Cmax):", comp[-1][-1])


# ------------------ Main ------------------

if __name__ == "__main__":
    # <<< CHANGE THIS TO YOUR INPUT FILE PATH >>>
    input_path = "data.txt"

    # Read input
    times = read_input_from_file(input_path)
    num_jobs = len(times)
    num_machines = len(times[0])

    print(f"Loaded {num_jobs} jobs and {num_machines} machines from '{input_path}'.")

    # Find optimal schedule (brute force)
    best_order, best_val, total_checked, num_best, total_time = brute_force_optimal(times)

    # Display results
    print("\n=== OPTIMAL SOLUTION FOUND ===")
    print("Job order (1-based):", [j+1 for j in best_order])
    print("Optimal makespan:", best_val)
    print(f"Total permutations checked: {total_checked:,}")
    print(f"Number of optimal permutations: {num_best:,}")
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    print_schedule(best_order, times)
