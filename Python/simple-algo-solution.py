#!/usr/bin/env python3
"""
Display a random solution and a heuristic solution (NEH)
for a flow-shop scheduling problem.

Input format (data.txt):
N M
p11 p12 ... p1M
p21 p22 ... p2M
...
pN1 pN2 ... pNM
"""

import random
from typing import List

# ------------------ Flow shop utility functions ------------------

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


# ------------------ Heuristics ------------------

def random_solution(num_jobs: int) -> List[int]:
    """Generate a random job sequence."""
    order = list(range(num_jobs))
    random.shuffle(order)
    return order

def neh_heuristic(times: List[List[int]]) -> List[int]:
    """
    NEH heuristic for flow shop scheduling.
    1. Sort jobs by total processing time (descending)
    2. Build schedule incrementally inserting the next job
       into the position giving minimal makespan.
    """
    num_jobs = len(times)
    # Step 1: sort jobs by total processing time
    job_order = sorted(range(num_jobs), key=lambda j: -sum(times[j]))

    # Step 2: build sequence incrementally
    seq = [job_order[0]]
    for j in job_order[1:]:
        best_seq = None
        best_makespan = float("inf")
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [j] + seq[pos:]
            cmax = makespan(trial, times)
            if cmax < best_makespan:
                best_seq = trial
                best_makespan = cmax
        seq = best_seq
    return seq


# ------------------ Display utilities ------------------

def print_schedule(order: List[int], times: List[List[int]]):
    """Print completion times for each job and machine."""
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

    # Optional random seed for reproducibility
    random.seed(None)

    # Read data
    times = read_input_from_file(input_path)
    num_jobs = len(times)

    print(f"Loaded {num_jobs} jobs and {len(times[0])} machines from '{input_path}'")

    # Generate and evaluate random solution
    rand_order = random_solution(num_jobs)
    rand_makespan = makespan(rand_order, times)

    # Generate heuristic (NEH) solution
    neh_order = neh_heuristic(times)
    neh_makespan = makespan(neh_order, times)

    # Display results
    print("\n=== RANDOM SOLUTION ===")
    print("Job order (1-based):", [j+1 for j in rand_order])
    print("Makespan:", rand_makespan)

    print("\n=== HEURISTIC (NEH) SOLUTION ===")
    print("Job order (1-based):", [j+1 for j in neh_order])
    print("Makespan:", neh_makespan)

    # Optionally print full schedule for heuristic
    print_schedule(neh_order, times)

    print("\nComparison:")
    print(f"Random makespan = {rand_makespan}")
    print(f"NEH makespan    = {neh_makespan}")
    improvement = 100 * (rand_makespan - neh_makespan) / rand_makespan if rand_makespan > 0 else 0
    print(f"Improvement: {improvement:.2f}%")
