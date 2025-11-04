import json
import itertools
import time
from typing import List, Tuple
from math import inf


def schedule_with_release_deadline(assignment, processing_times, release_times, deadlines, m):
    """
    Compute completion times, total tardiness, and lateness stats for given assignment.
    """
    n = len(processing_times)
    machine_times = [0.0] * m
    completion_times = [0.0] * n
    tardiness_sum = 0.0
    late_jobs = 0

    machine_jobs = [[] for _ in range(m)]
    for j, mach in enumerate(assignment):
        machine_jobs[mach].append(j)

    for k in range(m):
        machine_jobs[k].sort(key=lambda j: release_times[j])
        t = 0.0
        for j in machine_jobs[k]:
            start = max(t, release_times[j])
            finish = start + processing_times[j]
            completion_times[j] = finish
            tardiness_j = max(0, finish - deadlines[j])
            if tardiness_j > 0:
                late_jobs += 1
            tardiness_sum += tardiness_j
            t = finish
        machine_times[k] = t

    makespan = max(machine_times)
    avg_tardiness = tardiness_sum / n
    late_fraction = late_jobs / n

    return makespan, tardiness_sum, avg_tardiness, late_fraction, machine_times, completion_times


def brute_force_scheduler(processing_times: List[int], release_times: List[int], deadlines: List[int], m: int):
    """
    Brute-force search of all possible assignments (each job can go to any of m machines).
    Minimizes total tardiness (ΣTj).
    """
    n = len(processing_times)
    start = time.time()
    best_tardiness = inf
    best_solution = None
    best_stats = None
    count = 0
    total = m ** n

    print(f"Brute-forcing {n} jobs on {m} machines → {total:,} combinations...")

    for assignment in itertools.product(range(m), repeat=n):
        count += 1
        makespan, tardiness, avg_tardiness, late_frac, loads, completion = schedule_with_release_deadline(
            assignment, processing_times, release_times, deadlines, m
        )
        if tardiness < best_tardiness:
            best_tardiness = tardiness
            best_solution = assignment
            best_stats = (makespan, avg_tardiness, late_frac, loads)
        if count % 100000 == 0:
            print(f"  Checked {count:,} / {total:,}...")

    total_time = time.time() - start
    makespan, avg_tardiness, late_frac, loads = best_stats

    return {
        "best_assignment": list(best_solution),
        "best_total_tardiness": best_tardiness,
        "avg_tardiness": avg_tardiness,
        "late_fraction": late_frac,
        "late_percent": late_frac * 100,
        "makespan": makespan,
        "machine_loads": loads,
        "runtime_seconds": total_time,
        "evaluated": count
    }


# -------------------------
# Runner for dataset
# -------------------------
def run_bruteforce(dataset_path="p_parallel_deadlines_dataset.json", max_jobs=10):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    results = []
    for inst in dataset:
        n = inst["jobs"]
        m = inst["machines"]
        print(f"\nSolving {inst['id']} (n={n}, m={m}) by brute force...")
        result = brute_force_scheduler(
            inst["processing_times"],
            inst["release_times"],
            inst["deadlines"],
            m,
        )
        print(
            f"  Total tardiness: {result['best_total_tardiness']:.2f}\n"
            f"  Avg tardiness:   {result['avg_tardiness']:.2f}\n"
            f"  Late jobs:       {result['late_percent']:.1f}%\n"
            f"  Makespan:        {result['makespan']:.2f}\n"
            f"  Runtime:         {result['runtime_seconds']:.2f}s\n"
        )
        results.append({"instance": inst["id"], **result})

    return results


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    results = run_bruteforce("p_parallel_deadlines_dataset.json", max_jobs=10)
