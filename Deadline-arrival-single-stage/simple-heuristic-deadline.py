import json
import random
import time
from typing import List, Tuple


# -------------------------
# Scheduling simulation
# -------------------------
def simulate_schedule(assignment, processing_times, release_times, deadlines, m):
    """Compute total tardiness, makespan, etc., for a given assignment."""
    n = len(processing_times)
    machine_jobs = [[] for _ in range(m)]
    for j, mach in enumerate(assignment):
        machine_jobs[mach].append(j)

    machine_times = [0.0] * m
    tardiness = 0.0
    completion_times = [0.0] * n
    late_jobs = 0

    for k in range(m):
        # Sort by release time
        machine_jobs[k].sort(key=lambda j: release_times[j])
        t = 0.0
        for j in machine_jobs[k]:
            start = max(t, release_times[j])
            finish = start + processing_times[j]
            completion_times[j] = finish
            tardiness_j = max(0, finish - deadlines[j])
            tardiness += tardiness_j
            if tardiness_j > 0:
                late_jobs += 1
            t = finish
        machine_times[k] = t

    makespan = max(machine_times)
    return {
        "makespan": makespan,
        "total_tardiness": tardiness,
        "late_jobs": late_jobs,
        "late_fraction": late_jobs / n,
    }


# -------------------------
# Heuristic 1: Random assignment
# -------------------------
def heuristic_random(processing_times, release_times, deadlines, m):
    n = len(processing_times)
    assignment = [random.randrange(m) for _ in range(n)]
    return simulate_schedule(assignment, processing_times, release_times, deadlines, m)


# -------------------------
# Heuristic 2: Earliest Release Time (ERT)
# -------------------------
def heuristic_earliest_release(processing_times, release_times, deadlines, m):
    n = len(processing_times)
    assignment = [-1] * n
    machine_times = [0.0] * m

    # Sort jobs by release time
    jobs = sorted(range(n), key=lambda j: release_times[j])
    for j in jobs:
        # Assign to least loaded machine
        k = min(range(m), key=lambda x: machine_times[x])
        assignment[j] = k
        machine_times[k] = max(machine_times[k], release_times[j]) + processing_times[j]
    return simulate_schedule(assignment, processing_times, release_times, deadlines, m)


# -------------------------
# Heuristic 3: Earliest Deadline First (EDF)
# -------------------------
def heuristic_earliest_deadline(processing_times, release_times, deadlines, m):
    n = len(processing_times)
    assignment = [-1] * n
    machine_times = [0.0] * m

    jobs = sorted(range(n), key=lambda j: deadlines[j])
    for j in jobs:
        k = min(range(m), key=lambda x: machine_times[x])
        assignment[j] = k
        machine_times[k] = max(machine_times[k], release_times[j]) + processing_times[j]
    return simulate_schedule(assignment, processing_times, release_times, deadlines, m)


# -------------------------
# Heuristic 4: Shortest Processing Time (SPT)
# -------------------------
def heuristic_shortest_processing(processing_times, release_times, deadlines, m):
    n = len(processing_times)
    assignment = [-1] * n
    machine_times = [0.0] * m

    jobs = sorted(range(n), key=lambda j: processing_times[j])
    for j in jobs:
        k = min(range(m), key=lambda x: machine_times[x])
        assignment[j] = k
        machine_times[k] = max(machine_times[k], release_times[j]) + processing_times[j]
    return simulate_schedule(assignment, processing_times, release_times, deadlines, m)


# -------------------------
# Evaluation utility
# -------------------------
def run_heuristics(dataset_path="p_parallel_deadlines_dataset.json"):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    methods = {
        "Random": heuristic_random,
        "EarliestRelease": heuristic_earliest_release,
        "EarliestDeadline": heuristic_earliest_deadline,
        "ShortestProc": heuristic_shortest_processing,
    }

    results = []
    for inst in dataset:
        print(f"\nInstance {inst['id']} (n={inst['jobs']}, m={inst['machines']})")
        for name, method in methods.items():
            start = time.time()
            res = method(
                inst["processing_times"],
                inst["release_times"],
                inst["deadlines"],
                inst["machines"],
            )
            res["runtime"] = time.time() - start
            res["method"] = name
            results.append({**res, "instance": inst["id"]})
            print(
                f"  {name:16s}  tardiness={res['total_tardiness']:.1f}  "
                f"makespan={res['makespan']:.1f}  "
                f"late%={100*res['late_fraction']:.1f}%  "
                f"time={res['runtime']:.3f}s"
            )
    return results


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    run_heuristics("p_parallel_deadlines_dataset.json")
