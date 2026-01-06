#!/usr/bin/env python3
"""
Generate random flow shop scheduling input data and save to a text file.

Output format:
N M
p11 p12 ... p1M
p21 p22 ... p2M
...
pN1 pN2 ... pNM
"""

import random

def generate_flowshop_data(num_jobs: int, num_machines: int,
                           min_time: int = 1, max_time: int = 10) -> list[list[int]]:
    """Generate random processing times matrix."""
    return [
        [random.randint(min_time, max_time) for _ in range(num_machines)]
        for _ in range(num_jobs)
    ]

def save_flowshop_data(filename: str, data: list[list[int]]) -> None:
    """Save generated data to a text file."""
    num_jobs = len(data)
    num_machines = len(data[0]) if data else 0
    with open(filename, "w") as f:
        f.write(f"{num_jobs} {num_machines}\n")
        for row in data:
            f.write(" ".join(map(str, row)) + "\n")
    print(f"Saved flow shop data to '{filename}' ({num_jobs} jobs, {num_machines} machines).")

if __name__ == "__main__":
    # <<< CONFIGURE HERE >>>
    output_file = "data7.txt"     # file name to save
    num_jobs = 2000                 # number of jobs
    num_machines = 30             # number of machines
    min_proc_time = 1            # min processing time
    max_proc_time = 60           # max processing time
    random_seed = None             # for reproducibility (set to None for random each time)

    if random_seed is not None:
        random.seed(random_seed)

    data = generate_flowshop_data(num_jobs, num_machines, min_proc_time, max_proc_time)
    save_flowshop_data(output_file, data)

    print("Generated processing times:")
    for i, row in enumerate(data, start=1):
        print(f"Job {i}: {row}")
