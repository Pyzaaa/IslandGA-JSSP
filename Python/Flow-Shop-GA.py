#!/usr/bin/env python3
"""
Flow-shop scheduling using a Genetic Algorithm (GA)
with stagnation detection, bound proximity, and optional NEH initialization.

Data file format:
4 3
1 3 8
9 3 5
7 8 6
4 8 7
"""

import random
from typing import List, Tuple
import statistics




# ------------------ Flow-shop evaluation ------------------

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


# ------------------ Genetic operators ------------------

def random_permutation(n: int) -> List[int]:
    p = list(range(n))
    random.shuffle(p)
    return p

def tournament_selection(pop, fitnesses, k=3):
    """Select one individual via tournament selection."""
    selected = random.sample(range(len(pop)), k)
    best = min(selected, key=lambda i: fitnesses[i])
    return pop[best][:]

def order_crossover(p1, p2):
    """Order crossover (OX) for permutations."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    def ox(parent1, parent2):
        child = [-1] * n
        child[a:b+1] = parent1[a:b+1]
        fill_pos = (b + 1) % n
        p2pos = (b + 1) % n
        while -1 in child:
            if parent2[p2pos] not in child:
                child[fill_pos] = parent2[p2pos]
                fill_pos = (fill_pos + 1) % n
            p2pos = (p2pos + 1) % n
        return child
    return ox(p1, p2), ox(p2, p1)

def swap_mutation(ind, rate):
    """Swap mutation."""
    if random.random() < rate:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]


# ------------------ NEH heuristic ------------------

def neh_sequence(times: List[List[int]]) -> List[int]:
    """Construct a job sequence using the NEH heuristic."""
    n = len(times)
    jobs_sorted = sorted(range(n), key=lambda j: -sum(times[j]))
    seq = [jobs_sorted[0]]
    for j in jobs_sorted[1:]:
        best_seq = None
        best_val = float("inf")
        for pos in range(len(seq) + 1):
            trial = seq[:pos] + [j] + seq[pos:]
            val = makespan(trial, times)
            if val < best_val:
                best_val = val
                best_seq = trial
        seq = best_seq
    return seq


# ------------------ Bound calculation ------------------

def compute_bounds(times: List[List[int]]) -> Tuple[int, int, int, int]:
    """Return (LB_machine, LB_job, LB_simple, UB_trivial)."""
    n = len(times)
    m = len(times[0])
    lb_machine = max(sum(times[j][k] for j in range(n)) for k in range(m))
    lb_job = max(sum(times[j][k] for k in range(m)) for j in range(n))
    ub_trivial = sum(sum(row) for row in times)
    return lb_machine, lb_job, max(lb_machine, lb_job), ub_trivial


# ------------------ GA main loop ------------------

def genetic_algorithm(times: List[List[int]],
                      pop_size=100,
                      generations=500,
                      cx_prob=0.9,
                      mut_rate=0.2,
                      elitism=1,
                      tournament_k=3,
                      stagnation_limit=200,
                      mut_increase_threshold=500,
                      init_mode="random",
                      seed=None,
                      diversity_sample=10):
    """
    Run GA and return best permutation, its makespan, and history.

    init_mode: "random" (default) or "neh"
    mut_increase_threshold: after this many stagnant generations,
                            mutation rate temporarily increases 1.5×
    """
    if seed is not None:
        random.seed(seed)

    num_jobs = len(times)

    # ---------- Initialization ----------
    population = []
    if init_mode.lower() == "neh":
        neh = neh_sequence(times)
        population.append(neh)
        # Fill rest with small NEH perturbations
        for _ in range(pop_size - 1):
            p = neh[:]
            if num_jobs > 1:
                i, j = random.sample(range(num_jobs), 2)
                p[i], p[j] = p[j], p[i]
            population.append(p)
        print("→ Using NEH-based initialization")
    else:
        population = [random_permutation(num_jobs) for _ in range(pop_size)]
        print("→ Using random initialization")

    fitnesses = [makespan(ind, times) for ind in population]

    best_fitness = min(fitnesses)
    best_history = []
    stagnation_counter = 0

    # Precompute simple bounds
    lb_machine, lb_job, lb_simple, ub_trivial = compute_bounds(times)
    print(f"→ Lower bound (machine): {lb_machine}")
    print(f"→ Lower bound (job):     {lb_job}")
    print(f"→ Simple LB used:        {lb_simple}")
    print(f"→ Trivial UB:            {ub_trivial}\n")

    current_mut_rate = mut_rate

    for gen in range(generations):
        new_pop = []
        ranked_idx = sorted(range(pop_size), key=lambda i: fitnesses[i])

        # Elitism
        for i in range(elitism):
            new_pop.append(population[ranked_idx[i]][:])

        # Reproduction
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses, k=tournament_k)
            p2 = tournament_selection(population, fitnesses, k=tournament_k)
            if random.random() < cx_prob:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            swap_mutation(c1, current_mut_rate)
            swap_mutation(c2, current_mut_rate)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop
        fitnesses = [makespan(ind, times) for ind in population]
        current_best = min(fitnesses)
        best_history.append(current_best)

        # Check improvement
        if current_best < best_fitness:
            best_fitness = current_best
            stagnation_counter = 0
            current_mut_rate = mut_rate  # reset mutation
        else:
            stagnation_counter += 1

        # Increase mutation rate if stagnation passes threshold
        if stagnation_counter == mut_increase_threshold:
            current_mut_rate = min(0.9, current_mut_rate * 2)
            print(f"⚠️  Increased mutation rate to {current_mut_rate:.2f} after "
                  f"{mut_increase_threshold} stagnant generations")

        # Progress every 1% of total generations or first 10 gens
        if (gen + 1) % max(1, generations // 100) == 0 or gen < 10:
            gap = 100 * (current_best - lb_simple) / lb_simple
            div = population_diversity(population, sample_size=diversity_sample)
            print(f"Gen {gen+1:5d} | Best = {current_best:5d} | "
                  f"Gap vs LB = {gap:6.2f}% | "
                  f"Diversity = {div*100:5.1f}% | "
                  f"No improv: {stagnation_counter}")


        # Early stop if stagnated too long
        if stagnation_counter >= stagnation_limit:
            print(f"\n❌ Stagnation detected — no improvement for {stagnation_limit} generations.")
            print(f"Stopping early at generation {gen+1}.")
            break

    best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
    return population[best_idx], fitnesses[best_idx], best_history, lb_simple


# ------------------ Utilities ------------------
def population_diversity(population, sample_size=10):
    """Estimate population diversity by average Hamming distance over a sample."""
    n = len(population)
    if n < 2:
        return 0
    sample = random.sample(population, min(sample_size, n))
    dists = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            # Hamming distance: number of differing positions
            d = sum(a != b for a, b in zip(sample[i], sample[j]))
            dists.append(d / len(sample[0]))  # normalize
    return statistics.mean(dists) if dists else 0.0
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


# ------------------ Entry point ------------------

if __name__ == "__main__":
    input_path = "data6.txt"

    # Read data
    times = read_input_from_file(input_path)

    # Run GA (toggle init_mode and thresholds here)
    best_perm, best_val, history, lb = genetic_algorithm(
        times,
        pop_size=120,
        generations=5000,
        cx_prob=0.9,
        mut_rate=0.2,
        elitism=2,
        seed=None,
        stagnation_limit=2000,
        mut_increase_threshold=500,   # when to boost mutation
        init_mode="random",               # "random" or "neh"
        diversity_sample=10
    )

    # Show results
    print("\n=== Best solution found ===")
    print("Job order (1-based):", [j + 1 for j in best_perm])
    print(f"Makespan: {best_val}")
    print(f"Gap vs lower bound: {(best_val - lb)/lb*100:.2f}%")
    print_schedule(best_perm, times)
