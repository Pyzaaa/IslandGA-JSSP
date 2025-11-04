import json
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional


# -------------------------
# Helper functions
# -------------------------
def schedule_with_release_deadline(assignment, processing_times, release_times, deadlines, m):
    """
    Compute completion times, total tardiness, and lateness statistics for a given assignment.
    """
    n = len(processing_times)
    machine_times = [0.0] * m
    completion_times = [0.0] * n
    tardiness_sum = 0.0
    late_jobs = 0

    # group jobs by machine
    machine_jobs = [[] for _ in range(m)]
    for j, mach in enumerate(assignment):
        machine_jobs[mach].append(j)

    # schedule jobs on each machine
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


def lpt_seed_with_release(processing_times, release_times, m):
    """LPT-style heuristic seed considering release times."""
    n = len(processing_times)
    assignment = [-1] * n
    loads = [0.0] * m
    jobs_sorted = sorted(range(n), key=lambda j: (release_times[j], -processing_times[j]))
    for j in jobs_sorted:
        k = min(range(m), key=lambda x: loads[x])
        assignment[j] = k
        loads[k] += processing_times[j]
    return assignment


@dataclass
class GAParams:
    pop_size: int = 100
    generations: int = 500
    crossover_rate: float = 0.9
    mutation_rate: float = 0.02
    tournament_k: int = 3
    elitism: int = 2
    seed_with_lpt: bool = True
    random_seed: Optional[int] = None
    stagnation_limit: Optional[int] = 100


# -------------------------
# Genetic Algorithm
# -------------------------
class GeneticSchedulerWithDeadlines:
    def __init__(self, processing_times, release_times, deadlines, m, params: GAParams):
        self.p = processing_times
        self.r = release_times
        self.d = deadlines
        self.n = len(processing_times)
        self.m = m
        self.params = params
        if params.random_seed is not None:
            random.seed(params.random_seed)

    def initial_population(self):
        pop = []
        if self.params.seed_with_lpt:
            pop.append(lpt_seed_with_release(self.p, self.r, self.m))
        while len(pop) < self.params.pop_size:
            pop.append([random.randrange(self.m) for _ in range(self.n)])
        return pop

    def fitness(self, assignment):
        # minimize total tardiness
        _, total_tardiness, _, _, _, _ = schedule_with_release_deadline(
            assignment, self.p, self.r, self.d, self.m
        )
        return total_tardiness

    def tournament_select(self, population, fitnesses):
        k = self.params.tournament_k
        best, best_f = None, float("inf")
        for _ in range(k):
            i = random.randrange(len(population))
            if fitnesses[i] < best_f:
                best, best_f = population[i], fitnesses[i]
        return best.copy()

    def crossover(self, parent1, parent2):
        c1, c2 = parent1.copy(), parent2.copy()
        for j in range(self.n):
            if random.random() < 0.5:
                c1[j], c2[j] = c2[j], c1[j]
        return c1, c2

    def mutate(self, assignment):
        for j in range(self.n):
            if random.random() < self.params.mutation_rate:
                current = assignment[j]
                new_m = random.randrange(self.m - 1)
                if new_m >= current:
                    new_m += 1
                assignment[j] = new_m

    def run(self):
        pop = self.initial_population()
        fitnesses = [self.fitness(ind) for ind in pop]
        best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
        best, best_f = pop[best_idx].copy(), fitnesses[best_idx]
        stagnation = 0
        start = time.time()

        for gen in range(self.params.generations):
            new_pop = []
            sorted_idx = sorted(range(len(pop)), key=lambda i: fitnesses[i])
            for i in sorted_idx[:self.params.elitism]:
                new_pop.append(pop[i].copy())

            while len(new_pop) < self.params.pop_size:
                p1 = self.tournament_select(pop, fitnesses)
                p2 = self.tournament_select(pop, fitnesses)
                if random.random() < self.params.crossover_rate:
                    c1, c2 = self.crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()
                self.mutate(c1)
                self.mutate(c2)
                new_pop.append(c1)
                if len(new_pop) < self.params.pop_size:
                    new_pop.append(c2)

            pop = new_pop
            fitnesses = [self.fitness(ind) for ind in pop]
            gen_best_idx = min(range(len(pop)), key=lambda i: fitnesses[i])
            gen_best_f = fitnesses[gen_best_idx]

            if gen_best_f < best_f:
                best_f = gen_best_f
                best = pop[gen_best_idx].copy()
                stagnation = 0
            else:
                stagnation += 1

            if self.params.stagnation_limit and stagnation >= self.params.stagnation_limit:
                break

        total_time = time.time() - start
        makespan, tardiness, avg_tardiness, late_frac, loads, completion = schedule_with_release_deadline(
            best, self.p, self.r, self.d, self.m
        )
        return {
            "best_assignment": best,
            "best_makespan": makespan,
            "total_tardiness": tardiness,
            "avg_tardiness": avg_tardiness,
            "late_fraction": late_frac,
            "late_percent": late_frac * 100,
            "machine_loads": loads,
            "completion_times": completion,
            "time_seconds": total_time,
        }


# -------------------------
# Run GA directly
# -------------------------
def run_ga_with_deadlines(dataset_path="p_parallel_deadlines_dataset.json"):
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    params = GAParams(
        pop_size=100,
        generations=10000,
        mutation_rate=0.03,
        elitism=4,
        seed_with_lpt=True,
        stagnation_limit=1000,
        random_seed=None,
    )

    results = []
    for inst in dataset:
        print(f"\nSolving {inst['id']} (n={inst['jobs']}, m={inst['machines']}) ...")
        scheduler = GeneticSchedulerWithDeadlines(
            inst["processing_times"],
            inst["release_times"],
            inst["deadlines"],
            inst["machines"],
            params,
        )
        result = scheduler.run()
        print(
            f"  Total tardiness: {result['total_tardiness']:.2f}\n"
            f"  Avg tardiness:   {result['avg_tardiness']:.2f}\n"
            f"  Late jobs:       {result['late_percent']:.1f}%\n"
            f"  Makespan:        {result['best_makespan']:.2f}\n"
            f"  Runtime:         {result['time_seconds']:.2f}s\n"
        )
        results.append({"instance": inst["id"], **result})
    return results


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    results = run_ga_with_deadlines("p_parallel_deadlines_dataset.json")
