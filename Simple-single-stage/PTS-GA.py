import json
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional


# -------------------------
# Helper functions
# -------------------------
def evaluate_assignment(assignment: List[int], processing_times: List[int], m: int) -> Tuple[float, List[float]]:
    loads = [0.0] * m
    for j, machine in enumerate(assignment):
        loads[machine] += processing_times[j]
    makespan = max(loads)
    return makespan, loads


def lpt_seed(processing_times: List[int], m: int) -> List[int]:
    """Longest Processing Time first heuristic (for seeding GA)."""
    n = len(processing_times)
    assignment = [-1] * n
    loads = [0.0] * m
    jobs_sorted = sorted(range(n), key=lambda j: processing_times[j], reverse=True)
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
class GeneticScheduler:
    def __init__(self, processing_times: List[int], m: int, params: GAParams):
        self.p = processing_times
        self.n = len(processing_times)
        self.m = m
        self.params = params
        if params.random_seed is not None:
            random.seed(params.random_seed)

    def initial_population(self) -> List[List[int]]:
        pop = []
        if self.params.seed_with_lpt:
            pop.append(lpt_seed(self.p, self.m))
        while len(pop) < self.params.pop_size:
            pop.append([random.randrange(self.m) for _ in range(self.n)])
        return pop

    def fitness(self, assignment: List[int]) -> float:
        makespan, _ = evaluate_assignment(assignment, self.p, self.m)
        return makespan

    def tournament_select(self, population: List[List[int]], fitnesses: List[float]) -> List[int]:
        k = self.params.tournament_k
        best, best_f = None, float('inf')
        for _ in range(k):
            i = random.randrange(len(population))
            if fitnesses[i] < best_f:
                best, best_f = population[i], fitnesses[i]
        return best.copy()

    def crossover(self, parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
        c1, c2 = parent1.copy(), parent2.copy()
        for j in range(self.n):
            if random.random() < 0.5:
                c1[j], c2[j] = c2[j], c1[j]
        return c1, c2

    def mutate(self, assignment: List[int]):
        for j in range(self.n):
            if random.random() < self.params.mutation_rate:
                current = assignment[j]
                new_m = random.randrange(self.m - 1)
                if new_m >= current:
                    new_m += 1
                assignment[j] = new_m

    def run(self) -> dict:
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
        makespan, loads = evaluate_assignment(best, self.p, self.m)
        return {
            "best_assignment": best,
            "best_makespan": makespan,
            "loads": loads,
            "time_seconds": total_time
        }


# -------------------------
# Utility Functions
# -------------------------
def lower_bound(processing_times: List[int], m: int) -> float:
    return max(max(processing_times), sum(processing_times) / m)


def gap_to_lb(makespan: float, lb: float) -> float:
    return (makespan - lb) / lb * 100 if lb > 0 else 0


def print_result(result, inst):
    lb = lower_bound(inst["processing_times"], inst["machines"])
    print(f"\nInstance {inst['id']}: n={inst['jobs']}, m={inst['machines']}")
    print(f"Lower bound: {lb:.2f}")
    print(f"Best makespan: {result['best_makespan']:.2f}")
    print(f"Gap to LB: {gap_to_lb(result['best_makespan'], lb):.2f}%")
    print(f"Runtime: {result['time_seconds']:.2f}s")
    print(f"Machine loads: {result['loads']}")


# -------------------------
# Run GA directly (no CLI)
# -------------------------
def run_ga_solver(dataset_path="dataset.json"):
    """Load dataset and run GA solver for each instance."""
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    params = GAParams(
        pop_size=120,
        generations=1000,
        mutation_rate=0.02,
        elitism=4,
        seed_with_lpt=True,
        stagnation_limit=150,
        random_seed=42
    )

    results = []
    for inst in dataset:
        print(f"Solving {inst['id']} ...")
        scheduler = GeneticScheduler(inst["processing_times"], inst["machines"], params)
        result = scheduler.run()
        print_result(result, inst)
        results.append({"instance": inst["id"], **result})

    return results


# Example usage:
if __name__ == "__main__":
    # Just call run_ga_solver() with the dataset path
    results = run_ga_solver("p_parallel_dataset.json")
