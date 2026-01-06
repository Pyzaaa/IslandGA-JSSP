#include <iostream>
#include <vector>
#include <random>
#include "ga.h"
#include "gpu_interface.h"

int main() {
    int nJobs = 4;
    int nMachines = 3;

    std::vector<int> times = {
        1,3,8,
        9,3,5,
        7,8,6,
        4,8,7
    };

    int pop_size = 1024;
    std::vector<std::vector<int>> population(pop_size);
    std::vector<int> population_flat(pop_size * nJobs);
    std::vector<int> fitness(pop_size);

    std::mt19937 rng(42);

    for (int i = 0; i < pop_size; i++) {
        population[i].resize(nJobs);
        for (int j = 0; j < nJobs; j++)
            population[i][j] = j;
        std::shuffle(population[i].begin(), population[i].end(), rng);

        for (int j = 0; j < nJobs; j++)
            population_flat[i * nJobs + j] = population[i][j];
    }

    compute_fitness_gpu(
        population_flat.data(),
        times.data(),
        fitness.data(),
        pop_size,
        nJobs,
        nMachines
    );

    int best = *std::min_element(fitness.begin(), fitness.end());
    std::cout << "Best makespan = " << best << "\n";

    return 0;
}
