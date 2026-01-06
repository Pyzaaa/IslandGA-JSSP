#pragma once

void compute_fitness_gpu(
    const int* population,
    const int* times,
    int* fitness,
    int pop_size,
    int nJobs,
    int nMachines
);
