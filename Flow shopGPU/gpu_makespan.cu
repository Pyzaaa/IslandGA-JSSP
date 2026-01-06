#include <cuda_runtime.h>
#include "gpu_interface.h"

__global__
void makespan_kernel(
    const int* population,
    const int* times,
    int* fitness,
    int nJobs,
    int nMachines
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gridDim.x * blockDim.x) return;

    int offset = idx * nJobs;

    int machine_ready[64];  // max liczba maszyn

    for (int m = 0; m < nMachines; m++)
        machine_ready[m] = 0;

    for (int p = 0; p < nJobs; p++) {
        int job = population[offset + p];
        int prev = 0;
        for (int m = 0; m < nMachines; m++) {
            int start = max(machine_ready[m], prev);
            int comp  = start + times[job * nMachines + m];
            machine_ready[m] = comp;
            prev = comp;
        }
    }

    fitness[idx] = machine_ready[nMachines - 1];
}

void compute_fitness_gpu(
    const int* population,
    const int* times,
    int* fitness,
    int pop_size,
    int nJobs,
    int nMachines
) {
    int *d_pop, *d_times, *d_fit;

    cudaMalloc(&d_pop, pop_size * nJobs * sizeof(int));
    cudaMalloc(&d_times, nJobs * nMachines * sizeof(int));
    cudaMalloc(&d_fit, pop_size * sizeof(int));

    cudaMemcpy(d_pop, population,
               pop_size * nJobs * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_times, times,
               nJobs * nMachines * sizeof(int),
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (pop_size + threads - 1) / threads;

    makespan_kernel<<<blocks, threads>>>(
        d_pop, d_times, d_fit, nJobs, nMachines
    );

    cudaMemcpy(fitness, d_fit,
               pop_size * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(d_pop);
    cudaFree(d_times);
    cudaFree(d_fit);
}
