#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <random>
#include <climits>

#include "utils.h"

using namespace std;

/* ===================== RNG ===================== */
__global__ void init_rng(curandState* states, unsigned seed, int pop) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pop)
        curand_init(seed, i, 0, &states[i]);
}

/* ===================== MAKESPAN ===================== */
__global__ void makespan_kernel(
    const int* pop,
    const int* d_times,
    int* fitness,
    int n, int m, int pop_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    const int* perm = pop + idx * n;
    int mr[32] = {0};

    for (int j = 0; j < n; j++) {
        int job = perm[j];
        int prev = 0;
        for (int k = 0; k < m; k++) {
            int start = max(mr[k], prev);
            int c = start + d_times[job * m + k];
            mr[k] = c;
            prev = c;
        }
    }
    fitness[idx] = mr[m - 1];
}


/* ===================== CROSSOVER + MUT ===================== */
__global__ void crossover_mutation(
    const int* parents,
    int* offspring,
    curandState* rng,
    int n, int pop_size,
    double mut_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local = rng[idx];

    int p1 = idx;
    int p2 = curand(&local) % pop_size;

    const int* A = parents + p1 * n;
    const int* B = parents + p2 * n;
    int* C = offspring + idx * n;

    int a = curand(&local) % n;
    int b = curand(&local) % n;
    if (a > b) { int t = a; a = b; b = t; }

    bool used[256] = {0};

    for (int i = a; i <= b; i++) {
        C[i] = A[i];
        used[A[i]] = true;
    }

    int pos = (b + 1) % n;
    for (int i = 0; i < n; i++) {
        int job = B[(b + 1 + i) % n];
        if (!used[job]) {
            C[pos] = job;
            used[job] = true;
            pos = (pos + 1) % n;
        }
    }

    if (curand_uniform(&local) < mut_rate) {
        int i = curand(&local) % n;
        int j = curand(&local) % n;
        int t = C[i]; C[i] = C[j]; C[j] = t;
    }

    rng[idx] = local;
}

/* ===================== MAIN ===================== */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.txt [options]\n";
        return 1;
    }

    string input_file = argv[1];
    string init_mode = "neh";
    bool verbose = false;
    int pop_size = 120, generations = 5000;
    double cx_prob = 0.9, mut_rate = 0.2;

    for (int i = 2; i < argc; i++) {
        string a = argv[i];
        if (a == "--pop") pop_size = stoi(argv[++i]);
        else if (a == "--gen") generations = stoi(argv[++i]);
        else if (a == "--cx") cx_prob = stod(argv[++i]); // (nieużywane – GA GPU ma implicit cx)
        else if (a == "--mut") mut_rate = stod(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else if (a == "--init") init_mode = argv[++i];
    }

    auto times = read_input(input_file);
    int n = times.size(), m = times[0].size();


    /* ===================== POP INIT (CPU) ===================== */
    vector<vector<int>> population;
    if (init_mode == "neh") {
        for (int i = 0; i < pop_size; i++)
            population.push_back(random_permutation(n));
    } else {
        for (int i = 0; i < pop_size; i++)
            population.push_back(random_permutation(n));
    }

    vector<int> h_pop(pop_size * n);
    for (int i = 0; i < pop_size; i++)
        copy(population[i].begin(), population[i].end(), h_pop.begin() + i * n);

    vector<int> h_times = flatten_times(times);
    int* d_times;
    cudaMalloc(&d_times, sizeof(int) * n * m);
    cudaMemcpy(
        d_times,
        h_times.data(),
        sizeof(int) * n * m,
        cudaMemcpyHostToDevice
);

    /* ===================== GPU MEM ===================== */
    int *d_popA, *d_popB, *d_fit;
    curandState* d_rng;

    cudaMalloc(&d_popA, sizeof(int) * pop_size * n);
    cudaMalloc(&d_popB, sizeof(int) * pop_size * n);
    cudaMalloc(&d_fit, sizeof(int) * pop_size);
    cudaMalloc(&d_rng, sizeof(curandState) * pop_size);

    cudaMemcpy(d_popA, h_pop.data(), sizeof(int) * pop_size * n, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (pop_size + threads - 1) / threads;

    init_rng<<<blocks, threads>>>(d_rng, 1234, pop_size);

    auto t0 = chrono::high_resolution_clock::now();

    for (int gen = 0; gen < generations; gen++) {
        makespan_kernel<<<blocks, threads>>>(    d_popA,    d_times,    d_fit,    n, m, pop_size);


        thrust::sort_by_key(
            thrust::device,
            d_fit, d_fit + pop_size,
            d_popA
        );

        crossover_mutation<<<blocks, threads>>>(
            d_popA, d_popB, d_rng, n, pop_size, mut_rate
        );

        swap(d_popA, d_popB);

        if (verbose && (gen < 10 || (gen + 1) % (generations / 100) == 0)) {
            cout << "Gen " << gen + 1
                 << " | Best Cmax = [GPU]\n";
        }
    }

    cudaDeviceSynchronize();
    auto t1 = chrono::high_resolution_clock::now();

    cout << "Execution time: "
         << chrono::duration<double>(t1 - t0).count()
         << " s\n";

    cudaFree(d_popA);
    cudaFree(d_popB);
    cudaFree(d_fit);
    cudaFree(d_rng);
    cudaFree(d_times);

}
