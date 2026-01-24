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
    int* mr_all,
    int n, int m, int pop_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    // Każdy wątek korzysta z własnego kawałka globalnej pamięci
    int* mr = mr_all + idx * m;

    // zerujemy completion times dla maszyn
    for (int k = 0; k < m; k++)
        mr[k] = 0;

    const int* perm = pop + idx * n;

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





/* ===================== NEH ===================== */
vector<int> neh_sequence(const vector<vector<int>>& times) {
    int n = times.size();
    int m = times[0].size();
    vector<int> jobs(n);
    iota(jobs.begin(), jobs.end(), 0);
    vector<int> machine_ready(m);

    sort(jobs.begin(), jobs.end(), [&](int a, int b) {
        return accumulate(times[a].begin(), times[a].end(), 0) >
               accumulate(times[b].begin(), times[b].end(), 0);
    });

    vector<int> seq = { jobs[0] };
    for (int i = 1; i < n; i++) {
        int job = jobs[i];
        vector<int> best_seq;
        int best_val = INT_MAX;

        for (int pos = 0; pos <= seq.size(); pos++) {
            vector<int> trial = seq;
            trial.insert(trial.begin()+pos, job);

            int last = 0;
            vector<int> mr(m,0);
            for (int jj : trial) {
                int prev = 0;
                for (int k = 0; k < m; k++) {
                    int start = max(mr[k], prev);
                    int completion = start + times[jj][k];
                    mr[k] = completion;
                    prev = completion;
                }
                last = prev;
            }

            if (last < best_val) {
                best_val = last;
                best_seq = trial;
            }
        }
        seq = best_seq;
    }
    return seq;
}

/* ===================== CROSSOVER + MUT ===================== */
__global__ void crossover_mutation(
    const int* parents,
    int* offspring,
    const int* indices,      // Dodane: posortowane indeksy
    curandState* rng, 
    unsigned char* d_used_all,
    int n, int pop_size,
    double mut_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local_rng = rng[idx];

    // ELITARYZM: Najlepszy osobnik (indices[0]) przechodzi bez zmian do offspring[0]
    if (idx == 0) {
        int best_idx = indices[0];
        for (int i = 0; i < n; i++) {
            offspring[i] = parents[best_idx * n + i];
        }
        return;
    }

    // Wybór rodziców: p1 to jeden z 20% najlepszych, p2 to ktokolwiek
    int p1_idx = indices[curand(&local_rng) % (pop_size / 5)]; 
    int p2_idx = indices[curand(&local_rng) % pop_size];

    const int* A = parents + p1_idx * n;
    const int* B = parents + p2_idx * n;
    int* C = offspring + idx * n;

    unsigned char* used = d_used_all + idx * n;
    for (int i = 0; i < n; i++) used[i] = 0;

    // Operator OX (Order Crossover)
    int a = curand(&local_rng) % n;
    int b = curand(&local_rng) % n;
    if (a > b) { int t = a; a = b; b = t; }

    for (int i = a; i <= b; i++) {
        C[i] = A[i];
        used[A[i]] = 1;
    }

    int pos = (b + 1) % n;
    for (int i = 0; i < n; i++) {
        int job = B[(b + 1 + i) % n];
        if (!used[job]) {
            C[pos] = job;
            used[job] = 1;
            pos = (pos + 1) % n;
        }
    }

    // Mutacja typu Swap
    if (curand_uniform(&local_rng) < mut_rate) {
        int i = curand(&local_rng) % n;
        int j = curand(&local_rng) % n;
        int t = C[i]; C[i] = C[j]; C[j] = t;
    }

    rng[idx] = local_rng;
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
        vector<int> neh = neh_sequence(times);
        population.push_back(neh);

        mt19937 gen(123);
        for (int i = 1; i < pop_size; i++) {
            vector<int> p = neh;
            // lekka perturbacja NEH
            int a = gen() % n;
            int b = gen() % n;
            swap(p[a], p[b]);
            population.push_back(p);
        }
    } else {
        for (int i = 0; i < pop_size; i++)
            population.push_back(random_permutation(n));
    }


    /* ===================== D times ===================== */
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


    int* d_mr_all;
    cudaMalloc(&d_mr_all, pop_size * m * sizeof(int));       
    unsigned char* d_used_all;
    cudaMalloc(&d_used_all, pop_size * n);


    auto t0 = chrono::high_resolution_clock::now();

    // Przed pętlą zaalokuj tablicę indeksów
    int* d_indices;
    cudaMalloc(&d_indices, sizeof(int) * pop_size);
    thrust::device_ptr<int> dev_ptr_indices(d_indices);
    thrust::device_ptr<int> dev_ptr_fit(d_fit);

    for (int gen = 0; gen < generations; gen++) {
        // 1. Oblicz fitness
        makespan_kernel<<<blocks, threads>>>(d_popA, d_times, d_fit, d_mr_all, n, m, pop_size);

        // 2. Przygotuj indeksy: [0, 1, 2, ..., pop_size-1]
        thrust::sequence(dev_ptr_indices, dev_ptr_indices + pop_size);

        // 3. Posortuj indeksy według fitnessu
        thrust::sort_by_key(dev_ptr_fit, dev_ptr_fit + pop_size, dev_ptr_indices);

        // 4. Krzyżowanie i mutacja 
        // Przekazujemy d_indices, najlepszego ind
        crossover_mutation<<<blocks, threads>>>(
            d_popA, d_popB, d_indices, d_rng, d_used_all, n, pop_size, mut_rate
        );

        // 5. Swap buforów populacji
        swap(d_popA, d_popB);

        if (verbose && (gen % 100 == 0)) {
            int best_val;
            cudaMemcpy(&best_val, d_fit, sizeof(int), cudaMemcpyDeviceToHost);
            cout << "Gen " << gen << " | Best Cmax = " << best_val << endl;
        }
    }
    // Final evaluation
    makespan_kernel<<<blocks, threads>>>(    d_popA,    d_times,    d_fit, d_mr_all,    n, m, pop_size);
    cudaDeviceSynchronize();

    // Copy best fitness
    int best_cmax;
    cudaMemcpy(&best_cmax, d_fit, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy best permutation
    vector<int> best_perm(n);
    cudaMemcpy(
        best_perm.data(),
        d_popA,
        sizeof(int) * n,
        cudaMemcpyDeviceToHost
    );


    cudaDeviceSynchronize();
    auto t1 = chrono::high_resolution_clock::now();

    cout << "\n===== FINAL RESULT =====\n";
    cout << "Best Cmax: " << best_cmax << "\n";
    cout << "Best permutation:\n";
    for (int j : best_perm) cout << j+1 << " ";
    cout << "\n";


    cout << "Execution time: "
         << chrono::duration<double>(t1 - t0).count()
         << " s\n";

    cudaFree(d_popA);
    cudaFree(d_popB);
    cudaFree(d_fit);
    cudaFree(d_rng);
    cudaFree(d_times);
    cudaFree(d_indices);
    cudaFree(d_used_all);

}
