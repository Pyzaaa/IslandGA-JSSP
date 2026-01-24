#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <random>

#include <iostream>
#include <vector>
#include <numeric>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstring>

using namespace std;

/* ===================== CONSTANT ===================== */
__constant__ int d_times[4096]; // max n*m

/* ===================== RNG INIT ===================== */
__global__ void init_rng(curandState *states, unsigned seed, int pop_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pop_size)
        curand_init(seed, i, 0, &states[i]);
}

/* ===================== MAKESPAN ===================== */
__global__ void makespan_kernel(
    const int* population,
    int* fitness,
    int n, int m, int pop_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    const int* perm = population + idx * n;
    int mr[32] = {0};

    for (int j = 0; j < n; j++) {
        int job = perm[j];
        int prev = 0;
        for (int k = 0; k < m; k++) {
            int start = max(mr[k], prev);
            int t = d_times[job * m + k];
            int c = start + t;
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
    curandState* states,
    int n, int pop_size,
    double mut_rate
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    curandState local = states[idx];

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
        int t = C[i];
        C[i] = C[j];
        C[j] = t;
    }

    states[idx] = local;
}

/* ===================== DATA LOAD ===================== */
bool load_flowshop(
    const string& path,
    int& n, int& m,
    vector<int>& times
) {
    ifstream f(path);
    if (!f) return false;

    f >> n >> m;
    times.resize(n * m);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            f >> times[i * m + j];

    return true;
}

/* ===================== ARGS ===================== */
void usage(const char* p) {
    cout << "Usage:\n"
         << p << " --data file.txt [options]\n"
         << "  --pop N\n"
         << "  --gen N\n"
         << "  --mut R\n"
         << "  --seed S\n"
         << "  --chrono\n"
         << "  --verbose\n";
}

/* ===================== MAIN ===================== */
int main(int argc, char** argv) {

    string data_file;
    int pop_size = 256;
    int generations = 1000;
    double mut_rate = 0.2;
    unsigned seed = time(nullptr);
    bool chrono_on = false;
    bool verbose = false;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--data")) data_file = argv[++i];
        else if (!strcmp(argv[i], "--pop")) pop_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen")) generations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mut")) mut_rate = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed")) seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--chrono")) chrono_on = true;
        else if (!strcmp(argv[i], "--verbose")) verbose = true;
        else { usage(argv[0]); return 1; }
    }

    if (data_file.empty()) {
        usage(argv[0]);
        return 1;
    }

    int n, m;
    vector<int> h_times;
    if (!load_flowshop(data_file, n, m, h_times)) {
        cerr << "Cannot load " << data_file << "\n";
        return 1;
    }

    cudaMemcpyToSymbol(d_times, h_times.data(), sizeof(int) * n * m);

    vector<int> h_pop(pop_size * n);

    for (int i = 0; i < pop_size; i++) {
        iota(h_pop.begin() + i * n, h_pop.begin() + (i + 1) * n, 0);
        
        static std::mt19937 rng(seed + i);
        std::shuffle(
            h_pop.begin() + i * n,
            h_pop.begin() + (i + 1) * n,
            rng
        );

    }

    int *d_popA, *d_popB, *d_fitness;
    curandState* d_rng;

    cudaMalloc(&d_popA, sizeof(int) * pop_size * n);
    cudaMalloc(&d_popB, sizeof(int) * pop_size * n);
    cudaMalloc(&d_fitness, sizeof(int) * pop_size);
    cudaMalloc(&d_rng, sizeof(curandState) * pop_size);

    cudaMemcpy(d_popA, h_pop.data(), sizeof(int) * pop_size * n, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (pop_size + threads - 1) / threads;

    init_rng<<<blocks, threads>>>(d_rng, seed, pop_size);

    auto t0 = chrono::high_resolution_clock::now();

    for (int g = 0; g < generations; g++) {
        makespan_kernel<<<blocks, threads>>>(d_popA, d_fitness, n, m, pop_size);

        thrust::sort_by_key(
            thrust::device,
            d_fitness, d_fitness + pop_size,
            d_popA
        );

        crossover_mutation<<<blocks, threads>>>(
            d_popA, d_popB, d_rng, n, pop_size, mut_rate
        );

        swap(d_popA, d_popB);

        if (verbose && g % 100 == 0)
            cout << "Gen " << g << " done\n";
    }

    cudaDeviceSynchronize();

    auto t1 = chrono::high_resolution_clock::now();
    if (chrono_on)
        cout << "Time: "
             << chrono::duration<double>(t1 - t0).count()
             << " s\n";

    cudaFree(d_popA);
    cudaFree(d_popB);
    cudaFree(d_fitness);
    cudaFree(d_rng);
}
