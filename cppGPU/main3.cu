#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <limits>
#include <string>
#include <iomanip>
#include <chrono>
#include "utils.h"

using namespace std;

/* ===================== MAKESPAN GPU ===================== */
__global__ void gpu_makespan(
    const int* d_population,
    const int* d_times,
    int* d_fitness,
    int n, int m, int pop_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    const int* order = d_population + idx * n;

    // zakładamy m <= 32
    int mr[32];
    #pragma unroll
    for (int k = 0; k < m; k++) mr[k] = 0;

    for (int j = 0; j < n; j++) {
        int job = order[j];
        int prev = 0;

        #pragma unroll
        for (int k = 0; k < m; k++) {
            int start = max(mr[k], prev);
            int completion = start + d_times[job * m + k];
            mr[k] = completion;
            prev = completion;
        }
    }
    d_fitness[idx] = mr[m - 1];
}

/* ===================== GA OPERATORS ===================== */
pair<vector<int>, vector<int>> order_crossover(
    const vector<int>& p1, const vector<int>& p2
) {
    int n = p1.size();
    int a = rand_int(0, n - 1);
    int b = rand_int(0, n - 1);
    if (a > b) swap(a, b);

    auto ox = [&](const vector<int>& A, const vector<int>& B) {
        vector<int> child(n, -1);
        vector<char> used(n, 0);

        for (int i = a; i <= b; i++) {
            child[i] = A[i];
            used[A[i]] = 1;
        }
        int pos = (b + 1) % n;
        for (int i = 0; i < n; i++) {
            int job = B[(b + 1 + i) % n];
            if (!used[job]) {
                child[pos] = job;
                used[job] = 1;
                pos = (pos + 1) % n;
            }
        }
        return child;
    };

    return { ox(p1, p2), ox(p2, p1) };
}

void swap_mutation(vector<int>& ind, double rate) {
    if (rand_double() < rate) {
        int i = rand_int(0, ind.size() - 1);
        int j = rand_int(0, ind.size() - 1);
        swap(ind[i], ind[j]);
    }
}

vector<int> tournament_selection(
    const vector<vector<int>>& pop,
    const vector<int>& fitness,
    int k
) {
    int best = rand_int(0, pop.size() - 1);
    for (int i = 1; i < k; i++) {
        int cand = rand_int(0, pop.size() - 1);
        if (fitness[cand] < fitness[best]) best = cand;
    }
    return pop[best];
}

/* ===================== NEH ===================== */
vector<int> neh_sequence(const vector<vector<int>>& times) {
    int n = times.size();
    int m = times[0].size();
    vector<int> jobs(n);
    iota(jobs.begin(), jobs.end(), 0);

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
            trial.insert(trial.begin() + pos, job);

            vector<int> mr(m, 0);
            int last = 0;
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

/* ===================== BOUNDS ===================== */
tuple<int,int,int,int> compute_bounds(const vector<vector<int>>& times) {
    int n = times.size();
    int m = times[0].size();
    int lb_machine = 0, lb_job = 0;

    for (int k = 0; k < m; k++) {
        int sum = 0;
        for (int j = 0; j < n; j++) sum += times[j][k];
        lb_machine = max(lb_machine, sum);
    }
    for (int j = 0; j < n; j++) {
        int sum = accumulate(times[j].begin(), times[j].end(), 0);
        lb_job = max(lb_job, sum);
    }
    int ub = 0;
    for (auto& r : times)
        ub += accumulate(r.begin(), r.end(), 0);

    return { lb_machine, lb_job, max(lb_machine, lb_job), ub };
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
        else if (a == "--cx") cx_prob = stod(argv[++i]);
        else if (a == "--mut") mut_rate = stod(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else if (a == "--init") init_mode = argv[++i];
    }

    auto times = read_input(input_file);
    int n = times.size(), m = times[0].size();
    int lbm, lbj, lb, ub;
    tie(lbm, lbj, lb, ub) = compute_bounds(times);

    /* ===================== POPULATION ===================== */
    vector<vector<int>> population;
    if (init_mode == "neh") {
        auto neh = neh_sequence(times);
        population.push_back(neh);
        for (int i = 1; i < pop_size; i++) {
            auto p = neh;
            swap(p[rand_int(0, n - 1)], p[rand_int(0, n - 1)]);
            population.push_back(p);
        }
    } else {
        for (int i = 0; i < pop_size; i++)
            population.push_back(random_permutation(n));
    }

    vector<int> fitness(pop_size);

    /* ===================== GPU MEM ===================== */
    vector<int> h_population = flatten_population(population);
    vector<int> h_times = flatten_times(times);

    int *d_population, *d_times, *d_fitness;
    cudaMalloc(&d_population, sizeof(int) * pop_size * n);
    cudaMalloc(&d_times, sizeof(int) * n * m);
    cudaMalloc(&d_fitness, sizeof(int) * pop_size);

    cudaMemcpy(d_population, h_population.data(),
               sizeof(int) * pop_size * n,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_times, h_times.data(),
               sizeof(int) * n * m,
               cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (pop_size + threads - 1) / threads;

    auto start_time = chrono::high_resolution_clock::now();

    int best = INT_MAX;
    vector<int> best_perm;

    for (int gen = 0; gen < generations; gen++) {
        // 1️⃣ FITNESS NA GPU
        gpu_makespan<<<blocks, threads>>>(
            d_population, d_times, d_fitness, n, m, pop_size
        );

        cudaMemcpy(fitness.data(), d_fitness,
                   sizeof(int) * pop_size,
                   cudaMemcpyDeviceToHost);

        // 2️⃣ BEST
        int idx_best = min_element(fitness.begin(), fitness.end()) - fitness.begin();
        if (fitness[idx_best] < best) {
            best = fitness[idx_best];
            best_perm = population[idx_best];
        }

        if (verbose && (gen < 10 || (gen + 1) % (generations / 100) == 0)) {
            double gap = 100.0 * (best - lb) / lb;
            cout << "Gen " << gen + 1
                 << " | Best Cmax = " << best
                 << " | Gap = " << fixed << setprecision(2) << gap << "%\n";
        }

        // 3️⃣ NOWA POPULACJA NA CPU
        vector<vector<int>> new_pop;
        int elitism = 2;
        vector<int> idx(pop_size);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(),
             [&](int a, int b) { return fitness[a] < fitness[b]; });

        for (int i = 0; i < elitism; i++)
            new_pop.push_back(population[idx[i]]);

        while (new_pop.size() < pop_size) {
            auto p1 = tournament_selection(population, fitness, 3);
            auto p2 = tournament_selection(population, fitness, 3);
            auto children = order_crossover(p1, p2);
            auto c1 = children.first;
            auto c2 = children.second;
            swap_mutation(c1, mut_rate);
            swap_mutation(c2, mut_rate);
            new_pop.push_back(c1);
            if (new_pop.size() < pop_size) new_pop.push_back(c2);
        }

        population = new_pop;

        // 4️⃣ JEDYNY H2D W GENERACJI
        h_population = flatten_population(population);
        cudaMemcpy(d_population, h_population.data(),
                   sizeof(int) * pop_size * n,
                   cudaMemcpyHostToDevice);
    }

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "\nBEST ORDER:\n";
    for (int j : best_perm) cout << j + 1 << " ";
    cout << "\nCmax = " << best << "\n";
    cout << "Gap = " << (best - lb) * 100.0 / lb << "%\n";
    cout << "Execution time: " << fixed << setprecision(4)
         << elapsed.count() << " seconds\n";

    cudaFree(d_population);
    cudaFree(d_times);
    cudaFree(d_fitness);
}
