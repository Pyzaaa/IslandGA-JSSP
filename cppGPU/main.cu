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
    int* d_orders, 
    int* d_times, 
    int* d_fitness, 
    int n, int m, int pop_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    int* order = d_orders + idx * n;
    extern __shared__ int machine_ready_all[];
    int* mr = machine_ready_all + threadIdx.x * m;

    // Initialize machine ready times for this individual
    for (int k = 0; k < m; k++) mr[k] = 0;

    for (int j = 0; j < n; j++) {
        int job = order[j];
        int job_ready_on_next_machine = 0; // A job is ready for machine 0 at time 0
        
        for (int k = 0; k < m; k++) {
            // Start time is the max of (Machine k free, Job j finished on machine k-1)
            int start = max(mr[k], job_ready_on_next_machine);
            int completion = start + d_times[job * m + k];
            
            mr[k] = completion;             // Machine k will be free at this time
            job_ready_on_next_machine = completion; // Job j ready for machine k+1
        }
    }
    // The makespan is the completion time of the last job on the last machine
    d_fitness[idx] = mr[m - 1];
}

/* ===================== GA OPERATORS ===================== */
pair<vector<int>, vector<int>> order_crossover(const vector<int>& p1, const vector<int>& p2) {
    int n = p1.size();
    int a = rand_int(0, n-1);
    int b = rand_int(0, n-1);
    if (a > b) swap(a, b);

    auto ox = [&](const vector<int>& A, const vector<int>& B) {
        vector<int> child(n, -1);
        vector<char> used(n, 0);
        for (int i = a; i <= b; i++) {
            child[i] = A[i];
            used[A[i]] = 1;
        }
        int pos = (b+1) % n;
        for (int i = 0; i < n; i++) {
            int job = B[(b+1+i)%n];
            if (!used[job]) {
                child[pos] = job;
                used[job] = 1;
                pos = (pos+1)%n;
            }
        }
        return child;
    };

    return { ox(p1,p2), ox(p2,p1) };
}

void swap_mutation(vector<int>& ind, double rate) {
    if (rand_double() < rate) {
        int i = rand_int(0, ind.size()-1);
        int j = rand_int(0, ind.size()-1);
        swap(ind[i], ind[j]);
    }
}

vector<int> tournament_selection(const vector<vector<int>>& pop, const vector<int>& fitness, int k) {
    int best = rand_int(0, pop.size()-1);
    for (int i = 1; i < k; i++) {
        int cand = rand_int(0, pop.size()-1);
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

/* ===================== BOUNDS ===================== */
tuple<int,int,int,int> compute_bounds(const vector<vector<int>>& times) {
    int n = times.size();
    int m = times[0].size();
    int lb_machine=0, lb_job=0;

    for (int k=0;k<m;k++){
        int sum=0;
        for(int j=0;j<n;j++) sum+=times[j][k];
        lb_machine=max(lb_machine,sum);
    }
    for(int j=0;j<n;j++){
        int sum=accumulate(times[j].begin(),times[j].end(),0);
        lb_job=max(lb_job,sum);
    }
    int ub=0;
    for(auto& r:times) ub+=accumulate(r.begin(),r.end(),0);
    return {lb_machine, lb_job, max(lb_machine,lb_job), ub};
}

/* ===================== MAIN ===================== */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.txt [options]\n";
        return 1;
    }

    string input_file = argv[1];
    string init_mode="neh";
    bool verbose=false;
    int pop_size=120, generations=5000;
    double cx_prob=0.9, mut_rate=0.2;

    for(int i=2;i<argc;i++){
        string a=argv[i];
        if(a=="--pop") pop_size=stoi(argv[++i]);
        else if(a=="--gen") generations=stoi(argv[++i]);
        else if(a=="--cx") cx_prob=stod(argv[++i]);
        else if(a=="--mut") mut_rate=stod(argv[++i]);
        else if(a=="--verbose") verbose=true;
        else if(a=="--init") init_mode=argv[++i];
    }

    auto times = read_input(input_file);
    int n = times.size(), m = times[0].size();
    int lbm, lbj, lb, ub;
    tie(lbm,lbj,lb,ub)=compute_bounds(times);

    /* ===================== POPULATION ===================== */
    vector<vector<int>> population;
    if(init_mode=="neh") {
        auto neh=neh_sequence(times);
        population.push_back(neh);
        for(int i=1;i<pop_size;i++){
            auto p=neh;
            swap(p[rand_int(0,n-1)], p[rand_int(0,n-1)]);
            population.push_back(p);
        }
    } else {
        for(int i=0;i<pop_size;i++)
            population.push_back(random_permutation(n));
    }

    vector<int> machine_ready(m), fitness(pop_size);

    /* ===================== GPU MEM ===================== */
    vector<int> h_orders = flatten_population(population);
    vector<int> h_times = flatten_times(times);
    int *d_orders,*d_times,*d_fitness;
    cudaMalloc(&d_orders,sizeof(int)*pop_size*n);
    cudaMalloc(&d_times,sizeof(int)*n*m);
    cudaMalloc(&d_fitness,sizeof(int)*pop_size);
    cudaMemcpy(d_times,h_times.data(),sizeof(int)*n*m,cudaMemcpyHostToDevice);

    auto start_time=chrono::high_resolution_clock::now();

    int best = INT_MAX;
    vector<int> best_perm;
    double cur_mut=mut_rate;
    int threads=256, blocks=(pop_size+threads-1)/threads;
    size_t shared_mem = threads*m*sizeof(int);
    
    for(int gen=0; gen<generations; gen++){
        // 1️⃣ WYLICZENIE FITNESS GPU dla aktualnej populacji
        h_orders = flatten_population(population);
        cudaMemcpy(d_orders, h_orders.data(), sizeof(int)*pop_size*n, cudaMemcpyHostToDevice);
        gpu_makespan<<<blocks, threads, shared_mem>>>(d_orders, d_times, d_fitness, n, m, pop_size);
        cudaDeviceSynchronize();
        cudaMemcpy(fitness.data(), d_fitness, sizeof(int)*pop_size, cudaMemcpyDeviceToHost);

        // 2️⃣ ZNAJDŹ NAJLEPSZEGO
        int cur_best = *min_element(fitness.begin(), fitness.end());
        int idx_best = min_element(fitness.begin(), fitness.end()) - fitness.begin();
        if(cur_best < best){
            best = cur_best;
            best_perm = population[idx_best];
        }

        // 3️⃣ VERBOSE
        if(verbose && (gen<10 || (gen+1) % (generations/100)==0)){
            double gap = 100.0*(best-lb)/lb;
            cout << "Gen " << gen+1
                << " | Best Cmax = " << best
                << " | Gap = " << fixed << setprecision(2) << gap << "%"
                << "\n";
        }

        // 4️⃣ STWORZENIE NOWEJ POPULACJI (elitism, crossover, mutation)
        int elitism = 2;
        vector<vector<int>> new_pop;
        vector<int> idx(pop_size);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a,int b){ return fitness[a]<fitness[b]; });
        for(int i=0;i<elitism;i++) new_pop.push_back(population[idx[i]]);

        while(new_pop.size()<pop_size){
            auto p1 = tournament_selection(population, fitness, 3);
            auto p2 = tournament_selection(population, fitness, 3);

            pair<vector<int>, vector<int>> children = order_crossover(p1,p2);
            vector<int> c1 = children.first;
            vector<int> c2 = children.second;
            swap_mutation(c1, cur_mut);
            swap_mutation(c2, cur_mut);

            new_pop.push_back(c1);
            if(new_pop.size()<pop_size) new_pop.push_back(c2);
        }
        population = new_pop;
    }


    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "\nBEST ORDER:\n";
    for(int j: best_perm) cout << j+1 << " ";
    cout << "\nCmax = " << best << "\n";
    cout << "Gap = " << (best-lb)*100.0/lb << "%\n";
    cout << "Execution time: " << fixed << setprecision(4) << elapsed.count() << " seconds\n";

    cudaFree(d_orders);
    cudaFree(d_times);
    cudaFree(d_fitness);
}
