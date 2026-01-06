#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <limits>

// ===================== GA OPERATORS =====================
inline std::vector<int> random_permutation(int n) {
    std::vector<int> p(n);
    std::iota(p.begin(), p.end(), 0);
    std::shuffle(p.begin(), p.end(), rng);
    return p;
}

inline std::vector<int> tournament_selection(
    const std::vector<std::vector<int>>& pop,
    const std::vector<int>& fitness,
    int k
) {
    int best = rand_int(0, pop.size()-1);
    for(int i=1;i<k;i++){
        int cand = rand_int(0, pop.size()-1);
        if(fitness[cand] < fitness[best]) best = cand;
    }
    return pop[best];
}

inline std::pair<std::vector<int>, std::vector<int>> order_crossover(
    const std::vector<int>& p1,
    const std::vector<int>& p2
){
    int n = p1.size();
    int a = rand_int(0,n-1);
    int b = rand_int(0,n-1);
    if(a>b) std::swap(a,b);

    auto ox = [&](const std::vector<int>& A, const std::vector<int>& B){
        std::vector<int> child(n,-1);
        for(int i=a;i<=b;i++) child[i] = A[i];
        int pos = (b+1)%n;
        for(int i=0;i<n;i++){
            int job = B[(b+1+i)%n];
            if(std::find(child.begin(), child.end(), job) == child.end()){
                child[pos] = job;
                pos = (pos+1)%n;
            }
        }
        return child;
    };

    return {ox(p1,p2), ox(p2,p1)};
}

inline void swap_mutation(std::vector<int>& ind, double rate){
    if(rand_double() < rate){
        int i = rand_int(0, ind.size()-1);
        int j = rand_int(0, ind.size()-1);
        std::swap(ind[i], ind[j]);
    }
}

// ===================== NEH HEURISTIC =====================
inline std::vector<int> neh_sequence(const std::vector<std::vector<int>>& times){
    int n = times.size();
    std::vector<int> jobs(n);
    std::iota(jobs.begin(), jobs.end(), 0);

    std::sort(jobs.begin(), jobs.end(), [&](int a, int b){
        return std::accumulate(times[a].begin(), times[a].end(), 0) >
               std::accumulate(times[b].begin(), times[b].end(), 0);
    });

    std::vector<int> seq = {jobs[0]};
    for(int i=1;i<n;i++){
        int job = jobs[i];
        std::vector<int> best_seq;
        int best_val = std::numeric_limits<int>::max();
        for(int pos=0; pos<=seq.size(); pos++){
            std::vector<int> trial = seq;
            trial.insert(trial.begin()+pos, job);
            // Tutaj wywołaj funkcję makespan (CPU lub GPU)
            // int val = makespan(trial, times);
            // Zostaw placeholder:
            int val = 0; 
            if(val<best_val){ best_val=val; best_seq=trial; }
        }
        seq = best_seq;
    }
    return seq;
}

// ===================== BOUNDS =====================
inline std::tuple<int,int,int,int> compute_bounds(const std::vector<std::vector<int>>& times){
    int n = times.size();
    int m = times[0].size();
    int lb_machine = 0;
    for(int k=0;k<m;k++){
        int sum = 0;
        for(int j=0;j<n;j++) sum += times[j][k];
        lb_machine = std::max(lb_machine, sum);
    }

    int lb_job = 0;
    for(int j=0;j<n;j++){
        int sum = std::accumulate(times[j].begin(), times[j].end(), 0);
        lb_job = std::max(lb_job, sum);
    }

    int ub = 0;
    for(auto& r : times) ub += std::accumulate(r.begin(), r.end(), 0);

    return {lb_machine, lb_job, std::max(lb_machine, lb_job), ub};
}

// ===================== POPULATION DIVERSITY =====================
inline double population_diversity(
    const std::vector<std::vector<int>>& pop,
    int sample_size
){
    int n = pop.size();
    if(n<2) return 0.0;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    int s = std::min(sample_size, n);
    double total = 0.0;
    int cnt = 0;
    for(int i=0;i<s;i++){
        for(int j=i+1;j<s;j++){
            int d = 0;
            for(int k=0;k<pop[0].size();k++)
                if(pop[idx[i]][k] != pop[idx[j]][k]) d++;
            total += double(d)/pop[0].size();
            cnt++;
        }
    }
    return cnt ? total/cnt : 0.0;
}
