#include "utils.h"
#include "ga_helpers.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <string>

#include <random>
std::mt19937 rng(std::random_device{}());

// Forward declaration GPU
int makespan_gpu(const std::vector<int>& order, const std::vector<std::vector<int>>& times);

int main(int argc, char* argv[]) {
    std::string input_file = "data.txt";
    std::string init_mode = "random";

    if(argc > 1) input_file = argv[1];
    if(argc > 2) init_mode = argv[2];

    auto times = read_input(input_file);
    int n = times.size();

    int pop_size = 120;
    int generations = 5000;
    double cx_prob = 0.9;
    double mut_rate = 0.2;
    int elitism = 2;
    int stagnation_limit = 2000;
    int mut_boost = 500;

    std::vector<std::vector<int>> population;
    if(init_mode == "neh") {
        std::cout << "→ Using NEH initialization\n";
        auto neh = neh_sequence(times);
        population.push_back(neh);
        for(int i = 1; i < pop_size; i++) {
            auto p = neh;
            std::swap(p[rand_int(0,n-1)], p[rand_int(0,n-1)]);
            population.push_back(p);
        }
    } else {
        std::cout << "→ Using RANDOM initialization\n";
        for(int i=0;i<pop_size;i++)
            population.push_back(random_permutation(n));
    }

    std::vector<int> fitness(pop_size);
    for(int i=0;i<pop_size;i++)
        fitness[i] = makespan_gpu(population[i], times);

    int lbm, lbj, lb, ub;
    std::tie(lbm, lbj, lb, ub) = compute_bounds(times);

    int best = *std::min_element(fitness.begin(), fitness.end());
    int stagnation = 0;
    double cur_mut = mut_rate;

    for(int gen=0; gen<generations; gen++) {
        std::vector<std::vector<int>> new_pop;

        // elitism
        std::vector<int> idx(pop_size);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a,int b){ return fitness[a]<fitness[b]; });
        for(int i=0;i<elitism;i++) new_pop.push_back(population[idx[i]]);

        while(new_pop.size()<pop_size) {
            auto p1 = tournament_selection(population, fitness, 3);
            auto p2 = tournament_selection(population, fitness, 3);
            if(rand_double()<cx_prob) {
                auto children = order_crossover(p1,p2);
                auto c1 = children.first;
                auto c2 = children.second;
                swap_mutation(c1, cur_mut);
                swap_mutation(c2, cur_mut);
                new_pop.push_back(c1);
                if(new_pop.size()<pop_size) new_pop.push_back(c2);
            } else new_pop.push_back(p1);
        }

        population = new_pop;
        for(int i=0;i<pop_size;i++)
            fitness[i] = makespan_gpu(population[i], times);

        int cur_best = *std::min_element(fitness.begin(), fitness.end());
        if(cur_best < best) { best = cur_best; stagnation = 0; cur_mut = mut_rate; }
        else stagnation++;

        if(stagnation == mut_boost) cur_mut = std::min(0.9, cur_mut*2);

        if(stagnation >= stagnation_limit) break;

        // progress
        if(gen<10 || (gen+1)%(generations/100)==0) {
            double gap = 100.0*(best-lb)/lb;
            double div = population_diversity(population, 10);
            std::cout << "Gen " << gen+1
                      << " | Best = " << best
                      << " | Gap = " << std::fixed << std::setprecision(2) << gap << "%"
                      << " | Div = " << div*100.0 << "%"
                      << " | NoImprove = " << stagnation
                      << "\n";
        }
    }

    int best_idx = std::min_element(fitness.begin(), fitness.end()) - fitness.begin();
    std::cout << "\nBEST ORDER:\n";
    for(int j : population[best_idx]) std::cout << j+1 << " ";
    std::cout << "\nCmax = " << fitness[best_idx] << "\n";
    print_schedule(population[best_idx], times);
}
