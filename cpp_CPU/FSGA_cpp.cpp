#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <tuple>
#include <limits>
#include <string>
#include <iomanip>

using namespace std;

/* ===================== RANDOM ===================== */
mt19937 rng(random_device{}());

int rand_int(int a, int b) {
    uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

double rand_double() {
    uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

/* ===================== MAKESPAN ===================== */
int makespan(const vector<int>& order, const vector<vector<int>>& times) {
    int m = times[0].size();
    vector<int> machine_ready(m, 0);

    for (int job : order) {
        int prev = 0;
        for (int k = 0; k < m; k++) {
            int start = max(machine_ready[k], prev);
            int completion = start + times[job][k];
            machine_ready[k] = completion;
            prev = completion;
        }
    }
    return machine_ready.back();
}

/* ===================== GA OPERATORS ===================== */
vector<int> random_permutation(int n) {
    vector<int> p(n);
    iota(p.begin(), p.end(), 0);
    shuffle(p.begin(), p.end(), rng);
    return p;
}

vector<int> tournament_selection(
    const vector<vector<int>>& pop,
    const vector<int>& fitness,
    int k
) {
    int best = rand_int(0, pop.size() - 1);
    for (int i = 1; i < k; i++) {
        int cand = rand_int(0, pop.size() - 1);
        if (fitness[cand] < fitness[best])
            best = cand;
    }
    return pop[best];
}

pair<vector<int>, vector<int>> order_crossover(
    const vector<int>& p1,
    const vector<int>& p2
) {
    int n = p1.size();
    int a = rand_int(0, n - 1);
    int b = rand_int(0, n - 1);
    if (a > b) swap(a, b);

    auto ox = [&](const vector<int>& A, const vector<int>& B) {
        vector<int> child(n, -1);
        for (int i = a; i <= b; i++)
            child[i] = A[i];

        int pos = (b + 1) % n;
        for (int i = 0; i < n; i++) {
            int job = B[(b + 1 + i) % n];
            if (find(child.begin(), child.end(), job) == child.end()) {
                child[pos] = job;
                pos = (pos + 1) % n;
            }
        }
        return child;
    };

    return {ox(p1, p2), ox(p2, p1)};
}

void swap_mutation(vector<int>& ind, double rate) {
    if (rand_double() < rate) {
        int i = rand_int(0, ind.size() - 1);
        int j = rand_int(0, ind.size() - 1);
        swap(ind[i], ind[j]);
    }
}

/* ===================== NEH ===================== */
vector<int> neh_sequence(const vector<vector<int>>& times) {
    int n = times.size();
    vector<int> jobs(n);
    iota(jobs.begin(), jobs.end(), 0);

    sort(jobs.begin(), jobs.end(), [&](int a, int b) {
        return accumulate(times[a].begin(), times[a].end(), 0) >
               accumulate(times[b].begin(), times[b].end(), 0);
    });

    vector<int> seq = {jobs[0]};
    for (int i = 1; i < n; i++) {
        int job = jobs[i];
        vector<int> best_seq;
        int best_val = INT_MAX;

        for (int pos = 0; pos <= seq.size(); pos++) {
            vector<int> trial = seq;
            trial.insert(trial.begin() + pos, job);
            int val = makespan(trial, times);
            if (val < best_val) {
                best_val = val;
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

    int lb_machine = 0;
    for (int k = 0; k < m; k++) {
        int sum = 0;
        for (int j = 0; j < n; j++)
            sum += times[j][k];
        lb_machine = max(lb_machine, sum);
    }

    int lb_job = 0;
    for (int j = 0; j < n; j++) {
        int sum = accumulate(times[j].begin(), times[j].end(), 0);
        lb_job = max(lb_job, sum);
    }

    int ub = 0;
    for (auto& r : times)
        ub += accumulate(r.begin(), r.end(), 0);

    return {lb_machine, lb_job, max(lb_machine, lb_job), ub};
}

/* ===================== DIVERSITY ===================== */
double population_diversity(
    const vector<vector<int>>& pop,
    int sample_size
) {
    int n = pop.size();
    if (n < 2) return 0.0;

    vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);
    shuffle(idx.begin(), idx.end(), rng);
    int s = min(sample_size, n);

    double total = 0.0;
    int cnt = 0;
    for (int i = 0; i < s; i++) {
        for (int j = i + 1; j < s; j++) {
            int d = 0;
            for (int k = 0; k < pop[0].size(); k++)
                if (pop[idx[i]][k] != pop[idx[j]][k])
                    d++;
            total += double(d) / pop[0].size();
            cnt++;
        }
    }
    return cnt ? total / cnt : 0.0;
}

/* ===================== IO ===================== */
vector<vector<int>> read_input(const string& path) {
    ifstream f(path);
    if (!f) throw runtime_error("Cannot open file");

    int n, m;
    f >> n >> m;
    vector<vector<int>> times(n, vector<int>(m));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            f >> times[i][j];
    return times;
}

void print_schedule(const vector<int>& order, const vector<vector<int>>& times) {
    int n = order.size();
    int m = times[0].size();
    vector<vector<int>> comp(n, vector<int>(m));
    vector<int> ready(m, 0);

    for (int i = 0; i < n; i++) {
        int job = order[i];
        int prev = 0;
        for (int k = 0; k < m; k++) {
            int start = max(ready[k], prev);
            comp[i][k] = start + times[job][k];
            ready[k] = comp[i][k];
            prev = comp[i][k];
        }
    }

    cout << "\nSchedule:\nPos Job ";
    for (int k = 0; k < m; k++) cout << "M" << k+1 << " ";
    cout << "\n";

    for (int i = 0; i < n; i++) {
        cout << i+1 << "   " << order[i]+1 << "  ";
        for (int k = 0; k < m; k++)
            cout << comp[i][k] << " ";
        cout << "\n";
    }
    cout << "Cmax = " << comp.back().back() << "\n";
}

/* ===================== MAIN ===================== */
int main(int argc, char* argv[]) {
    string input_file = "data5.txt";
    string init_mode = "random";   // "neh" lub "random"

    if (argc > 1) input_file = argv[1];
    if (argc > 2) init_mode = argv[2];
    
    auto times = read_input(input_file);

    int n = times.size();

    int pop_size = 120;
    int generations = 5000;
    double cx_prob = 0.9;
    double mut_rate = 0.2;
    int elitism = 2;
    int stagnation_limit = 2000;
    int mut_boost = 500;
    bool useNEH = true;

    vector<vector<int>> population;

    if (init_mode == "neh") {
        cout << "→ Using NEH initialization\n";
        vector<int> neh = neh_sequence(times);
        population.push_back(neh);

        for (int i = 1; i < pop_size; i++) {
            auto p = neh;
            swap(p[rand_int(0,n-1)], p[rand_int(0,n-1)]);
            population.push_back(p);
        }
    } else {
        cout << "→ Using RANDOM initialization\n";
        for (int i = 0; i < pop_size; i++)
            population.push_back(random_permutation(n));
    }


    vector<int> fitness(pop_size);
    for (int i = 0; i < pop_size; i++)
        fitness[i] = makespan(population[i], times);

    int lbm, lbj, lb, ub;
    std::tie(lbm, lbj, lb, ub) = compute_bounds(times);


    int best = *min_element(fitness.begin(), fitness.end());
    int stagnation = 0;
    double cur_mut = mut_rate;

    for (int gen = 0; gen < generations; gen++) {
        vector<vector<int>> new_pop;

        vector<int> idx(pop_size);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b) {
            return fitness[a] < fitness[b];
        });

        for (int i = 0; i < elitism; i++)
            new_pop.push_back(population[idx[i]]);

        while (new_pop.size() < pop_size) {
            auto p1 = tournament_selection(population, fitness, 3);
            auto p2 = tournament_selection(population, fitness, 3);

            if (rand_double() < cx_prob) {
                auto children = order_crossover(p1, p2);
                vector<int> c1 = children.first;
                vector<int> c2 = children.second;

                swap_mutation(c1, cur_mut);
                swap_mutation(c2, cur_mut);

                new_pop.push_back(c1);
                if (new_pop.size() < pop_size)
                    new_pop.push_back(c2);
            } else {
                new_pop.push_back(p1);
            }
        }

        population = new_pop;
        for (int i = 0; i < pop_size; i++)
            fitness[i] = makespan(population[i], times);

        int cur_best = *min_element(fitness.begin(), fitness.end());
        if (cur_best < best) {
            best = cur_best;
            stagnation = 0;
            cur_mut = mut_rate;
        } else stagnation++;

        if (stagnation == mut_boost)
            cur_mut = min(0.9, cur_mut * 2);

        if (stagnation >= stagnation_limit)
            break;
        if (gen < 10 || (gen + 1) % (generations / 100) == 0) {
        double gap = 100.0 * (best - lb) / lb;
        cout << "Gen " << gen + 1
            << " | Best Cmax = " << best
            << " | Gap = " << fixed << setprecision(2) << gap << "%"
            << " | NoImprove = " << stagnation
            << "\n";
    }

    }

    int best_idx = min_element(fitness.begin(), fitness.end()) - fitness.begin();
    cout << "\nBEST ORDER:\n";
    for (int j : population[best_idx]) cout << j+1 << " ";
    cout << "\nCmax = " << fitness[best_idx] << "\n";
    cout << "Gap = " << (fitness[best_idx] - lb) * 100.0 / lb << "%\n";

    print_schedule(population[best_idx], times);
}
