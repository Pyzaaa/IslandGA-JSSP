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
#include <chrono> // Dodano dla obsługi czasu
#include <omp.h> // dla pararell


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
int makespan(
    const vector<int>& order,
    const vector<vector<int>>& times,
    vector<int>& machine_ready
) {
    int m = machine_ready.size();
    fill(machine_ready.begin(), machine_ready.end(), 0);

    for (int job : order) {
        int prev = 0;
        const auto& job_times = times[job];
        for (int k = 0; k < m; k++) {
            int start = max(machine_ready[k], prev);
            int completion = start + job_times[k];
            machine_ready[k] = completion;
            prev = completion;
        }
    }
    return machine_ready[m - 1];
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
            trial.insert(trial.begin() + pos, job);
            int val = makespan(trial, times, machine_ready);
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

    return { lb_machine, lb_job, max(lb_machine, lb_job), ub };
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

/* ===================== MAIN ===================== */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.txt [options]\n";
        return 1;
    }

    string input_file = argv[1];
    string init_mode = "neh";
    bool verbose = false;
    bool use_chrono = false; // Flaga pomiaru czasu

    int pop_size = 120;
    int generations = 5000;
    double cx_prob = 0.9;
    double mut_rate = 0.2;
    int elitism = 2;
    int stagnation_limit = 2000;
    int mut_boost = 500;

    for (int i = 2; i < argc; i++) {
        string a = argv[i];
        if (a == "--init") init_mode = argv[++i];
        else if (a == "--pop") pop_size = stoi(argv[++i]);
        else if (a == "--gen") generations = stoi(argv[++i]);
        else if (a == "--cx") cx_prob = stod(argv[++i]);
        else if (a == "--mut") mut_rate = stod(argv[++i]);
        else if (a == "--elitism") elitism = stoi(argv[++i]);
        else if (a == "--stagnation") stagnation_limit = stoi(argv[++i]);
        else if (a == "--mut-boost") mut_boost = stoi(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else if (a == "--chrono") use_chrono = true; // Obsługa argumentu
    }

    auto times = read_input(input_file);
    int n = times.size();
    int m = times[0].size();

    int lbm, lbj, lb, ub;
    tie(lbm, lbj, lb, ub) = compute_bounds(times);

    // ROZPOCZĘCIE POMIARU CZASU
    auto start_time = chrono::high_resolution_clock::now();

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

    vector<int> machine_ready(m);
    vector<int> fitness(pop_size);



    for (int i = 0; i < pop_size; i++)
        fitness[i] = makespan(population[i], times, machine_ready);


    int best = INT_MAX;
    vector<int> best_perm;
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
                auto [c1, c2] = order_crossover(p1, p2);
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
        #pragma omp parallel for
        for (int i = 0; i < pop_size; i++) {
            vector<int> local_machine_ready(m);
            fitness[i] = makespan(population[i], times, local_machine_ready);
        }


        int cur_best = *min_element(fitness.begin(), fitness.end());
        if (cur_best < best) {
            best = cur_best;
            best_perm = population[min_element(fitness.begin(), fitness.end()) - fitness.begin()];
            stagnation = 0;
            cur_mut = mut_rate;
        } else stagnation++;

        if (stagnation == mut_boost)
            cur_mut = min(0.9, cur_mut * 2);

        if (verbose && (gen < 10 || (gen + 1) % (generations / 100) == 0)) {
            double gap = 100.0 * (best - lb) / lb;
            cout << "Gen " << gen + 1
                 << " | Best Cmax = " << best
                 << " | Gap = " << fixed << setprecision(2) << gap << "%"
                 << " | NoImprove = " << stagnation << "\n";
        }

        if (stagnation >= stagnation_limit)
            break;
    }

    // KONIEC POMIARU CZASU
    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "\nBEST ORDER:\n";
    for (int j : best_perm) cout << j + 1 << " ";
    cout << "\nCmax = " << best << "\n";
    cout << "Gap = " << (best - lb) * 100.0 / lb << "%\n";

    if (use_chrono) {
        cout << "Execution time: " << fixed << setprecision(4) << elapsed.count() << " seconds\n";
    }
}