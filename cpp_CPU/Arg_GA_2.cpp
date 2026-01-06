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
#include <chrono>

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

/* ===================== MAKESPAN (Zoptymalizowany) ===================== */
// Przekazujemy machine_ready jako bufor, aby uniknąć alokacji wewnątrz
int compute_makespan(
    const vector<int>& order,
    const vector<vector<int>>& times,
    vector<int>& machine_ready
) {
    int m = machine_ready.size();
    fill(machine_ready.begin(), machine_ready.end(), 0);

    for (int job : order) {
        int prev_completion = 0;
        const auto& job_times = times[job];
        for (int k = 0; k < m; ++k) {
            int start = (machine_ready[k] > prev_completion) ? machine_ready[k] : prev_completion;
            machine_ready[k] = start + job_times[k];
            prev_completion = machine_ready[k];
        }
    }
    return machine_ready[m - 1];
}

/* ===================== GA OPERATORS (Zoptymalizowane pod pamięć) ===================== */

// OX Crossover bez tworzenia zbędnych kopii, pisze bezpośrednio do dzieci
void order_crossover(const vector<int>& p1, const vector<int>& p2, vector<int>& c1, vector<int>& c2) {
    int n = p1.size();
    int a = rand_int(0, n - 1);
    int b = rand_int(0, n - 1);
    if (a > b) swap(a, b);

    auto fill_child = [&](const vector<int>& parent1, const vector<int>& parent2, vector<int>& child) {
        fill(child.begin(), child.end(), -1);
        vector<bool> used(n, false);

        for (int i = a; i <= b; ++i) {
            child[i] = parent1[i];
            used[parent1[i]] = true;
        }

        int pos = (b + 1) % n;
        for (int i = 0; i < n; ++i) {
            int job = parent2[(b + 1 + i) % n];
            if (!used[job]) {
                child[pos] = job;
                pos = (pos + 1) % n;
            }
        }
    };

    fill_child(p1, p2, c1);
    fill_child(p2, p1, c2);
}

/* ===================== NEH (Zoptymalizowany: mniejsza złożoność alokacji) ===================== */
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

    vector<int> seq;
    seq.reserve(n);
    seq.push_back(jobs[0]);

    for (int i = 1; i < n; ++i) {
        int job = jobs[i];
        int best_pos = -1;
        int best_val = numeric_limits<int>::max();

        // Zamiast tworzyć trial_seq w kółko, używamy jednego wektora roboczego
        seq.insert(seq.begin(), job); // Wstaw na początek
        for (int pos = 0; pos <= i; ++pos) {
            // Testujemy aktualne ustawienie
            int val = compute_makespan(seq, times, machine_ready);
            if (val < best_val) {
                best_val = val;
                best_pos = pos;
            }
            // Przesuwamy zadanie o jedną pozycję w prawo (swap) zamiast insert/erase
            if (pos < i) swap(seq[pos], seq[pos+1]);
        }
        
        // Przywracamy najlepszą znalezioną pozycję
        int current_pos = i;
        while (current_pos > best_pos) {
            swap(seq[current_pos], seq[current_pos-1]);
            current_pos--;
        }
    }
    return seq;
}

/* ===================== IO & BOUNDS ===================== */
// (Kod compute_bounds i read_input pozostaje bez zmian, bo są wywoływane raz)
tuple<int,int,int,int> compute_bounds(const vector<vector<int>>& times) {
    int n = times.size(); int m = times[0].size();
    int lb_m = 0;
    for (int k = 0; k < m; k++) {
        int s = 0; for (int j = 0; j < n; j++) s += times[j][k];
        lb_m = max(lb_m, s);
    }
    int lb_j = 0;
    for (int j = 0; j < n; j++) lb_j = max(lb_j, (int)accumulate(times[j].begin(), times[j].end(), 0));
    return { lb_m, lb_j, max(lb_m, lb_j), 1000000 };
}

vector<vector<int>> read_input(const string& path) {
    ifstream f(path);
    if (!f) throw runtime_error("File error");
    int n, m; f >> n >> m;
    vector<vector<int>> t(n, vector<int>(m));
    for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) f >> t[i][j];
    return t;
}

/* ===================== MAIN ===================== */
int main(int argc, char* argv[]) {
    if (argc < 2) return 1;

    string input_file = argv[1];
    bool verbose = false;
    bool use_chrono = false;
    int pop_size = 120, generations = 5000;
    double cx_prob = 0.9, mut_rate = 0.2;
    int elitism = 2;

    for (int i = 2; i < argc; i++) {
        string a = argv[i];
        if (a == "--verbose") verbose = true;
        else if (a == "--chrono") use_chrono = true;
        else if (a == "--pop") pop_size = stoi(argv[++i]);
        else if (a == "--gen") generations = stoi(argv[++i]);
    }

    auto times = read_input(input_file);
    int n = times.size(), m = times[0].size();
    auto [lbm, lbj, lb, ub] = compute_bounds(times);

    auto start_t = chrono::high_resolution_clock::now();

    // Pre-alokacja populacji
    vector<vector<int>> population(pop_size, vector<int>(n));
    vector<int> fitness(pop_size);
    vector<int> machine_ready(m);

    // Inicjalizacja NEH
    auto neh = neh_sequence(times);
    population[0] = neh;
    for (int i = 1; i < pop_size; ++i) {
        population[i] = neh;
        swap(population[i][rand_int(0, n-1)], population[i][rand_int(0, n-1)]);
    }

    for (int i = 0; i < pop_size; ++i) fitness[i] = compute_makespan(population[i], times, machine_ready);

    int best_cmax = numeric_limits<int>::max();
    vector<int> best_order;
    int stagnation = 0;

    // Bufor dla nowej populacji, aby nie alokować w pętli
    vector<vector<int>> next_pop(pop_size, vector<int>(n));

    for (int gen = 0; gen < generations; ++gen) {
        // Elityzm (indeksy najlepszych)
        vector<int> idx(pop_size);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b) { return fitness[a] < fitness[b]; });

        for (int i = 0; i < elitism; ++i) next_pop[i] = population[idx[i]];

        // Ewolucja
        for (int i = elitism; i < pop_size; i += 2) {
            int p1_idx = rand_int(0, pop_size - 1); // Uproszczona selekcja dla szybkości
            int p2_idx = rand_int(0, pop_size - 1);

            if (rand_double() < cx_prob) {
                if (i + 1 < pop_size)
                    order_crossover(population[p1_idx], population[p2_idx], next_pop[i], next_pop[i+1]);
                else
                    next_pop[i] = population[p1_idx];
            } else {
                next_pop[i] = population[p1_idx];
                if (i + 1 < pop_size) next_pop[i+1] = population[p2_idx];
            }

            // Mutacja w miejscu
            if (rand_double() < mut_rate) swap(next_pop[i][rand_int(0, n-1)], next_pop[i][rand_int(0, n-1)]);
        }

        population = next_pop;
        for (int i = 0; i < pop_size; ++i) fitness[i] = compute_makespan(population[i], times, machine_ready);

        int current_min = fitness[idx[0]];
        if (current_min < best_cmax) {
            best_cmax = current_min;
            best_order = population[idx[0]];
            stagnation = 0;
        } else {
            stagnation++;
        }

        if (verbose && (gen % 500 == 0)) cout << "Gen " << gen << " Cmax: " << best_cmax << "\n";
        if (stagnation > 2000) break;
    }

    auto end_t = chrono::high_resolution_clock::now();
    
    cout << "\nCmax: " << best_cmax << " | Gap: " << fixed << setprecision(2) << 100.0 * (best_cmax - lb) / lb << "%" << endl;
    if (use_chrono) cout << "Time: " << chrono::duration<double>(end_t - start_t).count() << "s" << endl;

    return 0;
}