#define BOOST_ASIO_HEADER_ONLY
#define _WIN32_WINNT 0x0A00 // Targets Windows 10/11
#include <iostream>
#include <vector>

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
#include <omp.h>
#include <thread>
#include <mutex>
#include <boost/asio.hpp>



using namespace std;
using boost::asio::ip::tcp;

/* ===================== ASYNC MIGRATION GLOBALS ===================== */
vector<vector<int>> migrant_buffer;
mutex buffer_mutex;
bool keep_running = true;

// --- BACKGROUND RECEIVER ---
void start_receiver(int port) {
    try {
        boost::asio::io_context io_context;
        tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), port));
        while (keep_running) {
            tcp::socket socket(io_context);
            acceptor.accept(socket); // Waits for a migrant

            int n;
            boost::asio::read(socket, boost::asio::buffer(&n, sizeof(int)));
            vector<int> migrant(n);
            boost::asio::read(socket, boost::asio::buffer(migrant.data(), n * sizeof(int)));

            lock_guard<mutex> lock(buffer_mutex);
            migrant_buffer.push_back(migrant);
        }
    } catch (...) { /* Handle or log connection errors */ }
}

// --- FIRE-AND-FORGET SENDER ---
void send_migrant(string ip, int port, vector<int> individual) {
    try {
        boost::asio::io_context io_context;
        tcp::socket socket(io_context);
        tcp::resolver resolver(io_context);
        boost::asio::connect(socket, resolver.resolve(ip, to_string(port)));

        int n = individual.size();
        boost::asio::write(socket, boost::asio::buffer(&n, sizeof(int)));
        boost::asio::write(socket, boost::asio::buffer(individual.data(), n * sizeof(int)));
    } catch (...) { /* Peer might be offline, ignore */ }
}


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

/* ===================== UTILITY ===================== */
double get_distance(const vector<int>& a, const vector<int>& b) {
    int diff = 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) diff++;
    return (double)diff / a.size();
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

/* ===================== INTEGRATED MAIN ===================== */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " input.txt [options]\n";
        return 1;
    }

    string input_file = argv[1];
    string init_mode = "neh";
    bool verbose = false;
    bool use_chrono = false;

    // GA Parameters
    int pop_size = 120;
    int generations = 5000;
    double cx_prob = 0.9;
    double mut_rate = 0.2;
    int elitism = 2;
    int stagnation_limit = 2000;
    int mut_boost = 500;

    // Island/Network Parameters
    int my_port = 12345;
    string peer_ip = "127.0.0.1"; // Default to localhost
    int peer_port = 12345;
    int migration_interval = 100;

    // Argument Parsing
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
        else if (a == "--chrono") use_chrono = true;
        // Network args
        else if (a == "--my-port") my_port = stoi(argv[++i]);
        else if (a == "--peer-ip") peer_ip = argv[++i];
        else if (a == "--peer-port") peer_port = stoi(argv[++i]);
        else if (a == "--mig-int") migration_interval = stoi(argv[++i]);
    }

    auto times = read_input(input_file);
    int n = times.size();
    int m = times[0].size();

    int lbm, lbj, lb, ub;
    tie(lbm, lbj, lb, ub) = compute_bounds(times);

    // 1. START BACKGROUND NETWORK LISTENER
    thread net_thread(start_receiver, my_port);
    net_thread.detach();

    auto start_time = chrono::high_resolution_clock::now();

    // Population Initialization
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

    // --- EVOLUTION LOOP ---
    for (int gen = 0; gen < generations; gen++) {
        vector<vector<int>> new_pop;

        // Sort indices for Elitism and Migration replacement
        vector<int> idx(pop_size);
        iota(idx.begin(), idx.end(), 0);
        sort(idx.begin(), idx.end(), [&](int a, int b) {
            return fitness[a] < fitness[b];
        });

        // Elitism
        for (int i = 0; i < elitism; i++)
            new_pop.push_back(population[idx[i]]);

        // Crossover and Mutation
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

        // Parallel Fitness Evaluation
        #pragma omp parallel for
        for (int i = 0; i < pop_size; i++) {
            vector<int> local_machine_ready(m);
            fitness[i] = makespan(population[i], times, local_machine_ready);
        }

        // Update Global Best
        int cur_best_idx = min_element(fitness.begin(), fitness.end()) - fitness.begin();
        int cur_best_val = fitness[cur_best_idx];

        if (cur_best_val < best) {
            best = cur_best_val;
            best_perm = population[cur_best_idx];
            stagnation = 0;
            cur_mut = mut_rate;
        } else {
            stagnation++;
        }

// 2. MIGRATION LOGIC
        if (gen > 0 && gen % migration_interval == 0) {
            // Send current best to neighbor
            thread(send_migrant, peer_ip, peer_port, best_perm).detach();

            lock_guard<mutex> lock(buffer_mutex);
            if (!migrant_buffer.empty()) {
                // We'll analyze the most recent migrant
                vector<int> immigrant = migrant_buffer.back();

                // --- DIVERSITY CALCULATIONS ---
                auto get_dist = [](const vector<int>& a, const vector<int>& b) {
                    int d = 0;
                    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) d++;
                    return (double)d / a.size();
                };

                // 1. Novelty: Is the migrant different from our BEST?
                double novelty = get_dist(best_perm, immigrant);
                
                // 2. Internal Diversity: Is our population diverse? (Best vs Random)
                int rand_idx = rand_int(0, pop_size - 1);
                double internal_div = get_dist(best_perm, population[rand_idx]);

                if (verbose) {
                    cout << fixed << setprecision(2);
                    cout << ">>> [MIGRATION Gen " << gen << "]" << endl;
                    cout << "    Novelty (Migrant vs Local): " << (novelty * 100.0) << "%" << endl;
                    cout << "    Self-Diversity (Local):     " << (internal_div * 100.0) << "%";
                    
                    /*
                    if (novelty < 0.05) cout << "    [!] ALERT: Islands have converged (Redundant DNA)" << endl;
                    if (internal_div < 0.05) {
                        cout << "    [!] ALERT: Local stagnation! Boosting mutation..." << endl;
                        cur_mut = min(0.9, cur_mut * 1.5); // Self-healing boost
                    }
                    
                    cout << "------------------------------------------\n";*/
                    cout << "\n";
                }

                // --- INTEGRATION ---
                // Re-sort to ensure we replace the absolute worst
                vector<int> current_idx(pop_size);
                iota(current_idx.begin(), current_idx.end(), 0);
                sort(current_idx.begin(), current_idx.end(), [&](int a, int b) {
                    return fitness[a] < fitness[b];
                });

                int replaced = 0;
                // Limit replacement to not overwhelm local genes (max 10% or buffer size)
                int max_to_replace = max(1, pop_size / 10); 
                
                while (!migrant_buffer.empty() && replaced < max_to_replace) {
                    int worst_pos = current_idx[pop_size - 1 - replaced];
                    population[worst_pos] = migrant_buffer.back();
                    
                    vector<int> local_m(m);
                    fitness[worst_pos] = makespan(population[worst_pos], times, local_m);
                    
                    migrant_buffer.pop_back();
                    replaced++;
                }
            }
        }

        // Mutation Boost
        if (stagnation == mut_boost)
            cur_mut = min(0.9, cur_mut * 2);

        // Verbose Logging
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

    keep_running = false; // Signal network thread to stop (optional)

    auto end_time = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end_time - start_time;

    cout << "\nFINAL BEST ORDER:\n";
    for (int j : best_perm) cout << j + 1 << " ";
    cout << "\nCmax = " << best << "\n";
    cout << "Gap = " << (best - lb) * 100.0 / lb << "%\n";

    if (use_chrono) {
        cout << "Execution time: " << fixed << setprecision(4) << elapsed.count() << " seconds\n";
    }

    return 0;
}