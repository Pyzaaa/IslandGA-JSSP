#pragma once
#include <vector>
#include <numeric>
#include <random>
#include <fstream>
#include <stdexcept>
#include <algorithm>

using namespace std;

/* ===================== RANDOM ===================== */
mt19937 rng(random_device{}());

inline int rand_int(int a, int b) {
    uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

inline double rand_double() {
    uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

/* ===================== GA OPERATORS ===================== */
inline vector<int> random_permutation(int n) {
    vector<int> p(n);
    iota(p.begin(), p.end(), 0);
    shuffle(p.begin(), p.end(), rng);
    return p;
}

inline vector<vector<int>> read_input(const string& path) {
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

inline vector<int> flatten_population(const vector<vector<int>>& pop) {
    int n = pop[0].size();
    int pop_size = pop.size();
    vector<int> flat(pop_size * n);
    for (int i = 0; i < pop_size; i++)
        for (int j = 0; j < n; j++)
            flat[i * n + j] = pop[i][j];
    return flat;
}

inline vector<int> flatten_times(const vector<vector<int>>& times) {
    int n = times.size();
    int m = times[0].size();
    vector<int> flat(n * m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            flat[i * m + j] = times[i][j];
    return flat;
}
