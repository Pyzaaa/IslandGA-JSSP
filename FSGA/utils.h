#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <iostream>
#include <numeric>

extern std::mt19937 rng;

// ===================== RANDOM HELPERS =====================
inline int rand_int(int a, int b) {
    std::uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

inline double rand_double() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

// ===================== IO =====================
inline std::vector<std::vector<int>> read_input(const std::string& path) {
    std::ifstream f(path);
    if(!f) throw std::runtime_error("Cannot open file " + path);

    int n, m;
    f >> n >> m;

    std::vector<std::vector<int>> times(n, std::vector<int>(m));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            f >> times[i][j];
    return times;
}

inline void print_schedule(const std::vector<int>& order, const std::vector<std::vector<int>>& times) {
    int n = order.size();
    int m = times[0].size();
    std::vector<std::vector<int>> comp(n, std::vector<int>(m));
    std::vector<int> ready(m,0);

    for(int i=0;i<n;i++){
        int prev = 0;
        int job = order[i];
        for(int k=0;k<m;k++){
            int start = std::max(ready[k], prev);
            comp[i][k] = start + times[job][k];
            ready[k] = comp[i][k];
            prev = comp[i][k];
        }
    }

    std::cout << "\nSchedule:\nPos Job ";
    for(int k=0;k<m;k++) std::cout << "M" << k+1 << " ";
    std::cout << "\n";

    for(int i=0;i<n;i++){
        std::cout << i+1 << "   " << order[i]+1 << "  ";
        for(int k=0;k<m;k++) std::cout << comp[i][k] << " ";
        std::cout << "\n";
    }
    std::cout << "Cmax = " << comp.back().back() << "\n";
}
