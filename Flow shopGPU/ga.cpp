#include "ga.h"
#include <algorithm>

int makespan_cpu(const std::vector<int>& order,
                 const std::vector<int>& times,
                 int nJobs, int nMachines)
{
    std::vector<int> machine_ready(nMachines, 0);

    for (int pos = 0; pos < nJobs; pos++) {
        int job = order[pos];
        int prev = 0;
        for (int m = 0; m < nMachines; m++) {
            int start = std::max(machine_ready[m], prev);
            int comp  = start + times[job * nMachines + m];
            machine_ready[m] = comp;
            prev = comp;
        }
    }
    return machine_ready[nMachines - 1];
}

void compute_bounds(const std::vector<int>& times,
                    int nJobs, int nMachines,
                    int& lbm, int& lbj, int& lb, int& ub)
{
    lbm = 0;
    for (int m = 0; m < nMachines; m++) {
        int sum = 0;
        for (int j = 0; j < nJobs; j++)
            sum += times[j * nMachines + m];
        lbm = std::max(lbm, sum);
    }

    lbj = 0;
    for (int j = 0; j < nJobs; j++) {
        int sum = 0;
        for (int m = 0; m < nMachines; m++)
            sum += times[j * nMachines + m];
        lbj = std::max(lbj, sum);
    }

    lb = std::max(lbm, lbj);

    ub = 0;
    for (int x : times) ub += x;
}
