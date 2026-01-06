#pragma once
#include <vector>

int makespan_cpu(const std::vector<int>& order,
                 const std::vector<int>& times,
                 int nJobs, int nMachines);

void compute_bounds(const std::vector<int>& times,
                    int nJobs, int nMachines,
                    int& lbm, int& lbj, int& lb, int& ub);
