#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

__global__ void makespan_kernel(int* times, int* orders, int* results, int n_jobs, int n_machines) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_jobs) return;

    int job_idx = orders[idx];
    int completion = 0;
    int prev = 0;
    for (int m = 0; m < n_machines; m++) {
        int t = times[job_idx * n_machines + m];
        int start = max(prev, completion); // simplified per thread, can be expanded
        completion = start + t;
        prev = completion;
    }
    results[idx] = completion;
}

int makespan_gpu(const std::vector<int>& order, const std::vector<std::vector<int>>& times) {
    int n_jobs = order.size();
    int n_machines = times[0].size();

    std::vector<int> flat_times(n_jobs * n_machines);
    for(int i = 0; i < n_jobs; i++)
        for(int j = 0; j < n_machines; j++)
            flat_times[i * n_machines + j] = times[i][j];

    int *d_times, *d_orders, *d_results;
    cudaMalloc(&d_times, flat_times.size() * sizeof(int));
    cudaMalloc(&d_orders, order.size() * sizeof(int));
    cudaMalloc(&d_results, order.size() * sizeof(int));

    cudaMemcpy(d_times, flat_times.data(), flat_times.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_orders, order.data(), order.size() * sizeof(int), cudaMemcpyHostToDevice);

    makespan_kernel<<<1, n_jobs>>>(d_times, d_orders, d_results, n_jobs, n_machines);

    std::vector<int> results(n_jobs);
    cudaMemcpy(results.data(), d_results, n_jobs * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_times);
    cudaFree(d_orders);
    cudaFree(d_results);

    return results.back();
}
