import random
import json

def generate_p_parallel_instances(
    num_instances=10,
    num_jobs_list=[20, 50, 100],
    num_machines_list=[2, 5, 10],
    processing_time_distribution="uniform",
    p_min=1,
    p_max=100,
    output_file="p_parallel_dataset.json"
):
    """
    Generate synthetic dataset for P||Cmax (identical parallel machines scheduling).
    """

    def generate_processing_times(n, distribution, p_min, p_max):
        if distribution == "uniform":
            return [random.randint(p_min, p_max) for _ in range(n)]
        elif distribution == "normal":
            mu = (p_max + p_min) / 2
            sigma = (p_max - p_min) / 6  # ~99.7% within range
            times = [max(p_min, min(int(random.gauss(mu, sigma)), p_max)) for _ in range(n)]
            return times
        elif distribution == "exponential":
            scale = (p_max + p_min) / 4
            times = [max(p_min, min(int(random.expovariate(1/scale)), p_max)) for _ in range(n)]
            return times
        else:
            raise ValueError("Unknown distribution type")

    dataset = []
    instance_id = 1

    for m in num_machines_list:
        for n in num_jobs_list:
            for _ in range(num_instances):
                p_times = generate_processing_times(n, processing_time_distribution, p_min, p_max)
                instance = {
                    "id": f"instance_{instance_id}",
                    "machines": m,
                    "jobs": n,
                    "processing_times": p_times,
                    "lower_bound": max(max(p_times), sum(p_times) / m)
                }
                dataset.append(instance)
                instance_id += 1

    # Save dataset to file
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated {len(dataset)} instances and saved to '{output_file}'")
    return dataset


# Example usage
if __name__ == "__main__":
    dataset = generate_p_parallel_instances(
        num_instances=2,
        num_jobs_list=[3000],
        num_machines_list=[10],
        processing_time_distribution="normal",
        p_min=1,
        p_max=1000
    )
