import random
import json

def generate_p_parallel_with_deadlines(
    num_instances=10,
    num_jobs_list=[20, 50, 100],
    num_machines_list=[2, 5, 10],
    processing_time_distribution="uniform",
    p_min=1,
    p_max=100,
    release_time_factor=0.5,
    deadline_slack_factor=1.5,
    output_file="p_parallel_deadlines_dataset.json"
):
    """
    Generate synthetic dataset for P|r_j,d_j|Cmax (identical machines, release times, deadlines).

    Parameters
    ----------
    num_instances : int
        How many random instances per configuration to generate.
    num_jobs_list : list[int]
        Different job counts to generate (e.g., [20, 50, 100]).
    num_machines_list : list[int]
        Number of machines to generate (e.g., [2, 5, 10]).
    processing_time_distribution : str
        'uniform', 'normal', or 'exponential'.
    p_min, p_max : int
        Range for processing times.
    release_time_factor : float
        Controls how far in time jobs may be released. Larger = more spread out arrivals.
        Roughly defines max release time as release_time_factor * average processing time * n/m.
    deadline_slack_factor : float
        Deadlines = release_time + processing_time * random.uniform(1, deadline_slack_factor).
        Lower means tighter deadlines.
    output_file : str
        Path for JSON file.
    """

    def generate_processing_times(n, distribution, p_min, p_max):
        if distribution == "uniform":
            return [random.randint(p_min, p_max) for _ in range(n)]
        elif distribution == "normal":
            mu = (p_max + p_min) / 2
            sigma = (p_max - p_min) / 6
            return [max(p_min, min(int(random.gauss(mu, sigma)), p_max)) for _ in range(n)]
        elif distribution == "exponential":
            scale = (p_max + p_min) / 4
            return [max(p_min, min(int(random.expovariate(1/scale)), p_max)) for _ in range(n)]
        else:
            raise ValueError("Unknown distribution type")

    dataset = []
    instance_id = 1

    for m in num_machines_list:
        for n in num_jobs_list:
            for _ in range(num_instances):
                p_times = generate_processing_times(n, processing_time_distribution, p_min, p_max)
                avg_p = sum(p_times) / n
                max_release = int(release_time_factor * avg_p * n / m)

                release_times = [random.randint(0, max_release) for _ in range(n)]
                deadlines = [
                    r + int(p * random.uniform(1.0, deadline_slack_factor))
                    for r, p in zip(release_times, p_times)
                ]

                instance = {
                    "id": f"instance_{instance_id}",
                    "machines": m,
                    "jobs": n,
                    "processing_times": p_times,
                    "release_times": release_times,
                    "deadlines": deadlines,
                    "lower_bound": max(max(p_times), sum(p_times) / m)
                }

                dataset.append(instance)
                instance_id += 1

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated {len(dataset)} instances with release times & deadlines → '{output_file}'")
    return dataset


# Example usage
if __name__ == "__main__":
    dataset = generate_p_parallel_with_deadlines(
        num_instances=3,
        num_jobs_list=[100],
        num_machines_list=[5, 10],
        processing_time_distribution="normal",
        p_min=10,
        p_max=500,
        release_time_factor=0.8,     # controls spread of job arrivals
        deadline_slack_factor=1.4,   # smaller → tighter deadlines
        output_file="p_parallel_deadlines_dataset.json"
    )
