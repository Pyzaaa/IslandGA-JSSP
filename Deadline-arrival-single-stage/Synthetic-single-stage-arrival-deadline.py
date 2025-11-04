import random
import json

def generate_processing_times(n, distribution="uniform", p_min=1, p_max=100):
    """Generate n processing times from a given distribution."""
    if distribution == "uniform":
        return [random.randint(p_min, p_max) for _ in range(n)]
    elif distribution == "normal":
        mu = (p_max + p_min) / 2
        sigma = (p_max - p_min) / 6
        return [max(p_min, min(int(random.gauss(mu, sigma)), p_max)) for _ in range(n)]
    elif distribution == "exponential":
        scale = (p_max + p_min) / 4
        return [max(p_min, min(int(random.expovariate(1 / scale)), p_max)) for _ in range(n)]
    else:
        raise ValueError("Unknown processing time distribution")


def generate_release_times(p_times, num_machines, overlap_factor=0.8):
    """
    Generate realistic release times ensuring jobs slightly overlap.

    - Start first job at time 0.
    - Each subsequent job releases when the previous one would roughly finish on one machine,
      multiplied by an overlap factor (<1 means overlap).
    - Ensures max release <= total_work / num_machines.
    """
    releases = [0]
    total_work = sum(p_times)
    horizon = total_work / num_machines
    cumulative = 0

    for i in range(1, len(p_times)):
        cumulative += p_times[i - 1] * overlap_factor
        release = min(int(cumulative), int(horizon))
        releases.append(release)

    # Optionally add a small jitter to break ties
    jitter = [r + random.uniform(0, p_times[i] * 0.05) for i, r in enumerate(releases)]
    return [int(r) for r in jitter]


def generate_deadlines(release_times, p_times, num_machines, global_slack_factor, local_deadline_jitter):
    """
    Generate deadlines that allow feasible scheduling:
      d_j = r_j + p_j + (avg_load_per_machine * slack)
    """
    total_work = sum(p_times)
    nominal_horizon = total_work / num_machines
    deadlines = []

    for r, p in zip(release_times, p_times):
        slack = global_slack_factor * (nominal_horizon / len(p_times))
        jitter = random.uniform(1 - local_deadline_jitter, 1 + local_deadline_jitter)
        d = int(r + p + slack * jitter)
        deadlines.append(d)

    return deadlines


def generate_p_parallel_with_deadlines(
    num_instances=10,
    num_jobs_list=[20, 50, 100],
    num_machines_list=[2, 5, 10],
    processing_time_distribution="uniform",
    p_min=1,
    p_max=100,
    overlap_factor=0.8,
    global_slack_factor=1.3,
    local_deadline_jitter=0.2,
    output_file="p_parallel_deadlines_dataset.json"
):
    """
    Generate synthetic dataset for P|r_j,d_j|ΣT_j (identical machines, release times, deadlines),
    with overlapping release times and feasible deadlines.
    """

    dataset = []
    instance_id = 1

    for m in num_machines_list:
        for n in num_jobs_list:
            for _ in range(num_instances):
                # Generate components
                p_times = generate_processing_times(n, processing_time_distribution, p_min, p_max)
                r_times = generate_release_times(p_times, m, overlap_factor)
                d_times = generate_deadlines(r_times, p_times, m, global_slack_factor, local_deadline_jitter)

                total_work = sum(p_times)
                instance = {
                    "id": f"instance_{instance_id}",
                    "machines": m,
                    "jobs": n,
                    "processing_times": p_times,
                    "release_times": r_times,
                    "deadlines": d_times,
                    "lower_bound": max(max(p_times), total_work / m),
                }

                dataset.append(instance)
                instance_id += 1

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated {len(dataset)} overlapping-load instances → '{output_file}'")
    return dataset


# Example usage
if __name__ == "__main__":
    dataset = generate_p_parallel_with_deadlines(
        num_instances=1,
        num_jobs_list=[100],
        num_machines_list=[5],
        processing_time_distribution="normal",
        p_min=10,
        p_max=200,
        overlap_factor=0.8,       # 1.0 = no overlap, <1.0 = more overlap
        global_slack_factor=1.5,
        local_deadline_jitter=0.2,
        output_file="p_parallel_deadlines_dataset.json"
    )
