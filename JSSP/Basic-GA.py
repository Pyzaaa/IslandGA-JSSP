import random
from load_jobshop import load_jssp_instance
import matplotlib.pyplot as plt

# ---------- Fitness evaluation ----------
def evaluate_schedule(jobs, chromosome):
    n_jobs = len(jobs)
    n_machines = len(jobs[0])

    job_next_op = [0] * n_jobs          # next operation index per job
    job_ready = [0] * n_jobs            # when each job becomes free
    machine_ready = [0] * n_machines    # when each machine becomes free

    schedule = []

    for job_id in chromosome:
        op_idx = job_next_op[job_id]
        if op_idx >= len(jobs[job_id]):   # skip if all operations done
            continue

        machine, duration = jobs[job_id][op_idx]

        # operation can start when both job and machine are available
        start = max(job_ready[job_id], machine_ready[machine])
        end = start + duration

        # update availability
        job_ready[job_id] = end
        machine_ready[machine] = end
        job_next_op[job_id] += 1

        # record operation for Gantt chart
        schedule.append({
            "job": job_id,
            "machine": machine,
            "start": start,
            "end": end
        })

    makespan = max(job_ready)
    return makespan, schedule



# ---------- Chromosome utilities ----------
def generate_chromosome(n_jobs, n_ops):
    """Each job repeated n_ops times"""
    chromosome = [j for j in range(n_jobs) for _ in range(n_ops)]
    random.shuffle(chromosome)
    return chromosome

def crossover(parent1, parent2):
    """Job Order Crossover"""
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    hole = parent1[a:b]
    child = [j for j in parent2 if j not in hole]
    return child[:a] + hole + child[a:]

def mutate(chromosome, p=0.1):
    for i in range(len(chromosome)):
        if random.random() < p:
            j = random.randint(0, len(chromosome) - 1)
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

# ---------- GA loop ----------
def genetic_algorithm(jobs, pop_size=50, generations=100, mutation_rate=0.1):
    n_jobs = len(jobs)
    n_ops = len(jobs[0])
    population = [generate_chromosome(n_jobs, n_ops) for _ in range(pop_size)]

    best_chrom = None
    best_fit = float('inf')

    for gen in range(generations):
        scored = []
        for chrom in population:
            fit, _ = evaluate_schedule(jobs, chrom)
            scored.append((fit, chrom))
        scored.sort(key=lambda x: x[0])
        population = [c for _, c in scored]

        if scored[0][0] < best_fit:
            best_fit = scored[0][0]
            best_chrom = scored[0][1]
        print(f"Gen {gen}: Best makespan = {best_fit}")

        # Selection + reproduction
        new_pop = population[:5]  # elitism
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(population[:25], 2)
            child = crossover(p1, p2)
            mutate(child, mutation_rate)
            new_pop.append(child)
        population = new_pop

    best_fit, best_schedule = evaluate_schedule(jobs, best_chrom)
    return best_chrom, best_schedule, best_fit

def plot_gantt(schedule, n_machines):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.get_cmap('tab10', len(set(op['job'] for op in schedule)))

    for op in schedule:
        j, m, s, e = op['job'], op['machine'], op['start'], op['end']
        ax.barh(m, e - s, left=s, color=colors(j), edgecolor='black')
        ax.text(s + (e - s)/2, m, f"J{j}", va='center', ha='center', color='white', fontsize=8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_yticks(range(n_machines))
    ax.invert_yaxis()
    ax.set_title("Corrected Job Shop Schedule")
    plt.tight_layout()
    plt.show()





n_jobs, n_machines, jobs = load_jssp_instance("datasets/jobshop1.txt", "ft06")
print("Jobs:", n_jobs, "Machines:", n_machines)
for i, job in enumerate(jobs):
    print(f"Job {i+1}: {job}")

best_chrom, best_schedule, best_fit = genetic_algorithm(jobs, generations=50)
print("\nBest makespan:", best_fit)

plot_gantt(best_schedule, n_machines)
