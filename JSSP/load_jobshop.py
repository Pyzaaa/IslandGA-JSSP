def load_jssp_instance(filename, instance_name):
    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    # 1️⃣ Find where the instance starts
    start_idx = None
    for i, line in enumerate(lines):
        if line.lower().startswith(f"instance {instance_name.lower()}"):
            start_idx = i
            break
    if start_idx is None:
        raise ValueError(f"Instance {instance_name} not found in file.")

    # 2️⃣ Skip the divider line that follows "instance ..."
    i = start_idx + 1
    while i < len(lines) and (not lines[i].strip().isdigit()):
        if lines[i].startswith("++++++++++++++++"):
            # skip this divider and continue
            i += 1
            continue
        # first real data block begins with description or numbers
        if lines[i].strip() and not lines[i].startswith('+') and not lines[i].lower().startswith('instance'):
            break
        i += 1

    # 3️⃣ Collect data lines until next divider
    instance_data = []
    for line in lines[i:]:
        if line.startswith('+'):
            break
        if line:
            instance_data.append(line.strip())

    # 4️⃣ Find first numeric line (n_jobs, n_machines)
    numeric_idx = None
    for idx, line in enumerate(instance_data):
        parts = line.split()
        if len(parts) >= 2 and all(p.isdigit() for p in parts[:2]):
            numeric_idx = idx
            break
    if numeric_idx is None:
        raise ValueError(f"Could not find numeric line for {instance_name}")

    n_jobs, n_machines = map(int, instance_data[numeric_idx].split())

    # 5️⃣ Parse job lines
    jobs = []
    for line in instance_data[numeric_idx + 1: numeric_idx + 1 + n_jobs]:
        nums = list(map(int, line.split()))
        job = [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]
        jobs.append(job)

    return n_jobs, n_machines, jobs

n_jobs, n_machines, jobs = load_jssp_instance("datasets/jobshop1.txt", "ft06")
print("Jobs:", n_jobs, "Machines:", n_machines)
for i, job in enumerate(jobs):
    print(f"Job {i+1}: {job}")