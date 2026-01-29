import re
import matplotlib.pyplot as plt

# =====================
# Dane wejściowe
# =====================
data_text = """
============MAX 1
N100M10 pop 100 gen 1000
0,18
N1000M10 pop 100 gen 1000
1,48
N10000M10 pop 100 gen 1000
19,16

============MAX 2
N100M10 pop 100 gen 1000
0,16
N1000M10 pop 100 gen 1000
1,25
N10000M10 pop 100 gen 1000
14,58

============MAX 4
N100M10 pop 100 gen 1000
0,14
N1000M10 pop 100 gen 1000
1,04
N10000M10 pop 100 gen 1000
11,22

============MAX 6
N100M10 pop 100 gen 1000
0,15
N1000M10 pop 100 gen 1000
0,96
N10000M10 pop 100 gen 1000
9,52

============MAX 8
N100M10 pop 100 gen 1000
0,15
N1000M10 pop 100 gen 1000
1,00
N10000M10 pop 100 gen 1000
10,3

============MAX (12)
N100M10 pop 100 gen 1000
0,16
N1000M10 pop 100 gen 1000
0,94
N10000M10 pop 100 gen 1000
9,55
"""

# =====================
# Parsowanie danych
# =====================
block_pattern = r"=+MAX\s*\(?(\d+)\)?(.*?)(?==+MAX|\Z)"
entry_pattern = r"N(\d+)M(\d+)\s+pop\s+\d+\s+gen\s+\d+\s*\n\s*([\d,]+)"

records = []

for cpu, content in re.findall(block_pattern, data_text, re.S):
    cpu = int(cpu)
    for e in re.findall(entry_pattern, content):
        records.append({
            "cpu": cpu,
            "N": int(e[0]),
            "M": int(e[1]),
            "time": float(e[2].replace(",", "."))
        })

# =====================
# Wykres
# =====================
cpus = sorted(set(r["cpu"] for r in records))
fig, ax = plt.subplots(figsize=(7, 5))

for cpu in cpus:
    subset = [r for r in records if r["cpu"] == cpu]
    subset.sort(key=lambda x: x["N"])

    x = [r["N"] for r in subset]
    y = [r["time"] for r in subset]

    ax.plot(x, y, marker="o", label=f"{cpu} wątków")

ax.set_xlabel("Parametr N")
ax.set_ylabel("Czas wykonania [s]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend(title="CPU")

plt.title("Zależność czasu od N dla różnych liczby wątków CPU (M = 10)")
plt.tight_layout()
plt.savefig("czas_vs_N_cpu.png")
plt.show()
