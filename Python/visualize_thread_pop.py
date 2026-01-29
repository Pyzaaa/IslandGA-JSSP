import re
import matplotlib.pyplot as plt

# =====================
# Dane wejściowe
# =====================
data_text = """
============MAX 1
N1000M10 pop 100 gen 1000
1,47
N1000M10 pop 1000 gen 1000
19,09
N1000M10 pop 10000 gen 1000
213,04

============MAX 2
N1000M10 pop 100 gen 1000
1,24
N1000M10 pop 1000 gen 1000
15,25
N1000M10 pop 10000 gen 1000
175,48

============MAX 4
N1000M10 pop 100 gen 1000
1,00
N1000M10 pop 1000 gen 1000
13,00
N1000M10 pop 10000 gen 1000
154,08

============MAX 6
N1000M10 pop 100 gen 1000
0,97
N1000M10 pop 1000 gen 1000
12,76
N1000M10 pop 10000 gen 1000
137,59

============MAX 8
N1000M10 pop 100 gen 1000
0,99
N1000M10 pop 1000 gen 1000
12,09
N1000M10 pop 10000 gen 1000
137,36

============MAX (12)
N1000M10 pop 100 gen 1000
0,92
N1000M10 pop 1000 gen 1000
11,8
N1000M10 pop 10000 gen 1000
130,34
"""

# =====================
# Parsowanie danych
# =====================
block_pattern = r"=+MAX\s*\(?(\d+)\)?(.*?)(?==+MAX|\Z)"
entry_pattern = r"N(\d+)M(\d+)\s+pop\s+(\d+)\s+gen\s+\d+\s*\n\s*([\d,]+)"

records = []

for cpu, content in re.findall(block_pattern, data_text, re.S):
    cpu = int(cpu)
    for e in re.findall(entry_pattern, content):
        records.append({
            "cpu": cpu,
            "N": int(e[0]),
            "M": int(e[1]),
            "pop": int(e[2]),
            "time": float(e[3].replace(",", "."))
        })

# =====================
# Wykres
# =====================
cpus = sorted(set(r["cpu"] for r in records))
fig, ax = plt.subplots(figsize=(7, 5))

for cpu in cpus:
    subset = [r for r in records if r["cpu"] == cpu]
    subset.sort(key=lambda x: x["pop"])

    x = [r["pop"] for r in subset]
    y = [r["time"] for r in subset]

    ax.plot(x, y, marker="o", label=f"{cpu} wątków")

ax.set_xlabel("Wielkość populacji")
ax.set_ylabel("Czas wykonania [s]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, linestyle="--", alpha=0.7)
ax.legend(title="CPU")

plt.title("Zależność czasu od populacji (N = 1000, M = 10)")
plt.tight_layout()
plt.savefig("czas_vs_populacja_cpu.png")
plt.show()
