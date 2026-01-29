import re
import matplotlib.pyplot as plt

# =====================
# Dane wejściowe
# =====================
data_text = """

============MAX 1

N100M5 pop 100 gen 1000
0,15
N1000M5 pop 100 gen 1000
1,11

N100M20 pop 100 gen 1000
0,25
N1000M20 pop 100 gen 1000
2,31

N100M100 pop 100 gen 1000
1,10
N1000M100 pop 100 gen 1000
11,28

N100M1000 pop 100 gen 1000
11,34
N1000M1000 pop 100 gen 1000
124,23


============MAX 2
N100M5 pop 100 gen 1000
0,15
N1000M5 pop 100 gen 1000
1,01

N100M20 pop 100 gen 1000
0,21
N1000M20 pop 100 gen 1000
1,73

N100M100 pop 100 gen 1000
0,68
N1000M100 pop 100 gen 1000
6,50

N100M1000 pop 100 gen 1000
6,07
N1000M1000 pop 100 gen 1000
63,51

============MAX 4
N100M5 pop 100 gen 1000
0,14
N1000M5 pop 100 gen 1000
0,88

N100M20 pop 100 gen 1000
0,17
N1000M20 pop 100 gen 1000
1,31

N100M100 pop 100 gen 1000
0,42
N1000M100 pop 100 gen 1000
3,81

N100M1000 pop 100 gen 1000
3,18
N1000M1000 pop 100 gen 1000
32,81

============MAX 6
N100M5 pop 100 gen 1000
0,14
N1000M5 pop 100 gen 1000
0,88

N100M20 pop 100 gen 1000
0,16
N1000M20 pop 100 gen 1000
1,14

N100M100 pop 100 gen 1000
0,34
N1000M100 pop 100 gen 1000
2,97

N100M1000 pop 100 gen 1000
2,32
N1000M1000 pop 100 gen 1000
24,03

============MAX 8
N100M5 pop 100 gen 1000
0,15
N1000M5 pop 100 gen 1000
0,90

N100M20 pop 100 gen 1000
0,17
N1000M20 pop 100 gen 1000
1,20

N100M100 pop 100 gen 1000
0,34
N1000M100 pop 100 gen 1000
2,81

N100M1000 pop 100 gen 1000
2,16
N1000M1000 pop 100 gen 1000
20,93

============MAX (12)
N100M5 pop 100 gen 1000
0,15
N1000M5 pop 100 gen 1000
0,85

N100M20 pop 100 gen 1000
0,17
N1000M20 pop 100 gen 1000
1,09

N100M100 pop 100 gen 1000
0,31
N1000M100 pop 100 gen 1000
2,27

N100M1000 pop 100 gen 1000
1,66
N1000M1000 pop 100 gen 1000
15,81
"""

# =====================
# Parsowanie danych
# =====================
block_pattern = r"=+MAX\s*\(?(\d+)\)?(.*?)(?==+MAX|\Z)"
entry_pattern = r"N(\d+)M(\d+)\s+pop\s+(\d+)\s+gen\s+\d+\s*\n\s*([\d,]+)"

records = []

for block in re.findall(block_pattern, data_text, re.S):
    cpu = int(block[0])
    content = block[1]

    for e in re.findall(entry_pattern, content):
        records.append({
            "cpu": cpu,
            "N": int(e[0]),
            "M": int(e[1]),
            "time": float(e[3].replace(",", "."))
        })

# =====================
# Wykresy
# =====================
Ns = sorted(set(r["N"] for r in records))
cpus = sorted(set(r["cpu"] for r in records))

fig, axes = plt.subplots(1, len(Ns), figsize=(14, 5), sharey=True)

for i, N in enumerate(Ns):
    ax = axes[i]

    for cpu in cpus:
        subset = [r for r in records if r["N"] == N and r["cpu"] == cpu]
        subset.sort(key=lambda x: x["M"])

        x = [r["M"] for r in subset]
        y = [r["time"] for r in subset]

        ax.plot(x, y, marker="o", label=f"{cpu} wątków")

    ax.set_title(f"N = {N}")
    ax.set_xlabel("Parametr M")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.7)

    if i == 0:
        ax.set_ylabel("Czas wykonania [s]")

    ax.legend()

plt.suptitle("Zależność czasu od parametru M dla różnych ilości wątków CPU", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("czas_vs_M_cpu.png")
plt.show()
