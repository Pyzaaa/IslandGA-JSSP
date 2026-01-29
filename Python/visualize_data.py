import re
import matplotlib.pyplot as plt

# Twoje dane wejściowe w formie tekstu
data_text = """

N100M2000 pop 100 gen 1000
2,81
N1000M2000 pop 100 gen 1000
28,69

============== pop 1000
N100M5 pop 1000 gen 1000
1,54
N500M5 pop 1000 gen 1000
6,14
N1000M5 pop 1000 gen 1000
11,80
N5000M5 pop 1000 gen 1000
60,12
N10000M5 pop 1000 gen 1000
121,60
147% // 12%

N100M10 pop 1000 gen 1000
1,61
N500M10 pop 1000 gen 1000
6,20
N1000M10 pop 1000 gen 1000
11,80
N5000M10 pop 1000 gen 1000
58,40
N10000M10 pop 1000 gen 1000
115,48

N100M20 pop 1000 gen 1000
1,79
156% // 13%
N500M20 pop 1000 gen 1000
6,89
N1000M20 pop 1000 gen 1000
12,92
N5000M20 pop 1000 gen 1000
N10000M20 pop 1000 gen 1000
135,47
234% // 19%

N100M50 pop 1000 gen 1000
2,16
N500M50 pop 1000 gen 1000
8,91
N1000M50 pop 1000 gen 1000
17,21
N5000M50 pop 1000 gen 1000
N10000M50 pop 1000 gen 1000


N100M100 pop 1000 gen 1000
2,96
N500M100 pop 1000 gen 1000
N1000M100 pop 1000 gen 1000
24,18
N5000M100 pop 1000 gen 1000
127,45
N10000M100 pop 1000 gen 1000

N1000M1000 pop 1000 gen 1000
153,01


============== pop 100
N100M5 pop 100 gen 1000
0,16
N500M5 pop 100 gen 1000
0,51
N1000M5 pop 100 gen 1000
0,88
N5000M5 pop 100 gen 1000
3,87
N10000M5 pop 100 gen 1000
8,26
178% // 14%

N100M10 pop 100 gen 1000
0,19
N500M10 pop 100 gen 1000
0,72
N1000M10 pop 100 gen 1000
1,02
N5000M10 pop 100 gen 1000
4,28
N10000M10 pop 100 gen 1000
9,76
199% // 16%

N100M20 pop 100 gen 1000
0,17
N500M20 pop 100 gen 1000
N1000M20 pop 100 gen 1000
1,08
N5000M20 pop 100 gen 1000
N10000M20 pop 100 gen 1000
11,26
358% // 29%


N100M100 pop 100 gen 1000
0,30
N500M100 pop 100 gen 1000
N1000M100 pop 100 gen 1000
2,33
N5000M100 pop 100 gen 1000
11,65
N10000M100 pop 100 gen 1000
24,31
611% // 50%


N100M1000 pop 100 gen 1000
1,54
760% // 63%
N1000M1000 pop 100 gen 1000
15,83
790% przyspieszenie // 66% Efektywność





"""

# 1. Parsowanie danych za pomocą wyrażeń regularnych
# Szukamy wzorca: N...M... pop ... gen ... a potem w nowej linii liczby
pattern = r"N(\d+)M(\d+) pop (\d+) gen (\d+)\s*\n\s*([\d,]+)"
matches = re.findall(pattern, data_text)

records = []
for m in matches:
    records.append({
        'N': int(m[0]),
        'M': int(m[1]),
        'pop': int(m[2]),
        'time': float(m[4].replace(',', '.'))  # zamiana przecinka na kropkę dla float
    })

# 2 pop
Ns = sorted(set(r['N'] for r in records))
ms = sorted(set(r['M'] for r in records))


fig, axes = plt.subplots(1, len(Ns), figsize=(18, 5), sharey=True)

for i, N in enumerate(Ns):
    ax = axes[i]
    for m in ms:
        subset = [r for r in records if r['N'] == N and r['M'] == m]
        subset.sort(key=lambda x: x['pop'])

        x_vals = [r['pop'] for r in subset]
        y_vals = [r['time'] for r in subset]

        if x_vals:
            ax.plot(x_vals, y_vals, marker='o', label=f'M={m}')

    ax.set_title(f'N = {N}')
    ax.set_xlabel('Populacja')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.7)

    if i == 0:
        ax.set_ylabel('Czas [s]')
    ax.legend()

plt.suptitle('Wpływ parametrów N, M i pop na czas wykonania algorytmu', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Zapis i pokazanie wykresu
plt.savefig('wykres_algorytmu-pop.png')
plt.show()

# =====================
# Przygotowanie osi
# =====================
pops = sorted(set(r['pop'] for r in records), reverse=True)
Ns = sorted(set(r['N'] for r in records))

fig, axes = plt.subplots(1, len(pops), figsize=(18, 5), sharey=True)

# =====================
# Rysowanie wykresów
# =====================
for i, p in enumerate(pops):
    ax = axes[i]

    for N in Ns:
        subset = [r for r in records if r['pop'] == p and r['N'] == N]
        subset.sort(key=lambda x: x['M'])

        x_vals = [r['M'] for r in subset]
        y_vals = [r['time'] for r in subset]

        if x_vals:  # zabezpieczenie przed pustymi danymi
            ax.plot(x_vals, y_vals, marker='o', label=f'N={N}')

    ax.set_title(f'Populacja = {p}')
    ax.set_xlabel('Parametr M')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, linestyle='--', alpha=0.7)

    if i == 0:
        ax.set_ylabel('Czas wykonania [s]')

    ax.legend()

# Zapis i pokazanie wykresu
plt.savefig('wykres_algorytmu-M.png')
plt.show()

# 2. Przygotowanie struktury do wykresu
pops = sorted(list(set(r['pop'] for r in records)), reverse=True)
ms = sorted(list(set(r['M'] for r in records)))

# Tworzenie subpłotów (jeden dla każdej populacji)
fig, axes = plt.subplots(1, len(pops), figsize=(15, 5), sharey=True)

for i, p in enumerate(pops):
    ax = axes[i]
    for m in ms:
        # Filtrowanie danych dla konkretnego pop i M
        subset = [r for r in records if r['pop'] == p and r['M'] == m]

        subset.sort(key=lambda x: x['N'])  # Sortowanie po N
        x_vals = [r['N'] for r in subset]

        # subset.sort(key=lambda x: x['M'])
        # x_vals = [r['M'] for r in subset]

        #subset.sort(key=lambda x: x['pop'])
        #x_vals = [r['pop'] for r in subset]

        y_vals = [r['time'] for r in subset]

        ax.plot(x_vals, y_vals, marker='o', label=f'M={m}')

    ax.set_title(f'Wielkość populacji: {p}')
    ax.set_xlabel('Parametr N')
    ax.set_yscale('log')  # Skala logarytmiczna dla osi Y (100, 1000, 10000)
    ax.set_xscale('log')  # Skala logarytmiczna dla osi X (100, 1000, 10000)
    # ax.set_xticks([100, 1000, 10000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.grid(True, linestyle='--', alpha=0.7)

    if i == 0:
        ax.set_ylabel('Czas wykonania [s]')
    ax.legend()

plt.suptitle('Wpływ parametrów N, M i pop na czas wykonania algorytmu', fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Zapis i pokazanie wykresu
plt.savefig('wykres_algorytmu.png')
plt.show()