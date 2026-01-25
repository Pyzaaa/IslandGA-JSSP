import re
import matplotlib.pyplot as plt

# Twoje dane wejściowe w formie tekstu
data_text = """



============== pop 20000
N100M5 pop 20000 gen 1000
3,52

N1000M5 pop 20000 gen 1000
53,29

N3000M5 pop 20000 gen 1000
180,56



============== pop 10000
N100M5 pop 10000 gen 1000
1,24

N1000M5 pop 10000 gen 1000
22,51

N10000M5 pop 10000 gen 1000
256,88

N100M10 pop 10000 gen 1000
1,79

N1000M10 pop 10000 gen 1000
27,93

N10000M10 pop 10000 gen 1000
318,115

N100M20 pop 10000 gen 1000
2,95

N1000M20 pop 10000 gen 1000
40,26

N10000M20 pop 10000 gen 1000
463,995


============== pop 5000
N100M5 pop 5000 gen 1000
0,44 s

N1000M5 pop 5000 gen 1000
7,21 s

N10000M5 pop 5000 gen 1000
95,2 s


N100M10 pop 5000 gen 1000
0,76 s

N500M10 pop 5000 gen 1000
4,33 s

N1000M10 pop 5000 gen 1000
10,05 s

N2000M10 pop 5000 gen 1000
21,81 s

N3000M10 pop 5000 gen 1000
34,38

N5000M10 pop 5000 gen 1000
59,34 s

N10000M10 pop 5000 gen 1000
122,62 s


N100M20 pop 5000 gen 1000
1,29 s

N1000M20 pop 5000 gen 1000
15,74 s

N2000M20 pop 5000 gen 1000
33,51

N5000M20 pop 5000 gen 1000
88,21

N10000M20 pop 5000 gen 1000
180,78 s


N100M50 pop 5000 gen 1000
3,24

N500M50 pop 5000 gen 1000
17,54

N1000M50 pop 5000 gen 1000
36,47 s

N2000M50 pop 5000 gen 1000
76,36

N10000M50 pop 5000 gen 1000
498,86


N100M100 pop 5000 gen 1000
6,38

N500M100 pop 5000 gen 1000

N1000M100 pop 5000 gen 1000
70,02

N2000M100 pop 5000 gen 1000







============== pop 1000
N100M5 pop 1000 gen 1000
0,28 s

N1000M5 pop 1000 gen 1000
2,09 s

N10000M5 pop 1000 gen 1000
28,35 s

N100M10 pop 1000 gen 1000
0,44 s

N1000M10 pop 1000 gen 1000
3,35 s

N10000M10 pop 1000 gen 1000
40,92 s

N100M20 pop 1000 gen 1000
0,69 s

N1000M20 pop 1000 gen 1000
6,14 s

N10000M20 pop 1000 gen 1000
69,82 s


============== pop 100
N100M5 pop 100 gen 1000
0,22

N1000M5 pop 100 gen 1000
1,12

N10000M5 pop 100 gen 1000
16,92 s

N100M10 pop 100 gen 1000
0,27 s

N1000M10 pop 100 gen 1000
1,69 s

N10000M10 pop 100 gen 1000
20,79 s

N100M20 pop 100 gen 1000
0,39 s

N1000M20 pop 100 gen 1000
2,78 s

N10000M20 pop 100 gen 1000
30,48 s

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
        y_vals = [r['time'] for r in subset]

        ax.plot(x_vals, y_vals, marker='o', label=f'M={m}')

    ax.set_title(f'Wielkość populacji: {p}')
    ax.set_xlabel('Parametr N')
    # ax.set_xscale('log')  # Skala logarytmiczna dla osi X (100, 1000, 10000)
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