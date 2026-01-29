import re
import matplotlib.pyplot as plt

# Twoje dane wejściowe
data_text = """




============== pop 1000
N100M5 pop 1000 gen 10000 mig-int 100
17,06
N1000M5 pop 1000 gen 10000 mig-int 100
110,75
N10000M5 pop 1000 gen 10000 mig-int 100

N100M20 pop 1000 gen 10000 mig-int 100
N1000M20 pop 1000 gen 10000 mig-int 100
N10000M20 pop 1000 gen 10000 mig-int 100

N100M5 pop 1000 gen 10000 mig-int 1
17,91
N1000M5 pop 1000 gen 10000 mig-int 1
109,99
N10000M5 pop 1000 gen 10000 mig-int 1


N100M20 pop 1000 gen 10000 mig-int 1
N1000M20 pop 1000 gen 10000 mig-int 1
N10000M20 pop 1000 gen 10000 mig-int 1


============== pop 100
N100M5 pop 100 gen 10000
1,92
N1000M5 pop 100 gen 10000
8,62
N10000M5 pop 100 gen 10000 mig-int 100
83,60

N100M20 pop 100 gen 10000 mig-int 100
N1000M20 pop 100 gen 10000 mig-int 100
N10000M20 pop 100 gen 10000 mig-int 100

N100M5 pop 100 gen 10000 mig-int 1
2,44
N1000M5 pop 100 gen 10000 mig-int 1
9,76
N10000M5 pop 100 gen 10000 mig-int 1
90,83







"""

# --- KONFIGURACJA ---
# True = osobne wykresy dla każdej populacji
# False = jeden wspólny wiersz z wykresami (stary widok)
SINGLE_CHARTS = True
# ---------------------

# 1. Parsowanie danych
pattern = r"N(\d+)M(\d+) pop (\d+) gen (\d+)\s*\n\s*([\d,]+)"
matches = re.findall(pattern, data_text)

records = []
for m in matches:
    records.append({
        'N': int(m[0]),
        'M': int(m[1]),
        'pop': int(m[2]),
        'time': float(m[4].replace(',', '.'))
    })

pops = sorted(list(set(r['pop'] for r in records)), reverse=True)
ms = sorted(list(set(r['M'] for r in records)))

# 2. Rysowanie wykresów
if SINGLE_CHARTS:
    # Generowanie osobnych plików/okien dla każdej populacji
    for p in pops:
        plt.figure(figsize=(8, 6))
        for m in ms:
            subset = [r for r in records if r['pop'] == p and r['M'] == m]
            subset.sort(key=lambda x: x['N'])

            x_vals = [r['N'] for r in subset]
            y_vals = [r['time'] for r in subset]

            plt.plot(x_vals, y_vals, marker='o', label=f'M={m}')

        plt.title(f'Wpływ N i M na czas (Populacja: {p})')
        plt.xlabel('Parametr N')
        plt.ylabel('Czas wykonania [s]')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()

        # Zapisywanie każdego wykresu z unikalną nazwą
        filename = f'wykres_pop_{p}.png'
        plt.savefig(filename)
        print(f"Zapisano: {filename}")
        plt.show()

else:
    # Stary tryb: Wszystko w jednym rzędzie
    fig, axes = plt.subplots(1, len(pops), figsize=(15, 5), squeeze=False)
    axes = axes.flatten()

    for i, p in enumerate(pops):
        ax = axes[i]
        for m in ms:
            subset = [r for r in records if r['pop'] == p and r['M'] == m]
            subset.sort(key=lambda x: x['N'])
            x_vals = [r['N'] for r in subset]
            y_vals = [r['time'] for r in subset]
            ax.plot(x_vals, y_vals, marker='o', label=f'M={m}')

        ax.set_title(f'Wielkość populacji: {p}')
        ax.set_xlabel('Parametr N')
        ax.set_yscale('log')  # Skala logarytmiczna dla osi Y (100, 1000, 10000)
        ax.set_xscale('log')  # Skala logarytmiczna dla osi X (100, 1000, 10000)
        # ax.set_xticks([100, 1000, 10000])
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.set_ylabel('Czas wykonania [s]')
        ax.legend()

    plt.suptitle('Porównanie wpływu parametrów', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('wykres_zbiorczy.png')
    plt.show()
