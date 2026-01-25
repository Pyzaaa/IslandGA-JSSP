def wczytaj_dane(nazwa_pliku):
    """
    Wczytuje dane z pliku tekstowego.
    Pierwsza linia: liczba zadań i liczba maszyn
    Kolejne linie: czasy operacji dla każdego zadania
    """
    with open(nazwa_pliku, 'r') as f:
        linia = f.readline().strip()
        n, m = map(int, linia.split())
        czasy = []
        for _ in range(n):
            czasy.append(list(map(int, f.readline().strip().split())))
    return n, m, czasy

def wczytaj_best_order(nazwa_pliku):
    """
    Wczytuje permutację BEST ORDER z pliku tekstowego.
    Oczekiwany format: np. '1 3 2 4 8 5 11 6 7 9 10'
    """
    with open(nazwa_pliku, 'r') as f:
        linia = f.readline().strip()
        permutacja = list(map(int, linia.split()))
    return permutacja

def policz_makespan(permutacja, czasy):
    """
    Oblicza makespan dla podanej permutacji zadań w problemie flow shop.
    """
    n = len(permutacja)
    m = len(czasy[0])
    C = [[0]*m for _ in range(n)]

    for i, zadanie in enumerate(permutacja):
        zadanie_idx = zadanie - 1  # numeracja od 1 w pliku BEST ORDER
        for j in range(m):
            if i == 0 and j == 0:
                C[i][j] = czasy[zadanie_idx][j]
            elif i == 0:
                C[i][j] = C[i][j-1] + czasy[zadanie_idx][j]
            elif j == 0:
                C[i][j] = C[i-1][j] + czasy[zadanie_idx][j]
            else:
                C[i][j] = max(C[i-1][j], C[i][j-1]) + czasy[zadanie_idx][j]
    return C[-1][-1]

def main():
    plik_dane = r"../data/N1000M5.txt"  # zmień na swoją nazwę pliku
    plik_best_order = "../best.txt"   # plik z najlepszą permutacją

    n, m, czasy = wczytaj_dane(plik_dane)
    best_order = wczytaj_best_order(plik_best_order)

    makespan = policz_makespan(best_order, czasy)
    print(f"Permutacja z pliku BEST ORDER: {best_order}")
    print(f"Makespan: {makespan}")

if __name__ == "__main__":
    main()
