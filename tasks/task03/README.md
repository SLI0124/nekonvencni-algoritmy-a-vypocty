# Hopfieldova Síť

## Úvod

Během implementace jsem výcházel z [dokumentu](https://michaelmachu.eu/data/pdf/navy/navy-task3.pdf) pana učitele Machu,
který popisuje Hopfieldovu síť a její fungování. V dokumentu se nacházelo vše, ať už po teoretické vlastnosti i
implementaci. Více méně to je transkript tohoto dokumentu s vlastními poznámkami a úpravami.

## Implementace

### Reprezentace vzoru

V kódu je vzor reprezentován pomocnou třídou `Pattern`, která umožňuje konverzi binární matice na vektor vhodný pro
výpočty v Hopfieldově síti. Při inicializaci třídy dochází k převodu hodnot 0 na -1 a 1 zůstává beze změny. Jako vstup
má zadanou matci, kterou si vezme z plátna Tkinteru.

```python
class Pattern:
    def __init__(self, matrix):
        self.matrix = matrix
        self.vector = []
        for row in matrix:
            for value in row:
                if value == 0:
                    self.vector.append(-1)
                else:
                    self.vector.append(1)

        self.weight_matrix = self._calculate_weight_matrix()
```

Vektor `self.vector` slouží k uložení jednorozměrné reprezentace vzoru matice, která se následně využívá při výpočtu
váhové matice.

#### Výpočet váhové matice

Váhová matice alfou a omegou Hopfieldovy sítě a je určena metodou `_calculate_weight_matrix()`. Tato metoda realizuje
její výpočet:

```python
def _calculate_weight_matrix(self):
    n = len(self.vector)
    weight_matrix = np.zeros((n, n), dtype=np.int32)

    # Manuální výpočet váhové matice
    for i in range(n):
        for j in range(n):
            # Nastavíme váhu pouze pokud i != j (diagonála zůstane 0)
            if i != j:
                weight_matrix[i, j] = self.vector[i] * self.vector[j]

    return weight_matrix
```

K transpozici matice dochází při inicializaci proměnné `weight_matrix`, kdy se nejprve zjistí její rozměr `n` a
následně se vytvoří matice velikosti $n \times n$ obsahující pouze nuly:

```python
n = len(self.vector)
weight_matrix = np.zeros((n, n), dtype=np.int32)
```

Následně se provede dvojitý cyklus přes všechny prvky matice, každý s každým, kde se nastaví váha mezi "neurony" $i$
a $j$ na hodnotu součinu odpovídajících neuronů ve vzoru. V následujícím kroku se nastaví hodnoty na diagonále matice na
nulu, já to provedl již zde.

```python
for i in range(n):
    for j in range(n):
        if i != j:
            weight_matrix[i, j] = self.vector[i] * self.vector[j]
```

Tento výpočet implementuje pravidlo učení, kdy váhy mezi neurony se posilují, pokud jsou oba neurony ve stejném
stavu (oba +1 nebo oba -1, což dává součin +1), a oslabují, pokud jsou v opačných stavech (jeden +1 a druhý -1, což dává
součin -1).

#### Porovnání Vzorů

Pomocná a poslední metoda dřídy `Pattern` slouží k porovnání dvou vzorů slouží metoda `__eq__`. Díky této metodě lze
snadno ověřit, zda rekonstruovaný vzor odpovídá některému z uložených vzorů, aby se nepřidal podruhé a jednotlivé
vzory se nepřekrývaly.

```python
def __eq__(self, other):
    if not isinstance(other, Pattern):
        return False
    return np.array_equal(self.matrix, other.matrix)
```

### Hopfieldova síť

Hopfieldova síť je implementována jako třída `HopfieldNetwork`, která obsahuje metody pro učení a rekonstrukci vzorů.
Jako vstup má velikost matice, která je čtvercová, a volitelný parametr `stable_threshold`, který určuje počet
stabilních iterací pro asynchronní aktualizaci neuronů. To si vysvětlíme později.

Dále si třída v sobě uchovává velikost dimenze matice `grid_size`, počet jednotlivých "neuronů" v matici`size`, váhovou
mřížku `weights` a seznam vzorů `patterns`. Váhová matice je inicializována jako nulová matice o velikosti $n \times n$,
stejně jako při reprezentaci vzoru v předchozí kapitole.

```python
class HopfieldNetwork:
    def __init__(self, grid_size, stable_threshold=5):
        self.grid_size = grid_size
        self.size = grid_size ** 2
        self.weights = np.zeros((self.size, self.size), np.int32)
        self.patterns = []
        self.stable_threshold = stable_threshold
```

#### Převod vektoru na matici

Pomocná jednoduchá funkce na zpětnž převod z vektoru na matici, která se používá při rekonstrukci vzorů po fázi učení.

```python3
def _vector_to_matrix(self, vector):
    ...
```

#### Přidání vzoru

Metoda `add_pattern` slouží k přidání vzoru do sítě. Pokud je vzor již v seznamu vzorů, metoda vrátí `False`,
jinak **přidá** váhovou mřížku vzoru k váhové mřížce sítě a přidá vzor do seznamu vzorů. Skutečná váhová matice sítě se
neustále aktualizace a může nabývat i z rozmezí mimo -1 až 1. Do vzorů a boolean návratové hodnoty si ji ukálád pouze
pro uživatelské rozhraní.

```python
def add_pattern(self, pattern):
    if pattern in self.patterns:
        return False

    self.weights += pattern.weight_matrix
    self.patterns.append(deepcopy(pattern))
    return True
```

#### Učení vzoru

##### Synchronní učení

Metoda začíná vytvořením kopie vektoru vstupního vzoru, aby se zachoval původní vzor. Následně se vypočítá aktivace
všech neuronů pomocí násobení váhové matice sítě a vstupního vektoru. Pro každý neuron se pak určí nový stav na základě
aktivace - pokud je aktivace kladná, neuron získá hodnotu 1, jinak -1. Výsledný vektor se převede zpět z 1D vektroru na
2D matici pomocí metody `_vector_to_matrix()`, která byla popsána výše.

```python
def synchronous_recovery(self, input_pattern):
    vector = deepcopy(input_pattern.vector)

    activation = np.dot(self.weights, vector)
    new_vector = np.zeros_like(vector)
    for i in range(len(activation)):
        if activation[i] > 0:
            new_vector[i] = 1
        else:
            new_vector[i] = -1

    result_matrix = self._vector_to_matrix(new_vector)
    return Pattern(result_matrix)
```

##### Asynchronní učení

Asynchronní metoda obnovování vzoru postupuje iterativně a aktualizuje neurony jeden po druhém v náhodném pořadí. Na
rozdíl od synchronní metody, která aktualizuje všechny neurony najednou, asynchronní metoda postupně prochází neurony a
aktualizuje jejich stavy na základě aktuálního stavu ostatních neuronů.

```python
def asynchronous_recovery(self, input_pattern):
    vector = deepcopy(input_pattern.vector)
    stable_count = 0
    prev_vector = None
```

Metoda nejprve vytvoří kopii vektoru vstupního vzoru a inicializuje počítadlo stabilních iterací a proměnnou pro
předchozí stav vektoru.

```python
while stable_count < self.stable_threshold:
    indices = np.random.permutation(self.size)
```

Hlavní cyklus probíhá, dokud nedosáhneme dostatečného počtu stabilních iterací. V každé iteraci se vytvoří náhodné
pořadí indexů neuronů. Tady je teď důležitá promněnná `stable_count`, která určuje, kolikrát po sobě se vzor nezměnil a
na základě toho se rozhoduje, zda se má cyklus ukončit nebo ne.

```python
for i in indices:
    activation = np.dot(self.weights[i], vector)
    vector[i] = np.sign(activation)
```

Tady je alfa a omega asynchronní metody. Pro každý neuron se spočítá aktivace pomocí váhové matice a aktuálního vektoru
pomocí skalárního součinu. Na základě aktivace se pak určí nový stav neuronu pomocí funkce `np.sign()`, která vrací
hodnoty následovně:

$V_i = \begin{cases}
+1 & \text{pokud } \sum_{j} W_{ij} V_i > 0 \\
-1 & \text{jinak }
\end{cases}$

Potom se pouze zkontroluje, zda se vzor nezměnil. Pokud ano, tak se počítadlo stabilních iterací zvýší o 1, jinak se
nastaví na 0. Tímto způsobem se sleduje, kolikrát po sobě se vzor nezměnil, což naznačuje stabilitu vzoru.

```python
if prev_vector is not None and np.array_equal(vector, prev_vector):
    stable_count += 1
else:
    stable_count = 0

prev_vector = deepcopy(vector)

result_matrix = self._vector_to_matrix(vector)
return Pattern(result_matrix)
```

## Výsledky

Schopnost sítě byla testována na vzoru čísel dva a osm, a písmene X, jak bylo zmíněno v zadání. Moje řešení
podporuje více vzorů.

### Číslo 2

![Číslo 2](../../results/task03/number_2.gif)

### Číslo 8

![Číslo 8](../../results/task03/number_8.gif)

### Písmeno X

![Písmeno X](../../results/task03/letter_x.gif)