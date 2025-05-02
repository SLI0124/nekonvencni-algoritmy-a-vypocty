# Celulární automat lesního požáru

## Princip celulárního automatu

Implementace celulárního automatu simulujícího šíření lesního požáru se stavy:

```python
class CellState(Enum):
    """Výčtový typ pro možné stavy buněk v automatu lesního požáru"""
    EMPTY = 0  # Prázdné místo
    TREE = 1  # Strom
    FIRE = 2  # Hořící strom
    BURNT = 3  # Spáleniště
```

## Pravidla automatu

Jádro algoritmu simulujícího šíření požáru:

```python
def update_forest(self, forest, neighborhood_func):
    """Aplikace přechodových pravidel celulárního automatu lesního požáru"""
    new_forest = forest.copy()

    for i in range(self.size):
        for j in range(self.size):
            current_state = forest[i, j]

            # Prázdné místo nebo spáleniště - může vyrůst nový strom
            if current_state == CellState.EMPTY.value or current_state == CellState.BURNT.value:
                if np.random.random() < self.p:
                    new_forest[i, j] = CellState.TREE.value

            # Strom - může se vznítit od sousedů nebo spontánně
            elif current_state == CellState.TREE.value:
                neighbors = neighborhood_func(i, j, forest)
                if CellState.FIRE.value in neighbors:
                    # Šíření požáru od sousedů
                    new_forest[i, j] = CellState.FIRE.value
                elif np.random.random() < self.f:
                    # Spontánní vznícení bleskem
                    new_forest[i, j] = CellState.FIRE.value

            # Oheň - vždy přechází na spáleniště v dalším kroku
            elif current_state == CellState.FIRE.value:
                new_forest[i, j] = CellState.BURNT.value

    return new_forest
```

## Typy okolí

Chování automatu závisí na definici okolí buňky:

### Von Neumannovo okolí (4 sousedi)

```python
def get_von_neumann_neighbors(self, i, j, forest):
    """Získání Von Neumannova okolí - 4 sousedi v kardinálních směrech"""
    return [
        forest[(i - 1) % self.size, j],
        forest[(i + 1) % self.size, j],
        forest[i, (j - 1) % self.size],
        forest[i, (j + 1) % self.size]
    ]
```

### Moorovo okolí (8 sousedů)

```python
def get_moore_neighbors(self, i, j, forest):
    """Získání Moorova okolí - 8 sousedů včetně diagonálních"""
    return [
        forest[(i - 1) % self.size, (j - 1) % self.size],
        forest[(i - 1) % self.size, j],
        forest[(i - 1) % self.size, (j + 1) % self.size],
        forest[i, (j - 1) % self.size],
        forest[i, (j + 1) % self.size],
        forest[(i + 1) % self.size, (j - 1) % self.size],
        forest[(i + 1) % self.size, j],
        forest[(i + 1) % self.size, (j + 1) % self.size]
    ]
```

### Hexagonální okolí (6 sousedů)

```python
def get_hexagonal_neighbors(self, i, j, forest):
    """Získání hexagonálního okolí - 6 sousedů v šestiúhelníkovém uspořádání"""
    # Simulace hexagonální mřížky pomocí odlišného vzoru pro sudé a liché řádky
    if i % 2 == 0:
        return [
            forest[(i - 1) % self.size, j],
            forest[(i - 1) % self.size, (j + 1) % self.size],
            forest[i, (j + 1) % self.size],
            forest[(i + 1) % self.size, j],
            forest[(i + 1) % self.size, (j + 1) % self.size],
            forest[i, (j - 1) % self.size]
        ]
    else:
        return [
            forest[(i - 1) % self.size, (j - 1) % self.size],
            forest[(i - 1) % self.size, j],
            forest[i, (j + 1) % self.size],
            forest[(i + 1) % self.size, (j - 1) % self.size],
            forest[(i + 1) % self.size, j],
            forest[i, (j - 1) % self.size]
        ]
```

Poznámka: U všech typů okolí se používá operátor modulo (`%`) pro zajištění toroidální topologie, kde okraje mřížky jsou
propojené.

## Inicializace lesa

Počáteční stav lesa s náhodně rozmístěnými stromy a ohnisky požáru:

```python
def initialize_forest(self):
    """Inicializace lesa s náhodně rozmístěnými stromy podle zadané hustoty"""
    forest = np.zeros((self.size, self.size), dtype=int)
    forest[np.random.random((self.size, self.size)) < self.density] = CellState.TREE.value

    # Přidání několika počátečních ohnisek požáru
    for _ in range(3):
        i, j = np.random.randint(0, self.size, 2)
        if forest[i, j] == CellState.TREE.value:
            forest[i, j] = CellState.FIRE.value

    return forest
```

## Význam parametrů

Algoritmus ovlivňují tři klíčové parametry:

- **p**: Pravděpodobnost růstu nového stromu (výchozí: 0.05)
- **f**: Pravděpodobnost spontánního vzniku požáru (výchozí: 0.001)
- **density**: Počáteční hustota stromů v lese (výchozí: 0.5)

## Pozorované jevy

Simulace ukazuje několik důležitých charakteristik šíření požáru:

1. **Vliv typu okolí**: Různé typy okolí vedou k odlišným vzorům šíření požáru:
    - Von Neumannovo okolí: Šíření v kříži
    - Moorovo okolí: Kruhové šíření
    - Hexagonální okolí: Šestiúhelníkové šíření

2. **Dynamická rovnováha**: Při vhodných parametrech (p a f) se les dostává do dynamické rovnováhy, kde rychlost růstu
   nových stromů odpovídá rychlosti jejich zničení požárem.

3. **Emergentní vzory**: V průběhu simulace vznikají komplexní vzory šíření požáru, které nejsou explicitně definovány v
   pravidlech automatu.

## Výsledky

Simulace vytváří pro každý typ okolí charakteristické vzory šíření požáru:

### Von Neumannovo okolí

![Von Neumannovo okolí](results/forest_fire_von_neumann.gif)

### Moorovo okolí

![Moorovo okolí](results/forest_fire_moore.gif)

### Hexagonální okolí

![Hexagonální okolí](results/forest_fire_hexagonal.gif)
