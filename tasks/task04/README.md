# Q-learning a Matice Herního Prostředí

## Herní Prostředí

Herní prostředí je reprezentováno jako matice, kde jednotlivé buňky mohou obsahovat různé typy políček s odlišnými
hodnotami skóre. Skóre určuje atraktivitu políčka pro agenta (myš). Implementace prostředí je řešena pomocí třídy
`GameEnvironmentMatrix`.

### Implementace herního prostředí

```python
class GameEnvironmentMatrix:
    def __init__(self, environment, score_dict):
        self.environment_dimension = len(environment)
        self.environment = environment
        self.score_dict = score_dict
        matrix_dimension = self.environment_dimension ** 2
        self.target_states = set()

        self.matrix = np.full(shape=(matrix_dimension, matrix_dimension), fill_value=-1)
```

### Jednotlivé atributy třídy

- **Environment dimension**: Rozměr herního prostředí (předpokládá se čtvercová matice). Pomocná proměnná pro
  inicializaci Tkinter plátna.
- **Environment**: 2D matice reprezentující herní prostředí.
- **Score dict**: Slovník obsahující skóre pro jednotlivé typy políček. Postupně načte hodnoty z textového souboru a
  uloží váhu překážek a odměn pro každé políčko.
    - Jednotlivé skóre jsem uložil do Tkinter třídy `QLearningApp`:

```python
self.scores = {'floor': 0, 'wall': -10, 'cat': -100, 'cheese': 100, 'mouse': 0}
``` 

- **Matrix dimension**: Rozměr matice, která bude reprezentovat herní prostředí. Je vypočítána jako druhá mocnina
  rozměru prostředí. Jedná se pouze o pomocnou proměnnou pro inicializaci herní matice.
- **Target states**: Množina cílových stavů, které agent může dosáhnout. Tyto stavy jsou reprezentovány jako indexy
  v matici. Jedná se o sýry, které agent hledá. Může jich být přece více než jeden.

Následně se v této tříde inicializují hodnoty pro jednotlivé stavy a akce. Tady jsem to mohl trochu zjednodušit, to
kontrolování mezí tam není úplně za potřebí.

```python
for row in range(self.environment_dimension):
    for column in range(self.environment_dimension):
        matrix_index = row * self.environment_dimension + column

        if (row - 1) >= 0:
            top = (row - 1) * self.environment_dimension + column
            self.matrix[matrix_index][top] = score_dict[environment[row - 1][column]]

        # a tak dále a tak dále

        # Označení cílových stavů
        if environment[row][column] == 'cheese':
            self.matrix[matrix_index][matrix_index] = score_dict[environment[row][column]]
            self.target_states.add(matrix_index)
```

## Q-learning Algoritmus

Q-learning je model volného učení vycházející z reinforcment learningu. Umožňuje učení, kdy agent (myš) nemá k dispozici
žádné informace o prostředí, ve kterém se pohybuje. Učení probíhá na základě odměn a trestů, které agent dostává
za své akce. Cílem je maximalizovat celkovou odměnu, kterou agent získá během svého pohybu prostředím. Výsledkem
je cesta, kterou agent zvolí, aby dosáhl cílového stavu (sýra) s co nejmenšími náklady.

### Implementace třídy `QLearning`

Třída `QLearning` implementuje následující kroky:

#### Inicializace Q-matice:

```python
class QLearning:
    def __init__(self, environment):
        self.q_matrix = np.zeros(environment.matrix.shape)
        self.environment = environment
```

Q-matice má dimenzi `environment.matrix.shape`. Druhou proměnnou je prostředí, které je předáno jako argument do
konstruktoru. Toto je instance třídy `GameEnvironmentMatrix`, která obsahuje informace o prostředí, ve kterém se agent
pohybuje. Získává se z Tkinteru, kde je zobrazeno herní prostředí.

#### Trénovací smyčka

Trénink začíná inicializací s rozměrem mřížky pro lepší indexaci v poli.

```python
def train(self, num_epochs, learning_rate):
    grid_size = self.environment.environment_dimension
```

V každé epoše agent startuje z náhodné pozice, myšleno směr, neboli následující pohyb.

```python
for epoch in range(num_epochs):
    start_position = self._get_valid_start_position()
    current_state = start_position[0] * grid_size + start_position[1]
    print(f"Epocha {epoch + 1} / {num_epochs}")
```

V průběhu učení agent vybírá mezi dostupnými stavy, do kterých se může přesunout, a pro každý přechod aktualizuje
Q-hodnotu podle vzorce:
$ aktuální odměna + (rychlost učení × maximální budoucí odměna)$.
Trénink končí, když buď agent dosáhne cíle, nebo se dostane do slepé uličky. Epochy jsou cykly, které se opakují
početkrát, dokud agent nenajde lepší nebo snad nejlepší cestu k cíli. Nejlepší políčka jsou ta, která mají nejvyšší
skóre.

(Flatten je super funkce, která převede 2D pole na 1D pole, takže se nemusíme starat o indexy.)

```python
while True:
    valid_next_states = np.argwhere(self.environment.matrix[current_state] >= 0).flatten()

    if len(valid_next_states) == 0:
        break  # Žádné platné následující stavy

    chosen_next_state = random.choice(valid_next_states)
    reward = self.environment.matrix[current_state][chosen_next_state]
    max_future_reward = learning_rate * np.max(self.q_matrix[chosen_next_state])
    updated_q_value = reward + max_future_reward

    self.q_matrix[current_state][chosen_next_state] = updated_q_value

    if current_state in self.environment.target_states:
        break

    current_state = chosen_next_state
```

#### Získání počátečního směru

Na začátku každé epochy je třeba najít platnou počáteční směr pro agenta. Zde jsem to nešťastně pojmnoval jako
startovní pozici. Tato pozice následujícího směru by měla být na políčku, které není překážkou (např. zeď nebo kočka).
Pokud se agentovi podaří najít platnou pozici, vrátí ji jako výstupní hodnotu.

```python
def _get_valid_start_position(self):
    grid_size = self.environment.environment_dimension
    position = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
```

Pokud je pozice neplatná, agent se pokusí najít jinou pozici až do maximálního počtu pokusů, což je ošetřeno prevenci
zacyklení. Získá náhodnou pozici a zkontroluje, zda je učinná. Pokud ne, pokračuje v hledání, dokud nenajde
platnou pozici nebo nedosáhne maximálního počtu pokusů.

```python
attempts_count = 0
max_attempts = 100  # Prevence nekonečné smyčky

cell_type = self.environment.environment[position[0]][position[1]]
cell_score = self.environment.score_dict[cell_type]
```

Pokud najde platnou pozici, vrátí ji jako výstupní hodnotu a ukončí funkci. Pokud se neposune, pokračuje v hledání a
vrátí první náhodný krok. To vede na to, že je agent ve slepé uličce a nemůže se posunout dál.

```python
while cell_score < 0 and attempts_count < max_attempts:
    position = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
    cell_type = self.environment.environment[position[0]][position[1]]
    cell_score = self.environment.score_dict[cell_type]
    attempts_count += 1

return position
```

#### Prohledání Q-matice

Po úspěšném trénování agenta je možné prohledat Q-matice a najít nejlepší cestu k cíli. Tato funkce
`get_steps` prohledá Q-matice a vrátí nejlepší cestu k cíli. Začíná na počáteční pozici a prochází matici,
dokud nenajde cílový stav nebo nedosáhne maximální délky cesty.

```python
def get_steps(self, start_position, max_length=1000):
    grid_size = self.environment.environment_dimension
    current_state = start_position[0] * grid_size + start_position[1]

    path = [start_position]
    steps_taken = 0
```

Inicializace proměnných pro sledování aktuálního stavu, cesty a počtu kroků. Cílový stav je definován jako
souřadnice sýra, které jsou uloženy v množině `target_states`.

```python    
while current_state not in self.environment.target_states and steps_taken < max_length:
    state_q_values = self.q_matrix[current_state]
    if np.max(state_q_values) == 0:
        return None 
```

Pokud není žádná Q-hodnota, agent se dostal do slepé uličky a nemůže pokračovat. V takovém případě se vrátí
`None`. Jinak se vybere nejlepší následující stav na základě Q-hodnoty.

```python 
best_next_states = np.argwhere(state_q_values >= np.amax(state_q_values)).flatten()
best_next_state = random.choice(best_next_states)
```

Konverze nejlepšího následujícího stavu na souřadnice matice a přidání do cesty. Počet kroků se zvyšuje o 1.

```python
next_position = (best_next_state // grid_size, best_next_state % grid_size)
path.append(next_position)
steps_taken += 1

current_state = best_next_state

return path
```

Toto je v kostce celý algoritmus Q-learningu.

## Výsledky

### Výsledek s jedním sýrem

![Q-learning s jedním sýrem](results/chase_the_cheese.gif)

### Výsledek se několika sýry

![Q-learning se dvěma sýry](results/chase_multiple_cheese.gif)