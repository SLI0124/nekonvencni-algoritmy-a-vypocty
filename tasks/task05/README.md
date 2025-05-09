# Balancování Tyče s Deep Q-Learning

Tento projekt implementuje řešení klasického problému "Cart-Pole" (balancování tyče) pomocí techniky Deep Q-Learning.
Aplikace umožňuje trénovat neuronovou síť, která se naučí udržet tyč ve vzpřímené poloze, a vizualizovat výsledky.

## Klíčové komponenty

### QNetwork

```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

Neuronová síť má architekturu:

- Vstupní vrstva: 4 neurony (pozice vozíku, rychlost vozíku, úhel tyče, úhlová rychlost tyče)
- Dvě skryté vrstvy po 64 neuronech s ReLU aktivací
- Výstupní vrstva: 2 neurony (možné akce - pohyb vlevo nebo vpravo)

### DQNAgent

```python
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # Discount faktor pro budoucí odměny
        self.epsilon = 1.0  # Míra průzkumu prostředí
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
```

Agent implementuje:

- **Epsilon-greedy strategie**: Balancuje průzkum nových akcí a využívání naučených znalostí
- **Experience replay**: Ukládá zkušenosti (stavy, akce, odměny) a učí se z nich v dávkách
- **Vylepšené odměny**: Systém penalizace za úhel a pozici vozíku
- **Adaptivní učení**: Postupné snižování míry průzkumu (epsilon decay)

## Proces učení

Algoritmus trénuje agenta pomocí následujících kroků:

1. **Inicializace**: Začíná s náhodným výběrem akcí (vysoká hodnota epsilon)
2. **Sběr zkušeností**: Agent interaguje s prostředím a ukládá zkušenosti do paměti
3. **Dávkové učení**: Trénuje model na náhodných vzorcích z paměti
4. **Adaptace**: Postupně snižuje míru průzkumu (epsilon) pro lepší využití naučených znalostí
5. **Evaluace**: Sleduje průměrné skóre (počet kroků, po které agent udrží tyč vzpřímeně)

```python
def run_training(self):
    # ...existing code...
    for e in range(self.episodes):
        state = self.env.reset()[0]
        total_reward = 0
        self.step_counter = 0

        while self.step_counter < self.total_steps:
            action = self.agent.act(state)
            next_state, reward, done, _, _ = self.env.step(action)

            # Vylepšená odměna
            pole_angle = abs(state[2])
            position = abs(state[0])
            angle_penalty = 1.0 - (pole_angle / (np.pi / 4))
            position_penalty = 1.0 - (position / 2.4)
            reward += angle_penalty + position_penalty

            self.agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += 1
            self.step_counter += 1
            if done:
                break

        scores.append(total_reward)
        self.agent.replay(self.batch_size)
        # ...existing code...
```

## Vizualizace a výstupy

Aplikace poskytuje:

1. **Interaktivní GUI**: Umožňuje nastavení parametrů a sledování tréninku
2. **Živá vizualizace**: Zobrazení aktuálního stavu balancování tyče
3. **Graf tréninku**: Průběžné zobrazení vývoje skóre během tréninku
4. **Export výsledků**:
    - Uložení trénovacího grafu jako PNG
    - Vytvoření GIF animace demonstrující naučené řešení

## Použití aplikace

1. **Konfigurace parametrů**:
    - Počet trénovacích epizod
    - Velikost trénovací dávky
    - Celkový počet kroků na epizodu

2. **Trénování modelu**:
    - Kliknutím na "Start Training" se spustí trénink
    - Průběžný stav je zobrazován v textovém poli a grafu

3. **Vizualizace řešení**:
    - Po dokončení tréninku klikněte na "Visualize Solution"

4. **Export výsledků**:
    - "Save Training Graph" - uloží graf průběhu tréninku
    - "Save Animation" - vytvoří GIF animaci demonstrující naučené řešení

## Ukázky výstupů

### Grafické uživatelské rozhraní

![GUI](results/gui.jpg)

### Graf průběhu tréninku

#### Průběh tréninku #1

![Training Progress](results/graph/training_graph_1.png)

#### Průběh tréninku #2

![Training Progress](results/graph/training_graph_2.png)

### Animace řešení

#### Animace #1

![Solution Animation](results/animation/pole_balancing_1.gif)

#### Animace #2

![Solution Animation](results/animation/pole_balancing_2.gif)
`