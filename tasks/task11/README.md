# Chaotický pohyb dvojitého kyvadla

## Úvod do dvojitého kyvadla

Dvojité kyvadlo je jednoduchý fyzikální systém, ve kterém můžeme pozorovat chaotické chování. Skládá se z:
- 2 kyvadel s hmotnostmi 𝑚₁ a 𝑚₂
- 2 nehmotných lanek délek 𝑙₁ a 𝑙₂
- Obě kyvadla jsou vychýlena o úhly: 𝜃₁ a 𝜃₂

Přestože se jedná o poměrně jednoduchý mechanický systém, jeho dynamika může být extrémně složitá a nepředvídatelná, což je typickým rysem chaotických systémů.

## Matematický model

### Pozice kyvadel

Pozice prvního kyvadla:
- 𝑥₁ = 𝑙₁ sin(𝜃₁)
- 𝑦₁ = −𝑙₁ cos(𝜃₁)

Pozice druhého kyvadla:
- 𝑥₂ = 𝑙₁ sin(𝜃₁) + 𝑙₂ sin(𝜃₂)
- 𝑦₂ = −𝑙₁ cos(𝜃₁) − 𝑙₂ cos(𝜃₂)

### Rychlosti

- ẋ₁ = 𝑙₁θ̇₁ cos(𝜃₁)
- ẏ₁ = 𝑙₁θ̇₁ sin(𝜃₁)
- ẋ₂ = 𝑙₁θ̇₁ cos(𝜃₁) + 𝑙₂θ̇₂ cos(𝜃₂)
- ẏ₂ = 𝑙₁θ̇₁ sin(𝜃₁) + 𝑙₂θ̇₂ sin(𝜃₂)

### Energie systému

**Potenciální energie:**
```
V = m₁gy₁ + m₂gy₂ = -(m₁ + m₂)l₁g cos(𝜃₁) - m₂l₂g cos(𝜃₂)
```

**Kinetická energie:**
```
T = (1/2)m₁v₁² + (1/2)m₂v₂² = (1/2)m₁(ẋ₁² + ẏ₁²) + (1/2)m₂(ẋ₂² + ẏ₂²)
  = (1/2)m₁l₁²θ̇₁² + (1/2)m₂[l₁²θ̇₁² + l₂²θ̇₂² + 2l₁l₂θ̇₁θ̇₂ cos(𝜃₁ - 𝜃₂)]
```

**Lagrangián:**
```
ℒ = T - V = (1/2)(m₁ + m₂)l₁²θ̇₁² + (1/2)m₂l₂²θ̇₂² + m₂l₁l₂θ̇₁θ̇₂ cos(𝜃₁ - 𝜃₂) + (m₁ + m₂)l₁g cos(𝜃₁) + m₂gl₂ cos(𝜃₂)
```

### Pohybové rovnice

Použitím Euler-Lagrangeových rovnic:

```
d/dt(∂ℒ/∂θ̇ᵢ) - ∂ℒ/∂θᵢ = 0, kde θᵢ = 𝜃₁, 𝜃₂
```

Získáme diferenciální rovnice popisující zrychlení obou kyvadel:

**Zrychlení prvního kyvadla:**
```
θ̈₁ = [m₂g sin(𝜃₂) cos(𝜃₁ - 𝜃₂) - m₂ sin(𝜃₁ - 𝜃₂)(l₁θ̇₁² cos(𝜃₁ - 𝜃₂) + l₂θ̇₂²) - (m₁ + m₂)g sin(𝜃₁)] / [l₁(m₁ + m₂ sin²(𝜃₁ - 𝜃₂))]
```

**Zrychlení druhého kyvadla:**
```
θ̈₂ = [(m₁ + m₂)(l₁θ̇₁² sin(𝜃₁ - 𝜃₂) - g sin(𝜃₂) + g sin(𝜃₁) cos(𝜃₁ - 𝜃₂)) + m₂l₂θ̇₂² sin(𝜃₁ - 𝜃₂) cos(𝜃₁ - 𝜃₂)] / [l₂(m₁ + m₂ sin²(𝜃₁ - 𝜃₂))]
```

## Implementace

Implementace využívá funkci `odeint` z knihovny SciPy pro numerické řešení diferenciálních rovnic.

### Definice derivační funkce

```python
def get_derivative(state, t, l1, l2, m1, m2):
    """
    Vypočítá derivace stavových proměnných dvojitého kyvadla.
    
    Args:
        state: aktuální stav [theta1, omega1, theta2, omega2]
        t: čas (nepoužívá se pro autonomní systém)
        l1, l2: délky kyvadel
        m1, m2: hmotnosti kyvadel
    
    Returns:
        derivace stavových proměnných [omega1, alpha1, omega2, alpha2]
    """
    theta1, omega1, theta2, omega2 = state
    delta = theta2 - theta1
    
    # Pomocné výpočty pro zlepšení čitelnosti
    den1 = l1 * (m1 + m2 * np.sin(delta)**2)
    den2 = l2 * (m1 + m2 * np.sin(delta)**2)
    
    # Výpočet zrychlení pomocí odvozených vzorců
    alpha1 = (m2 * g * np.sin(theta2) * np.cos(delta) 
             - m2 * np.sin(delta) * (l1 * omega1**2 * np.cos(delta) + l2 * omega2**2) 
             - (m1 + m2) * g * np.sin(theta1)) / den1
    
    alpha2 = ((m1 + m2) * (l1 * omega1**2 * np.sin(delta) - g * np.sin(theta2) + g * np.sin(theta1) * np.cos(delta))
             + m2 * l2 * omega2**2 * np.sin(delta) * np.cos(delta)) / den2
    
    return [omega1, alpha1, omega2, alpha2]
```

### Inicializace a řešení

```python
def simulate_double_pendulum(theta1_0, theta2_0, l1, l2, m1, m2, time_span=10, dt=0.01):
    """
    Simuluje pohyb dvojitého kyvadla.
    
    Args:
        theta1_0, theta2_0: počáteční úhly
        l1, l2: délky kyvadel
        m1, m2: hmotnosti kyvadel
        time_span: doba simulace
        dt: časový krok
        
    Returns:
        časové kroky a odpovídající stavy systému
    """
    # Počáteční stav: [theta1, omega1, theta2, omega2]
    state_0 = [theta1_0, 0, theta2_0, 0]  # Počáteční úhlové rychlosti jsou nulové
    
    # Časové kroky pro simulaci
    t = np.arange(0, time_span, dt)
    
    # Řešení diferenciálních rovnic
    states = odeint(get_derivative, state_0, t, args=(l1, l2, m1, m2))
    
    return t, states
```

### Výpočet pozic kyvadel

```python
def calculate_positions(states, l1, l2):
    """
    Vypočítá pozice kyvadel ze stavů.
    
    Args:
        states: stavy systému [theta1, omega1, theta2, omega2]
        l1, l2: délky kyvadel
        
    Returns:
        pozice obou kyvadel (x1, y1, x2, y2)
    """
    theta1 = states[:, 0]
    theta2 = states[:, 2]
    
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    
    return x1, y1, x2, y2
```

## Chaotické chování

Dvojité kyvadlo je klasickým příkladem chaotického systému, který vykazuje tyto vlastnosti:

1. **Extrémní citlivost na počáteční podmínky**: Nepatrná změna v počátečních úhlech může vést k dramaticky odlišným trajektoriím.
2. **Nepředvídatelnost dlouhodobého chování**: Přestože je systém plně deterministický, není prakticky možné předpovědět jeho dlouhodobé chování bez přesné simulace.
3. **Nedostatek periodicity**: Chaotický režim systému nevykazuje zjevnou periodicitu v pohybu.

## Výsledky

Simulace dvojitého kyvadla generuje tyto vizuální výstupy:

1. **Trajektorie koncových bodů**: Vizualizace dráhy, kterou opisuje koncový bod druhého kyvadla, často vytváří komplexní a esteticky zajímavé vzory.

2. **Fázový diagram**: Zobrazení závislosti úhlů a úhlových rychlostí, které ukazuje strukturu chaotického atraktoru.

3. **Animace pohybu**: Dynamická vizualizace pohybu obou kyvadel v čase, která názorně ukazuje chaotickou povahu systému.

4. **Lyapunovův exponent**: Numerická charakteristika míry chaotičnosti systému, která kvantifikuje citlivost na počáteční podmínky.

### Ukázka chaotické trajektorie

![Chaotická trajektorie](results/chaotic_trajectory.png)

### Porovnání pro malou změnu počátečních podmínek

![Citlivost na počáteční podmínky](results/sensitivity.png)

## Závěr

Simulace dvojitého kyvadla názorně demonstruje, jak jednoduchý mechanický systém může vykazovat složité chaotické chování. I když je jeho dynamika plně popsána deterministickými diferenciálními rovnicemi, jeho dlouhodobé chování je efektivně nepředvídatelné. Toto je základní vlastnost chaotických systémů a ilustruje limity předvídatelnosti ve fyzikálních systémech obecně.

## Literatura

- [The Double Pendulum (SciPython)](https://scipython.com/blog/the-double-pendulum/)
- Strogatz, S. H. (2000). Nonlinear Dynamics and Chaos. Westview Press.
