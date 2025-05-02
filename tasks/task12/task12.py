import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import imageio
import os
from enum import Enum
from tqdm import tqdm


class CellState(Enum):
    """Výčtový typ pro možné stavy buněk v automatu lesního požáru"""
    EMPTY = 0
    TREE = 1
    FIRE = 2
    BURNT = 3


class ForestFireAutomaton:
    def __init__(self, size=200, p=0.04, f=0.0005, density=0.6):
        self.size = size
        self.p = p  # Pravděpodobnost růstu nového stromu
        self.f = f  # Pravděpodobnost spontánního vzniku požáru
        self.density = density  # Počáteční hustota stromů

        # Barevná mapa pro vizualizaci
        self.cmap = colors.ListedColormap(['lightgray', 'forestgreen', 'red', 'darkgray'])
        self.bounds = [s.value for s in CellState] + [CellState.BURNT.value + 1]
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)

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

    def get_von_neumann_neighbors(self, i, j, forest):
        """Získání Von Neumannova okolí - 4 sousedi v kardinálních směrech"""
        return [
            forest[(i - 1) % self.size, j],
            forest[(i + 1) % self.size, j],
            forest[i, (j - 1) % self.size],
            forest[i, (j + 1) % self.size]
        ]

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

    def update_forest(self, forest, neighborhood_func):
        """Aplikace přechodových pravidel celulárního automatu lesního požáru na všechny buňky"""
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

    def simulate_and_save(self, neighborhood_func, name, frames=150):
        """Spuštění simulace a uložení výsledku jako GIF"""
        forest = self.initialize_forest()

        os.makedirs('results', exist_ok=True)
        os.makedirs('results/temp_frames', exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 12))

        for frame in tqdm(range(frames), desc=f"Generování {name} okolí", unit="snímek"):
            # Aplikace pravidel automatu
            forest = self.update_forest(forest, neighborhood_func)

            # Vykreslení aktuálního stavu
            ax.clear()
            ax.imshow(forest, cmap=self.cmap, norm=self.norm)
            ax.set_title(f"{name} okolí - Snímek {frame + 1}/{frames}", fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])

            temp_filename = f"results/temp_frames/{name}_frame_{frame:03d}.png"
            plt.savefig(temp_filename, dpi=120, bbox_inches='tight')

        output_path = f'results/forest_fire_{name.lower().replace(" ", "_")}.gif'

        with imageio.get_writer(output_path, mode='I', duration=0.08) as writer:
            for frame in tqdm(range(frames), desc="Tvorba GIFu", unit="snímek"):
                temp_filename = f"results/temp_frames/{name}_frame_{frame:03d}.png"
                image = imageio.v2.imread(temp_filename)
                writer.append_data(image)

        for frame in tqdm(range(frames), desc="Mazání dočasných souborů", unit="soubor"):
            temp_filename = f"results/temp_frames/{name}_frame_{frame:03d}.png"
            os.remove(temp_filename)

        os.rmdir('results/temp_frames')
        plt.close()
        print(f"Simulace uložena jako '{output_path}'")


def main():
    simulator = ForestFireAutomaton(size=200, p=0.05, f=0.001, density=0.5)

    neighborhood_types = [
        ("Von Neumann", simulator.get_von_neumann_neighbors),
        ("Moore", simulator.get_moore_neighbors),
        ("Hexagonal", simulator.get_hexagonal_neighbors)
    ]

    for name, func in neighborhood_types:
        simulator.simulate_and_save(func, name, frames=150)


if __name__ == "__main__":
    main()
