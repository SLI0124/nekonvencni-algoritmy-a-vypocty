import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import random


class GameEnvironmentMatrix:
    """Reprezentuje herní prostředí jako matici pro Q-learning algoritmus."""

    def __init__(self, environment, score_dict):
        """
        Inicializuje matici prostředí. V podstatě se jedná o matici sousedních políček.

        :param environment: 2D seznam představující herní mapu
        :param score_dict: Slovník mapující typy políček na jejich skóre
        """
        self.environment_dimension = len(environment)  # Předpokládá se čtvercová matice
        self.environment = environment
        self.score_dict = score_dict
        matrix_dimension = self.environment_dimension ** 2
        self.target_states = set()

        self.matrix = np.full(shape=(matrix_dimension, matrix_dimension), fill_value=-1)

        for row in range(self.environment_dimension):
            for column in range(self.environment_dimension):
                matrix_index = row * self.environment_dimension + column

                # Kontrola horního souseda
                if (row - 1) >= 0:
                    top = (row - 1) * self.environment_dimension + column
                    self.matrix[matrix_index][top] = score_dict[environment[row - 1][column]]

                # Kontrola dolního souseda
                if (row + 1) < self.environment_dimension:
                    bottom = (row + 1) * self.environment_dimension + column
                    self.matrix[matrix_index][bottom] = score_dict[environment[row + 1][column]]

                # Kontrola levého souseda
                if (column - 1) >= 0:
                    left = matrix_index - 1
                    self.matrix[matrix_index][left] = score_dict[environment[row][column - 1]]

                # Kontrola pravého souseda
                if (column + 1) < self.environment_dimension:
                    right = matrix_index + 1
                    self.matrix[matrix_index][right] = score_dict[environment[row][column + 1]]

                # Označení cílových stavů
                if environment[row][column] == 'cheese':
                    self.matrix[matrix_index][matrix_index] = score_dict[environment[row][column]]
                    self.target_states.add(matrix_index)


class QLearning:
    """Implementace algoritmu Q-learning pro hledání optimální cesty."""

    def __init__(self, environment):
        """Inicializuje Q-learning s daným prostředím.

        :param environment: Instance třídy GameEnvironmentMatrix
        """
        self.q_matrix = np.zeros(environment.matrix.shape)
        self.environment = environment

    def train(self, num_epochs, learning_rate):
        """
        Trénuje Q-learning model.
        :param num_epochs: Počet epoch pro trénink
        :param learning_rate: Rychlost učení (0-1)
        :return: None
        """
        grid_size = self.environment.environment_dimension

        for epoch in range(num_epochs):
            # Generování náhodné startovací pozice
            start_position = self._get_valid_start_position()
            current_state = start_position[0] * grid_size + start_position[1]
            print(f"Epocha {epoch + 1} / {num_epochs}")

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

    def _get_valid_start_position(self):
        """Vrací náhodnou platnou startovací pozici."""
        grid_size = self.environment.environment_dimension
        position = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))

        attempts_count = 0
        max_attempts = 100  # Prevence nekonečné smyčky

        cell_type = self.environment.environment[position[0]][position[1]]
        cell_score = self.environment.score_dict[cell_type]

        while cell_score < 0 and attempts_count < max_attempts:
            position = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
            cell_type = self.environment.environment[position[0]][position[1]]
            cell_score = self.environment.score_dict[cell_type]
            attempts_count += 1

        return position

    def get_steps(self, start_position, max_length=1000):
        """
        Vrací kroky cesty od počáteční pozice k cíli.
        :param start_position: Počáteční pozice (řádek, sloupec)
        :param max_length: Maximální počet kroků pro cestu
        :return: Seznam kroků nebo None, pokud cesta nebyla nalezena
        """
        grid_size = self.environment.environment_dimension
        current_state = start_position[0] * grid_size + start_position[1]

        path = [start_position]
        steps_taken = 0

        while current_state not in self.environment.target_states and steps_taken < max_length:
            state_q_values = self.q_matrix[current_state]
            if np.max(state_q_values) == 0:
                return None  # Žádná naučená cesta

            best_next_states = np.argwhere(state_q_values >= np.amax(state_q_values)).flatten()
            best_next_state = random.choice(best_next_states)

            # Konverze indexu zpět na souřadnice
            next_position = (best_next_state // grid_size, best_next_state % grid_size)
            path.append(next_position)
            steps_taken += 1

            current_state = best_next_state

        return path


class QLearningApp:
    """Hlavní aplikace pro vizualizaci Q-learning algoritmu."""

    def __init__(self, master, grid_size):
        """Inicializace GUI aplikace"""

        self.canvas = None
        self.master = master
        self.grid_size = grid_size
        self.cell_size = 32
        self.grid = [['floor' for _ in range(grid_size)] for _ in range(grid_size)]
        self._load_sprites()

        # Skóre pro odměnu pro jednotlivé typy políček
        self.scores = {'floor': 0, 'wall': -10, 'cat': -100, 'cheese': 100, 'mouse': 0}

        self.current_sprite = 'floor'
        self.sprite_ids = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        self.mouse_coords = None
        self.network = None
        self.steps = []
        self.animation_id = None

        self.create_widgets()

    def _load_sprites(self):
        """Načte obrázky pro jednotlivé typy políček."""
        self.sprites = {
            'floor': tk.PhotoImage(file="images/floor.png"),
            'mouse': tk.PhotoImage(file="images/mouse.png"),
            'cat': tk.PhotoImage(file="images/cat.png"),
            'wall': tk.PhotoImage(file="images/wall.png"),
            'cheese': tk.PhotoImage(file="images/cheese.png")
        }

    def create_widgets(self):
        """Vytvoří všechny prvky GUI rozhraní."""
        # Hlavní canvas
        self.canvas = tk.Canvas(
            self.master,
            width=self.grid_size * self.cell_size,
            height=self.grid_size * self.cell_size,
            bg='white'
        )
        self.canvas.pack(side=tk.LEFT)

        # Inicializace prázdné mřížky
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.canvas.create_image(
                    j * self.cell_size,
                    i * self.cell_size,
                    anchor=tk.NW,
                    image=self.sprites['floor']
                )

        self.canvas.bind("<Button-1>", self.set_cell)

        # Panel nástrojů vpravo
        self._create_edit_panel()

        # Panel tlačítek vlevo
        self._create_control_panel()

    def _create_edit_panel(self):
        """Vytvoří panel pro editaci mapy."""
        edit_frame = tk.Frame(self.master, bd=2, relief=tk.RAISED)
        edit_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        edit_label = tk.Label(edit_frame, text="Edit")
        edit_label.pack()

        for sprite_name in self.sprites:
            sprite_button = tk.Button(
                edit_frame,
                text=sprite_name.capitalize(),
                command=lambda s=sprite_name: self.select_sprite(s)
            )
            sprite_button.pack(pady=5)

    def _create_control_panel(self):
        """Vytvoří panel s ovládacími tlačítky."""
        control_frame = tk.Frame(self.master)
        control_frame.pack(side=tk.LEFT)

        buttons = [
            ("Chase the cheese", self.chase),
            ("Train", self.train),
            ("Save Map", self.save_map),
            ("Load Map", lambda: self.load_map()),
            ("Clear map", self.clear_level)
        ]

        for text, command in buttons:
            button = tk.Button(control_frame, text=text, command=command)
            button.pack(pady=5, padx=5)

    def select_sprite(self, sprite_name):
        """Nastaví aktuálně vybraný typ políčka."""
        self.current_sprite = sprite_name

    def set_cell(self, event):
        """Umístí aktuální typ políčka na vybranou pozici v mřížce."""
        row = event.y // self.cell_size
        col = event.x // self.cell_size

        # Kontrola platnosti souřadnic
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return

        # Odstranění předchozího obrázku
        if self.sprite_ids[row][col] is not None:
            self.canvas.delete(self.sprite_ids[row][col])

        # Speciální případ pro myš - může být jen jedna
        if self.current_sprite == 'mouse':
            if self.mouse_coords is not None:
                old_row, old_col = self.mouse_coords
                self.canvas.delete(self.sprite_ids[old_row][old_col])
                self.grid[old_row][old_col] = 'floor'
                self.sprite_ids[old_row][old_col] = self.canvas.create_image(
                    old_col * self.cell_size,
                    old_row * self.cell_size,
                    anchor=tk.NW,
                    image=self.sprites['floor']
                )
            self.mouse_coords = (row, col)

        # Aktualizace mřížky a vykreslení
        self.grid[row][col] = self.current_sprite
        self.sprite_ids[row][col] = self.canvas.create_image(
            col * self.cell_size,
            row * self.cell_size,
            anchor=tk.NW,
            image=self.sprites[self.current_sprite]
        )

    def train(self):
        """Spustí trénink Q-learning algoritmu."""
        if self.mouse_coords is None:
            messagebox.showerror("Chyba", "Nejprve umístěte myš na mapu!")
            return

        play_matrix = GameEnvironmentMatrix(self.grid, self.scores)
        self.network = QLearning(play_matrix)

        # Trénink s 1000 epochami a rychlostí učení 0.5
        self.network.train(1000, 0.5)
        messagebox.showinfo("Training has finished", "Q-learning training has finished!")

    def chase(self):
        """Spustí animaci cesty k sýru podle natrénovaného modelu."""
        if self.network is None:
            messagebox.showerror("Error", "First train the Q-learning model!")
            return

        if self.mouse_coords is None:
            messagebox.showerror("Error", "First place the mouse on the map!")
            return

        # Vymazání předchozí animace
        if self.animation_id:
            self.canvas.after_cancel(self.animation_id)

        # Získání kroků cesty
        self.steps = self.network.get_steps(self.mouse_coords)

        if not self.steps:
            messagebox.showinfo("Nenalezena cesta", "Nebyla nalezena žádná cesta k cíli!")
            return

        # Spuštění animace
        self.animation_id = self.canvas.after(300, self.next_step)

    def next_step(self):
        """Pokračuje v animaci cesty k sýru."""
        if not self.steps or len(self.steps) <= 1:  # Konec animace
            return

        # Odstranění aktuální pozice myši
        self.steps = self.steps[1:]  # První krok je aktuální pozice
        if not self.steps:
            return

        step = self.steps[0]

        # Aktualizace pozice myši
        self.canvas.delete(self.sprite_ids[self.mouse_coords[0]][self.mouse_coords[1]])

        # Obnovení původního podkladového políčka
        original_sprite = self.grid[self.mouse_coords[0]][self.mouse_coords[1]]
        if original_sprite == 'mouse':
            original_sprite = 'floor'

        self.grid[self.mouse_coords[0]][self.mouse_coords[1]] = original_sprite
        self.sprite_ids[self.mouse_coords[0]][self.mouse_coords[1]] = self.canvas.create_image(
            self.mouse_coords[1] * self.cell_size,
            self.mouse_coords[0] * self.cell_size,
            anchor=tk.NW,
            image=self.sprites[original_sprite]
        )

        # Přesun myši na novou pozici
        self.mouse_coords = step
        self.grid[step[0]][step[1]] = "mouse"
        self.sprite_ids[step[0]][step[1]] = self.canvas.create_image(
            step[1] * self.cell_size,
            step[0] * self.cell_size,
            anchor=tk.NW,
            image=self.sprites["mouse"]
        )

        # Plánování dalšího kroku
        self.animation_id = self.canvas.after(300, self.next_step)

    def save_map(self):
        """Uloží aktuální mapu do souboru."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )

        if not filename:
            return

        try:
            with open(filename, 'w') as f:
                for row in self.grid:
                    f.write(' '.join(row) + '\n')
            messagebox.showinfo("Mapa uložena", "Mapa byla úspěšně uložena!")
        except Exception as e:
            messagebox.showerror("Chyba při ukládání", f"Nastala chyba: {str(e)}")

    def load_map(self, filename=None):
        """Načte mapu ze souboru."""
        if not filename:
            filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])

        if not filename:
            return

        try:
            with open(filename, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                new_grid_size = len(lines)

                # Kontrola správnosti formátu mapy
                if not all(len(line.split()) == new_grid_size for line in lines):
                    raise ValueError("Invalid map format: all rows must have the same number of columns.")

                # Aktualizace velikosti mřížky
                self.grid_size = new_grid_size
                self.grid = [None] * self.grid_size
                self.mouse_coords = None

                # Načtení políček
                for row, line in enumerate(lines):
                    cells = line.strip().split()
                    if 'mouse' in cells:
                        col = cells.index('mouse')
                        self.mouse_coords = (row, col)
                    self.grid[row] = cells

                # Příprava nových ID pro tkinter canvas
                self.sprite_ids = [[None for _ in range(self.grid_size)] for _ in range(self.grid_size)]

                # Úprava velikosti plátna
                self.canvas.config(
                    width=self.grid_size * self.cell_size,
                    height=self.grid_size * self.cell_size
                )

                # Překreslení celého plátna
                self.redraw_canvas()
                messagebox.showinfo("Map loaded", "The game map was successfully loaded!")
        except Exception as e:
            messagebox.showerror("Error during loading", f"An error occurred: {str(e)}")

    def redraw_canvas(self):
        """Překreslí celé plátno podle aktuální mřížky."""
        self.canvas.delete("all")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                sprite_name = self.grid[i][j]
                if sprite_name not in self.sprites:
                    sprite_name = 'floor'  # Kdyby byl neplatný typ
                self.sprite_ids[i][j] = self.canvas.create_image(
                    j * self.cell_size,
                    i * self.cell_size,
                    anchor=tk.NW,
                    image=self.sprites[sprite_name]
                )

    def clear_level(self):
        """Vymaže mapu a resetuje aplikaci."""
        self.canvas.delete("all")
        self.grid = [['floor' for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.sprite_ids[i][j] = self.canvas.create_image(
                    j * self.cell_size,
                    i * self.cell_size,
                    anchor=tk.NW,
                    image=self.sprites["floor"]
                )

        self.mouse_coords = None
        self.network = None

        if self.animation_id:
            self.canvas.after_cancel(self.animation_id)
            self.animation_id = None


def main():
    root = tk.Tk()
    root.title("QLearning")
    QLearningApp(root, 10)
    root.mainloop()


if __name__ == "__main__":
    main()
