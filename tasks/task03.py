import numpy as np
from copy import deepcopy
import tkinter as tk
from tkinter import messagebox


class Pattern:
    """Reprezentace vzoru pro Hopfieldovu síť."""

    def __init__(self, matrix):
        self.matrix = matrix
        # Konverze 0 na -1 pro výpočty Hopfieldovy sítě
        self.vector = []
        for row in matrix:
            for value in row:
                if value == 0:
                    self.vector.append(-1)
                else:
                    self.vector.append(1)
        # Váhová matice pro vzor
        self.weight_matrix = self._calculate_weight_matrix()

    def _calculate_weight_matrix(self):
        """Vypočítá váhovou matici pro vzor pomocí Hebbova pravidla učení."""
        n = len(self.vector)
        weight_matrix = np.zeros((n, n), dtype=np.int32)

        # Manuální výpočet váhové matice podle Hebbova pravidla
        for i in range(n):
            for j in range(n):
                # Nastavíme váhu pouze pokud i != j (diagonála zůstane 0)
                if i != j:
                    weight_matrix[i, j] = self.vector[i] * self.vector[j]

        return weight_matrix

    def __eq__(self, other):
        """Porovnání dvou vzorů na základě jejich matic."""
        if not isinstance(other, Pattern):
            return False
        return np.array_equal(self.matrix, other.matrix)


class HopfieldNetwork:
    """Implementace Hopfieldovy sítě pro rozpoznávání a obnovení vzorů."""

    def __init__(self, grid_size, stable_threshold=5):
        self.grid_size = grid_size
        self.size = grid_size ** 2
        self.weights = np.zeros((self.size, self.size), np.int32)
        self.patterns = []
        self.stable_threshold = stable_threshold  # Počet iterací bez změny pro považování sítě za stabilní

    def add_pattern(self, pattern):
        """Přidá vzor do sítě, pokud ještě není přítomen."""
        if pattern in self.patterns:
            return False

        self.weights += pattern.weight_matrix
        self.patterns.append(deepcopy(pattern))
        return True

    def _vector_to_matrix(self, vector):
        """Převede vektor (-1/1) zpět na matici (0/1)."""
        result_matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                index = i * self.grid_size + j
                if vector[index] == -1:
                    result_matrix[i][j] = 0
                else:
                    result_matrix[i][j] = 1
        return result_matrix

    def synchronous_recovery(self, input_pattern):
        """Provede synchronní rekonstrukci vzoru - všechny neurony se aktualizují současně."""
        vector = deepcopy(input_pattern.vector)

        # Aktualizace všech neuronů najednou
        aktivace = np.dot(self.weights, vector)
        new_vector = np.zeros_like(vector)
        for i in range(len(aktivace)):
            if aktivace[i] > 0:
                new_vector[i] = 1
            else:
                new_vector[i] = -1

        result_matrix = self._vector_to_matrix(new_vector)
        return Pattern(result_matrix)

    def asynchronous_recovery(self, input_pattern):
        """Provede asynchronní rekonstrukci vzoru - neurony se aktualizují postupně v náhodném pořadí."""
        vector = deepcopy(input_pattern.vector)
        stable_count = 0
        prev_vector = None

        while stable_count < self.stable_threshold:
            # Aktualizace neuronů v náhodném pořadí
            indices = np.random.permutation(self.size)
            for i in indices:
                activation = np.dot(self.weights[i], vector)
                vector[i] = np.sign(activation)

            if prev_vector is not None and np.array_equal(vector, prev_vector):
                stable_count += 1
            else:
                stable_count = 0

            prev_vector = deepcopy(vector)

        result_matrix = self._vector_to_matrix(vector)
        return Pattern(result_matrix)


class GridCanvas(tk.Canvas):
    """Plátno pro zobrazení a editaci vzoru v mřížce."""

    def __init__(self, master, grid_size, cell_size, on_cell_toggle):
        width = height = grid_size * cell_size
        super().__init__(master, width=width, height=height, bg='white')

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.on_cell_toggle = on_cell_toggle
        self.cells = []

        self._create_grid()

    def _create_grid(self):
        """Vytvoří mřížku buněk."""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x1, y1 = j * self.cell_size, i * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                rect = self.create_rectangle(x1, y1, x2, y2, fill='white', outline='black')
                self.tag_bind(rect, '<Button-1>', lambda e, r=i, c=j: self.on_cell_toggle(r, c))
                self.cells.append(rect)

    def update_display(self, grid):
        """Aktualizuje zobrazení podle matice."""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                color = 'black' if grid[i][j] == 1 else 'white'
                self.itemconfig(self.cells[idx], fill=color)


class ControlPanel(tk.Frame):
    """Panel s ovládacími tlačítky."""

    def __init__(self, master, callbacks):
        super().__init__(master)

        buttons = [
            ("Save Pattern", callbacks['save_pattern']),
            ("Show Saved Pattern", callbacks['show_pattern']),
            ("Clear Grid", callbacks['clear']),
            ("Recover Synchronously", callbacks['sync']),
            ("Recover Asynchronously", callbacks['async'])
        ]

        for text, command in buttons:
            btn = tk.Button(self, text=text, command=command)
            btn.pack(fill=tk.X, padx=5, pady=5)


class HopfieldApp:
    """Hlavní aplikace pro práci s Hopfieldovou sítí."""

    def __init__(self, master, grid_size=5, cell_size=50):
        self.master = master
        self.grid_size = grid_size
        self.cell_size = cell_size

        self.grid = np.zeros((grid_size, grid_size), np.int32)
        self.network = HopfieldNetwork(grid_size)
        self.current_pattern_index = -1

        # Vytvoření GUI
        callbacks = {
            'save_pattern': self.add_pattern,
            'show_pattern': self.next_pattern,
            'clear': self.clear_grid,
            'sync': self.recover_sync,
            'async': self.recover_async
        }

        self.canvas = GridCanvas(master, grid_size, cell_size, self.toggle_cell)
        self.canvas.pack(side=tk.LEFT)

        self.control_panel = ControlPanel(master, callbacks)
        self.control_panel.pack(side=tk.RIGHT, fill=tk.Y)

    def toggle_cell(self, i, j):
        """Přepne stav buňky na pozici [i, j]."""
        self.grid[i][j] = 1 - self.grid[i][j]
        self.canvas.update_display(self.grid)
        self.current_pattern_index = -1

    def clear_grid(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), np.int32)
        self.canvas.update_display(self.grid)

    def add_pattern(self):
        pattern = Pattern(self.grid.copy())
        if self.network.add_pattern(pattern):
            messagebox.showinfo("Pattern Added", "The pattern has been added to the network.")
        else:
            messagebox.showinfo("Pattern Exists", "This pattern already exists.")

    def next_pattern(self):
        if not self.network.patterns:
            messagebox.showinfo("No Patterns", "There are no patterns to show.")
            return

        self.current_pattern_index = (self.current_pattern_index + 1) % len(self.network.patterns)
        pattern = self.network.patterns[self.current_pattern_index]
        self.grid = pattern.matrix.copy()
        self.canvas.update_display(self.grid)

    def recover_sync(self):
        pattern = Pattern(self.grid.copy())
        result = self.network.synchronous_recovery(pattern)
        self.grid = result.matrix.copy()
        self.canvas.update_display(self.grid)

    def recover_async(self):
        pattern = Pattern(self.grid.copy())
        result = self.network.asynchronous_recovery(pattern)
        self.grid = result.matrix.copy()
        self.canvas.update_display(self.grid)


def main():
    """Spustí aplikaci Hopfieldovy sítě."""
    root = tk.Tk()
    root.title("Hopfieldova síť")
    HopfieldApp(root, grid_size=5, cell_size=50)
    root.mainloop()


if __name__ == "__main__":
    main()
