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

    def __init__(self, master, grid_size, min_cell_size, on_cell_toggle):
        super().__init__(master, bg='white')

        self.current_grid = None
        self.grid_size = grid_size
        self.min_cell_size = min_cell_size
        self.on_cell_toggle = on_cell_toggle
        self.cells = []
        self.cell_size = min_cell_size

        self.bind("<Configure>", self.on_resize)
        self._create_grid()

    def on_resize(self, event):
        """Reaguje na změnu velikosti plátna."""
        width, height = event.width, event.height
        new_cell_size = min(width, height) // self.grid_size

        new_cell_size = max(new_cell_size, self.min_cell_size)  # Zajištění minimální velikosti

        if new_cell_size != self.cell_size:
            self.cell_size = new_cell_size
            self.delete("all")  # po roztažení plátna vymažeme vše a znovu vytvoříme mřížku
            self.cells = []
            self._create_grid()

            if hasattr(self, 'current_grid') and self.current_grid is not None:
                self.update_display(self.current_grid)

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
        self.current_grid = grid  # Uložíme aktuální stav pro případné překreslení
        if not self.cells:
            return  # Pokud nejsou buňky vytvořeny, nelze aktualizovat

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                if idx < len(self.cells):  # Kontrola pro případ, že by se velikost gridu změnila
                    color = 'black' if grid[i][j] == 1 else 'white'
                    self.itemconfig(self.cells[idx], fill=color)


class ControlPanel(tk.Frame):
    """Panel s ovládacími tlačítky."""

    def __init__(self, master, callbacks):
        super().__init__(master)

        self.buttons = []
        button_texts = [
            ("Save Pattern", callbacks['save_pattern'], "green", "white"),
            ("Show Saved Pattern", callbacks['show_pattern'], "blue", "white"),
            ("Clear Grid", callbacks['clear'], "red", "white"),
            ("Recover Synchronously", callbacks['sync'], "yellow", "black"),
            ("Recover Asynchronously", callbacks['async'], "yellow", "black")
        ]

        self.columnconfigure(0, weight=1)  # Umožníme roztahování sloupce

        for i, (text, command, bg_color, fg_color) in enumerate(button_texts):
            btn = tk.Button(self, text=text, command=command,
                            bg=bg_color, fg=fg_color,
                            activebackground=bg_color, activeforeground=fg_color,
                            padx=10, pady=5)
            btn.grid(row=i, column=0, sticky="ew", padx=10, pady=5)
            self.buttons.append(btn)

        self.configure(width=200)  # Nastavení minimální šířky panelu

        self.rowconfigure(len(button_texts), weight=1)  # Tlačítka se mohou roztahovat


class HopfieldApp:
    """Hlavní aplikace pro práci s Hopfieldovou sítí."""

    def __init__(self, master, grid_size=5, min_cell_size=30):
        self.master = master
        self.grid_size = grid_size
        self.min_cell_size = min_cell_size

        self.grid = np.zeros((grid_size, grid_size), np.int32)
        self.network = HopfieldNetwork(grid_size)
        self.current_pattern_index = -1

        # Nastavení hlavního okna
        master.title("Hopfield Network Pattern Recognition")
        master.minsize(500, 400)  # Minimální velikost okna

        # Hlavní rozdělení okna na dvě části
        self.main_frame = tk.Frame(master)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Umožníme, aby se obě části roztahovaly
        self.main_frame.columnconfigure(0, weight=3)  # Canvas dostane více prostoru
        self.main_frame.columnconfigure(1, weight=1)  # Panel s tlačítky méně prostoru
        self.main_frame.rowconfigure(0, weight=1)

        # Vytvoření GUI
        callbacks = {
            'save_pattern': self.add_pattern,
            'show_pattern': self.next_pattern,
            'clear': self.clear_grid,
            'sync': self.recover_sync,
            'async': self.recover_async
        }

        # Canvas frame - levá část
        self.canvas_frame = tk.Frame(self.main_frame)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.canvas = GridCanvas(self.canvas_frame, grid_size, min_cell_size, self.toggle_cell)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Control panel - pravá část
        self.control_panel = ControlPanel(self.main_frame, callbacks)
        self.control_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

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
    root = tk.Tk()
    root.geometry("700x500")
    HopfieldApp(root, grid_size=10, min_cell_size=30)
    root.mainloop()


if __name__ == "__main__":
    main()
