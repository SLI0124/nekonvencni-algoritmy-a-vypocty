import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import random


class FractalLandscapeApp:
    def __init__(self, root):
        self.color_preview = None
        self.canvas = None
        self.root = root
        self.root.title("Fractal Landscape Generator")

        # Velikost plátna
        self.canvas_width = 800
        self.canvas_height = 600

        # Proměnné pro parametry
        self.start_x = tk.DoubleVar(value=0.0)
        self.start_y = tk.DoubleVar(value=300.0)
        self.end_x = tk.DoubleVar(value=800.0)
        self.end_y = tk.DoubleVar(value=300.0)
        self.iterations = tk.IntVar(value=8)
        self.offset_size = tk.DoubleVar(value=100.0)
        self.roughness = tk.DoubleVar(value=0.5)

        self.landscape_color = "#228b22"  # Zelená
        self.outline_color = "black"

        self.saved_layers = []
        self.landscape_points = []

        self.create_widgets()

    def create_widgets(self):
        # Hlavní rámec
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Rámec pro plátno
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height,
                                background="white", bd=2, relief=tk.SUNKEN)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Rámec pro ovládací prvky
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Parametry
        input_frame = ttk.LabelFrame(control_frame, text="Parametry", padding=(10, 5))
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(input_frame, text="Počáteční X pozice").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.start_x, width=10).grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Počáteční Y pozice").grid(row=1, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.start_y, width=10).grid(row=1, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Koncová X pozice").grid(row=2, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.end_x, width=10).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Koncová Y pozice").grid(row=3, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.end_y, width=10).grid(row=3, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Počet iterací").grid(row=4, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.iterations, width=10).grid(row=4, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Velikost výchylky").grid(row=5, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.offset_size, width=10).grid(row=5, column=1, padx=5, pady=2)

        ttk.Label(input_frame, text="Drsnost (0.1-0.9)").grid(row=6, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.roughness, width=10).grid(row=6, column=1, padx=5, pady=2)

        # Výběr barvy
        ttk.Label(input_frame, text="Barva krajiny:").grid(row=7, column=0, sticky="w")
        color_frame = ttk.Frame(input_frame)
        color_frame.grid(row=7, column=1, sticky="w", padx=5, pady=2)

        self.color_preview = tk.Canvas(color_frame, width=30, height=20, bd=1, relief=tk.SUNKEN)
        self.color_preview.pack(side=tk.LEFT, padx=2)
        self.color_preview.create_rectangle(0, 0, 30, 20, fill=self.landscape_color, outline="black")

        ttk.Button(color_frame, text="Změnit", command=self.choose_fill_color).pack(side=tk.LEFT, padx=2)

        # Tlačítka pro ovládání - vertikální uspořádání
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Generovat krajinu", command=self.generate_landscape).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Přidat vrstvu", command=self.add_layer).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Vyčistit plátno", command=self.clear_canvas).pack(fill=tk.X, pady=2)

    def choose_fill_color(self):
        """Dialog pro výběr barvy krajiny"""
        color = colorchooser.askcolor(title="Vyberte barvu krajiny", initialcolor=self.landscape_color)
        if color[1]:
            self.landscape_color = color[1]
            self.color_preview.delete("all")
            self.color_preview.create_rectangle(0, 0, 30, 20, fill=self.landscape_color, outline="black")

            if self.landscape_points:
                self.draw_landscape()

    def fractal_landscape(self, iterations, start_x, start_y, end_x, end_y, height_variation, roughness=0.5):
        """Generuje fraktální krajinu metodou půlení úseček s náhodným posunutím"""
        landscape = [(start_x, start_y), (end_x, end_y)]  # Počáteční přímka

        for _ in range(iterations):
            new_landscape = []
            for i in range(len(landscape) - 1):
                x1, y1 = landscape[i]
                x2, y2 = landscape[i + 1]

                # Středový bod
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                # Náhodné posunutí středu
                if random.random() < 0.5:
                    mid_y += random.uniform(0, height_variation)
                else:
                    mid_y -= random.uniform(0, height_variation)

                new_landscape.append((x1, y1))
                new_landscape.append((mid_x, mid_y))

                if i == len(landscape) - 2:
                    new_landscape.append((x2, y2))

            landscape = new_landscape
            # Zmenšení výchylky s každou iterací
            height_variation *= roughness

        return landscape

    def generate_landscape(self):
        try:
            iterations = self.iterations.get()
            start_x = self.start_x.get()
            start_y = self.start_y.get()
            end_x = self.end_x.get()
            end_y = self.end_y.get()
            offset = self.offset_size.get()
            roughness = self.roughness.get()

            # Validace
            if iterations < 1 or offset <= 0 or roughness <= 0 or roughness >= 1:
                messagebox.showerror("Chyba", "Neplatné hodnoty parametrů")
                return

            self.landscape_points = self.fractal_landscape(
                iterations, start_x, start_y, end_x, end_y, offset, roughness
            )

            self.draw_landscape()

        except Exception as e:
            messagebox.showerror("Chyba", f"Chyba: {str(e)}")

    def add_layer(self):
        """Přidá aktuální krajinu jako novou vrstvu"""
        if not self.landscape_points:
            messagebox.showinfo("Informace", "Nejprve vygenerujte krajinu.")
            return

        layer = {
            'points': self.landscape_points.copy(),
            'color': self.landscape_color,
            'start_x': self.start_x.get(),
            'end_x': self.end_x.get()
        }
        self.saved_layers.append(layer)

        # Přidání vrstvy bez smazání plátna
        self.draw_landscape(preserve_existing=True)
        self.landscape_points = []

        messagebox.showinfo("Informace", f"Vrstva přidána. Celkem vrstev: {len(self.saved_layers)}")

    def draw_landscape(self, preserve_existing=False):
        """Vykreslí krajinu i s uloženými vrstvami"""
        if not self.landscape_points and not self.saved_layers:
            return

        if not preserve_existing:
            self.canvas.delete("all")

            # Vykreslení všech uložených vrstev
            for layer in self.saved_layers:
                self._draw_layer(layer)

        # Vykreslení aktuální krajiny
        if self.landscape_points:
            current_layer = {
                'points': self.landscape_points,
                'color': self.landscape_color,
                'start_x': self.start_x.get(),
                'end_x': self.end_x.get()
            }
            self._draw_layer(current_layer)

    def _draw_layer(self, layer):
        """Vykreslí jednu vrstvu krajiny"""
        points = []
        for x, y in layer['points']:
            points.extend([x, y])

        # Uzavřený polygon pro výplň
        fill_points = list(points)
        fill_points.extend([layer['end_x'], self.canvas_height])
        fill_points.extend([layer['start_x'], self.canvas_height])

        # Vykreslení
        self.canvas.create_polygon(fill_points, fill=layer['color'], outline="")
        self.canvas.create_line(points, fill="black", width=2)

    def clear_canvas(self):
        """Vyčistí plátno a všechna data krajiny"""
        self.canvas.delete("all")
        self.landscape_points = []
        self.saved_layers = []


if __name__ == "__main__":
    root = tk.Tk()
    app = FractalLandscapeApp(root)
    root.mainloop()
