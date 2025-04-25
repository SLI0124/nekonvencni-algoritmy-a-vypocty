import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser
import random


class FractalLandscapeApp:
    def __init__(self, root):
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

        # Barva krajiny
        self.landscape_color = "#228b22"  # Zelená
        self.outline_color = "black"  # Barva obrysu - vždy černá

        # Vytvoření GUI prvků
        self.create_widgets()

        # Seznam pro uchování bodů krajiny
        self.landscape_points = []

    def create_widgets(self):
        # Rozdělení okna na dvě části - plátno a ovládací panel
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Rámec pro plátno
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tkinter Canvas pro vykreslování
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_width, height=self.canvas_height,
                                background="white", bd=2, relief=tk.SUNKEN)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Rámec pro ovládací prvky
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Rámec pro parametry
        input_frame = ttk.LabelFrame(control_frame, text="Parametry", padding=(10, 5))
        input_frame.pack(fill=tk.X, padx=5, pady=5)

        # Počáteční X pozice
        ttk.Label(input_frame, text="Počáteční X pozice").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.start_x, width=10).grid(row=0, column=1, padx=5, pady=2)

        # Počáteční Y pozice
        ttk.Label(input_frame, text="Počáteční Y pozice").grid(row=1, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.start_y, width=10).grid(row=1, column=1, padx=5, pady=2)

        # Koncová X pozice
        ttk.Label(input_frame, text="Koncová X pozice").grid(row=2, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.end_x, width=10).grid(row=2, column=1, padx=5, pady=2)

        # Koncová Y pozice
        ttk.Label(input_frame, text="Koncová Y pozice").grid(row=3, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.end_y, width=10).grid(row=3, column=1, padx=5, pady=2)

        # Počet iterací
        ttk.Label(input_frame, text="Počet iterací").grid(row=4, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.iterations, width=10).grid(row=4, column=1, padx=5, pady=2)

        # Velikost výchylky
        ttk.Label(input_frame, text="Velikost výchylky").grid(row=5, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.offset_size, width=10).grid(row=5, column=1, padx=5, pady=2)

        # Drsnost terénu
        ttk.Label(input_frame, text="Drsnost (0.1-0.9)").grid(row=6, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.roughness, width=10).grid(row=6, column=1, padx=5, pady=2)

        # Barva výplně
        ttk.Label(input_frame, text="Barva krajiny:").grid(row=7, column=0, sticky="w")
        color_frame = ttk.Frame(input_frame)
        color_frame.grid(row=7, column=1, sticky="w", padx=5, pady=2)

        self.fill_color_preview = ttk.Label(color_frame, text="■■■", foreground=self.landscape_color,
                                            font=("Arial", 12))
        self.fill_color_preview.pack(side=tk.LEFT, padx=2)
        ttk.Button(color_frame, text="Změnit", command=self.choose_fill_color).pack(side=tk.LEFT, padx=2)

        # Tlačítka pro vykreslení a vyčištění plátna
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Generovat krajinu", command=self.generate_landscape).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Vyčistit plátno", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)

    def choose_fill_color(self):
        """Otevře dialog pro výběr barvy krajiny."""
        color = colorchooser.askcolor(title="Vyberte barvu krajiny", initialcolor=self.landscape_color)
        if color[1]:  # Pokud uživatel vybral barvu a neklikl Cancel
            self.landscape_color = color[1]
            self.fill_color_preview.config(foreground=color[1])
            # Pokud už existuje krajina, překresli ji s novou barvou
            if self.landscape_points:
                self.draw_landscape()

    def fractal_landscape(self, iterations, start_x, start_y, end_x, end_y, height_variation, roughness=0.5):
        """Generuje fraktální krajinu pomocí metody přidávání středových bodů."""
        landscape = [(start_x, start_y), (end_x, end_y)]  # Počáteční přímka

        for _ in range(iterations):
            new_landscape = []
            for i in range(len(landscape) - 1):
                x1, y1 = landscape[i]
                x2, y2 = landscape[i + 1]

                # Najdi středový bod
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                # Aplikuj náhodnou výchylku středového bodu
                if random.random() < 0.5:
                    mid_y += random.uniform(0, height_variation)
                else:
                    mid_y -= random.uniform(0, height_variation)

                # Přidej body do nové krajiny
                new_landscape.append((x1, y1))
                new_landscape.append((mid_x, mid_y))

                # Přidej koncový bod poslední části
                if i == len(landscape) - 2:
                    new_landscape.append((x2, y2))

            landscape = new_landscape
            # Snížení výchylky pro každou iteraci podle nastavené drsnosti
            height_variation *= roughness

        return landscape

    def generate_landscape(self):
        try:
            # Získání parametrů z GUI
            iterations = self.iterations.get()
            start_x = self.start_x.get()
            start_y = self.start_y.get()
            end_x = self.end_x.get()
            end_y = self.end_y.get()
            offset = self.offset_size.get()
            roughness = self.roughness.get()

            # Kontrola platnosti parametrů
            if iterations < 1 or offset <= 0 or roughness <= 0 or roughness >= 1:
                messagebox.showerror("Chyba", "Neplatné hodnoty parametrů")
                return

            # Vyčištění plátna
            self.clear_canvas()

            # Generování krajiny
            self.landscape_points = self.fractal_landscape(
                iterations, start_x, start_y, end_x, end_y, offset, roughness
            )

            # Vykreslení terénu
            self.draw_landscape()

        except Exception as e:
            messagebox.showerror("Chyba", f"Chyba: {str(e)}")

    def draw_landscape(self):
        """Vykreslí jednoduchou krajinu jako jednu vrstvu."""
        if not self.landscape_points:
            return

        # Vyčisti plátno před překreslením
        self.canvas.delete("all")

        # Extrahování x a y souřadnic pro vykreslení
        points = []
        for x, y in self.landscape_points:
            points.extend([x, y])

        # Vytvoření uzavřeného polygonu pro vyplnění
        fill_points = list(points)  # Kopie bodů

        # Přidání bodů pro uzavření polygonu
        fill_points.extend([self.end_x.get(), self.canvas_height])
        fill_points.extend([self.start_x.get(), self.canvas_height])

        # Vykreslení vyplněné krajiny
        self.canvas.create_polygon(fill_points, fill=self.landscape_color, outline="")

        # Vykreslení obrysu krajiny
        self.canvas.create_line(points, fill="black", width=2)

    def clear_canvas(self):
        """Vyčistí plátno a resetuje data krajiny."""
        self.canvas.delete("all")
        self.landscape_points = []


if __name__ == "__main__":
    root = tk.Tk()
    app = FractalLandscapeApp(root)
    root.mainloop()
