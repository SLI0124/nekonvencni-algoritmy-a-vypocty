import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, colorchooser


class FractalLandscapeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fractal Landscape Generator")

        # Variables for parameters
        self.start_x = tk.DoubleVar(value=0.0)
        self.start_y = tk.DoubleVar(value=0.0)
        self.end_x = tk.DoubleVar(value=10.0)
        self.end_y = tk.DoubleVar(value=0.0)
        self.iterations = tk.IntVar(value=5)
        self.offset_size = tk.DoubleVar(value=1.0)
        self.color = "#000000"

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.root, text="Parameters", padding=(10, 5))
        input_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Start X position
        ttk.Label(input_frame, text="Start X position (float)").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.start_x).grid(row=0, column=1, padx=5, pady=2)

        # Start Y position
        ttk.Label(input_frame, text="Start Y position (float)").grid(row=1, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.start_y).grid(row=1, column=1, padx=5, pady=2)

        # End X position
        ttk.Label(input_frame, text="End X position (float)").grid(row=2, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.end_x).grid(row=2, column=1, padx=5, pady=2)

        # End Y position
        ttk.Label(input_frame, text="End Y position (float)").grid(row=3, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.end_y).grid(row=3, column=1, padx=5, pady=2)

        # Number of iterations
        ttk.Label(input_frame, text="The number of iteration (int)").grid(row=4, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.iterations).grid(row=4, column=1, padx=5, pady=2)

        # Offset size
        ttk.Label(input_frame, text="Offset size (float)").grid(row=5, column=0, sticky="w")
        ttk.Entry(input_frame, textvariable=self.offset_size).grid(row=5, column=1, padx=5, pady=2)

        # Color selection
        ttk.Label(input_frame, text="Selected color is").grid(row=6, column=0, sticky="w")
        color_display = ttk.Label(input_frame, text=self.color, background=self.color)
        color_display.grid(row=6, column=1, padx=5, pady=2, sticky="w")

        def choose_color():
            color = colorchooser.askcolor(title="Choose color")[1]
            if color:
                self.color = color
                color_display.config(text=color, background=color)

        ttk.Button(input_frame, text="Pick a color", command=choose_color).grid(row=7, column=0, columnspan=2, pady=5)

        # Button frame
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        ttk.Button(button_frame, text="Draw", command=self.draw_landscape).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear canvas", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)

        # Plot frame
        plot_frame = ttk.Frame(self.root)
        plot_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.fig, self.ax = plt.subplots(figsize=(8, 8))  # Změna figsize na (8, 8)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)

    def fractal_landscape(self, iterations, start_x, start_y, end_x, end_y, height_variation):
        landscape = [(start_x, start_y), (end_x, end_y)]  # Initial line

        for _ in range(iterations):
            new_landscape = []
            for i in range(len(landscape) - 1):
                x1, y1 = landscape[i]
                x2, y2 = landscape[i + 1]

                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2

                if np.random.rand() < 0.5:
                    mid_y += np.random.uniform(-height_variation, height_variation)
                else:
                    mid_y -= np.random.uniform(-height_variation, height_variation)

                new_landscape.append((x1, y1))
                new_landscape.append((mid_x, mid_y))

                if i == len(landscape) - 2:
                    new_landscape.append((x2, y2))

            landscape = new_landscape
            height_variation *= 0.5  # Decrease variation with each iteration

        return landscape

    def draw_landscape(self):
        try:
            iterations = self.iterations.get()
            start_x = self.start_x.get()
            start_y = self.start_y.get()
            end_x = self.end_x.get()
            end_y = self.end_y.get()
            offset = self.offset_size.get()

            landscape = self.fractal_landscape(iterations, start_x, start_y, end_x, end_y, offset)

            x = [point[0] for point in landscape]
            y = [point[1] for point in landscape]

            self.ax.plot(x, y, color=self.color)
            # Adjust fill_between lower bound dynamically
            fill_min_y = min(y) - (max(y) - min(y)) * 0.1 if max(y) != min(y) else min(y) - 1  # Add small padding below
            self.ax.fill_between(x, y, fill_min_y, color=self.color, alpha=0.7)

            self.ax.set_title("Fractal Landscape")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.grid(True)

            # Set axis limits to fit data tightly
            self.ax.set_xlim(min(x), max(x))
            # Add a small padding to y limits for better visualization of the fill
            y_padding = (max(y) - min(y)) * 0.05 if max(y) != min(y) else 0.5
            self.ax.set_ylim(fill_min_y, max(y) + y_padding)

            self.canvas.draw()
        except Exception as e:
            tk.messagebox.showerror("Error", f"Invalid input: {str(e)}")

    def clear_canvas(self):
        self.ax.clear()
        # Reapply default settings after clearing if needed, or set limits again
        self.ax.set_title("Fractal Landscape")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)
        self.canvas.draw()


if __name__ == "__main__":
    root = tk.Tk()
    app = FractalLandscapeApp(root)
    root.mainloop()
