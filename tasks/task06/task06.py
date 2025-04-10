import tkinter as tk
from tkinter import ttk
import math


class LSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("L-Systems Fractal Generator")

        # Hlavní rozdělení na levou část (plátno) a pravou část (kontroly)
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Levá část - plátno pro kreslení
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.left_frame, width=800, height=600, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pravá část - ovládací prvky
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Example buttons
        self.example_frame = ttk.LabelFrame(self.right_frame, text="Examples")
        self.example_frame.pack(fill=tk.X, padx=5, pady=5)

        self.draw_buttons = []
        for i in range(4):
            btn = ttk.Button(self.example_frame, text=f"Draw Example {i + 1}",
                             command=lambda idx=i + 1: self.load_example(idx))
            btn.pack(fill=tk.X, padx=5, pady=2)
            self.draw_buttons.append(btn)

        # Custom settings frame
        self.custom_frame = ttk.LabelFrame(self.right_frame, text="Custom L-System")
        self.custom_frame.pack(fill=tk.X, padx=5, pady=10)

        # Position and angle controls
        param_frame = ttk.Frame(self.custom_frame)
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(param_frame, text="Starting X:").grid(row=0, column=0, sticky="e", pady=2)
        self.start_x_var = tk.IntVar(value=400)
        self.start_x_entry = ttk.Entry(param_frame, textvariable=self.start_x_var, width=8)
        self.start_x_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(param_frame, text="Starting Y:").grid(row=1, column=0, sticky="e", pady=2)
        self.start_y_var = tk.IntVar(value=300)
        self.start_y_entry = ttk.Entry(param_frame, textvariable=self.start_y_var, width=8)
        self.start_y_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(param_frame, text="Starting Angle (deg):").grid(row=2, column=0, sticky="e", pady=2)
        self.start_angle_var = tk.StringVar()
        self.start_angle_entry = ttk.Entry(param_frame, textvariable=self.start_angle_var, width=8)
        self.start_angle_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        # L-System parameters
        ls_frame = ttk.Frame(self.custom_frame)
        ls_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(ls_frame, text="Axiom:").grid(row=0, column=0, sticky="e", pady=2)
        self.axiom_var = tk.StringVar()
        self.axiom_entry = ttk.Entry(ls_frame, textvariable=self.axiom_var, width=20)
        self.axiom_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(ls_frame, text="Rule:").grid(row=1, column=0, sticky="e", pady=2)
        self.rule_var = tk.StringVar()
        self.rule_entry = ttk.Entry(ls_frame, textvariable=self.rule_var, width=20)
        self.rule_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(ls_frame, text="Angle (deg):").grid(row=2, column=0, sticky="e", pady=2)
        self.angle_var = tk.StringVar()
        self.angle_entry = ttk.Entry(ls_frame, textvariable=self.angle_var, width=8)
        self.angle_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(ls_frame, text="Nesting:").grid(row=3, column=0, sticky="e", pady=2)
        self.nesting_var = tk.IntVar(value=3)
        self.nesting_entry = ttk.Entry(ls_frame, textvariable=self.nesting_var, width=8)
        self.nesting_entry.grid(row=3, column=1, sticky="w", padx=5, pady=2)

        ttk.Label(ls_frame, text="Line Size:").grid(row=4, column=0, sticky="e", pady=2)
        self.line_size_var = tk.IntVar(value=5)
        self.line_size_entry = ttk.Entry(ls_frame, textvariable=self.line_size_var, width=8)
        self.line_size_entry.grid(row=4, column=1, sticky="w", padx=5, pady=2)

        # Action buttons
        btn_frame = ttk.Frame(self.right_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.draw_custom_btn = ttk.Button(btn_frame, text="Draw Custom", command=self.draw_custom)
        self.draw_custom_btn.pack(fill=tk.X, padx=5, pady=2)

        self.clear_btn = ttk.Button(btn_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_btn.pack(fill=tk.X, padx=5, pady=2)

        # Initialize example systems
        self.examples = {
            1: {
                "axiom": "F+F+F+F",
                "rule": "F->F+F-F-FF+F+F-F",
                "angle": "90",
                "nesting": 3,
                "line_size": 5,
                "start_x": 400,
                "start_y": 300,
                "start_angle": "0"
            },
            2: {
                "axiom": "F++F++F",
                "rule": "F->F+F--F+F",
                "angle": "60",
                "nesting": 3,
                "line_size": 14,
                "start_x": 200,
                "start_y": 200,
                "start_angle": ""
            },
            3: {
                "axiom": "F",
                "rule": "F->F[+F]F[-F]F",
                "angle": "25.714",  # π/7 ≈ 25.714°
                "nesting": 5,
                "line_size": 4,
                "start_x": 500,
                "start_y": 500,
                "start_angle": ""
            },
            4: {
                "axiom": "F",
                "rule": "F->FF+[+F-F-F]-[-F+F+F]",
                "angle": "22.5",  # π/8 ≈ 22.5°
                "nesting": 3,
                "line_size": 14,
                "start_x": 200,
                "start_y": 200,
                "start_angle": ""
            }
        }

    def clear_canvas(self):
        self.canvas.delete("all")

    def load_example(self, example_num):
        example = self.examples[example_num]
        self.axiom_var.set(example["axiom"])
        self.rule_var.set(example["rule"])
        self.angle_var.set(example["angle"])
        self.nesting_var.set(example["nesting"])
        self.line_size_var.set(example["line_size"])
        self.start_x_var.set(example["start_x"])
        self.start_y_var.set(example["start_y"])
        self.start_angle_var.set(example["start_angle"])
        self.draw_custom()

    def draw_custom(self):
        axiom = self.axiom_var.get()
        rule = self.rule_var.get()
        angle_deg = self.angle_var.get()
        nesting = self.nesting_var.get()
        line_size = self.line_size_var.get()
        start_x = self.start_x_var.get()
        start_y = self.start_y_var.get()
        start_angle = self.start_angle_var.get()

        # Convert angle to radians
        try:
            angle = math.radians(float(angle_deg)) if angle_deg else math.pi / 2
        except ValueError:
            angle = math.pi / 2

        # Set default start angle
        try:
            start_angle_rad = math.radians(float(start_angle)) if start_angle else 0
        except ValueError:
            start_angle_rad = 0

        # Generate the L-system string
        l_string = self.generate_l_string(axiom, rule, nesting)

        # Draw the L-system
        self.draw_l_string(l_string, angle, line_size, start_x, start_y, start_angle_rad)

    def generate_l_string(self, axiom, rule, nesting):
        current = axiom
        rule_from, rule_to = rule.split("->") if "->" in rule else ("F", "")

        for _ in range(nesting):
            next_str = []
            for char in current:
                if char == rule_from:
                    next_str.append(rule_to)
                else:
                    next_str.append(char)
            current = "".join(next_str)

        return current

    def draw_l_string(self, l_string, angle, line_length, start_x, start_y, start_angle):
        x, y = start_x, start_y
        current_angle = start_angle
        stack = []

        for char in l_string:
            if char == 'F':
                # Draw forward
                new_x = x + line_length * math.cos(current_angle)
                new_y = y + line_length * math.sin(current_angle)
                self.canvas.create_line(x, y, new_x, new_y, fill="black")
                x, y = new_x, new_y
            elif char == 'b':
                # Move forward without drawing
                new_x = x + line_length * math.cos(current_angle)
                new_y = y + line_length * math.sin(current_angle)
                x, y = new_x, new_y
            elif char == '+':
                # Turn right
                current_angle += angle
            elif char == '-':
                # Turn left
                current_angle -= angle
            elif char == '[':
                # Push current state to stack
                stack.append((x, y, current_angle))
            elif char == ']':
                # Pop state from stack
                if stack:
                    x, y, current_angle = stack.pop()


if __name__ == "__main__":
    root = tk.Tk()
    app = LSystemApp(root)
    root.mainloop()
