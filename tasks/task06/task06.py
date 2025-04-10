import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import math
import json
import os


class LSystemApp:
    def __init__(self, root):
        # Inicializace hlavního okna aplikace
        self.root = root
        self.root.title("L-Systems Fractal Generator")

        # Hlavní rámec aplikace, rozdělený na levou a pravou část
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Levá část - plátno pro kreslení fraktálů
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.left_frame, width=800, height=600, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Pravá část - ovládací prvky
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Sekce s příklady L-systémů
        self.example_frame = ttk.LabelFrame(self.right_frame, text="Examples")
        self.example_frame.pack(fill=tk.X, padx=5, pady=5)

        # Tlačítka pro vykreslení předdefinovaných příkladů
        self.draw_buttons = []
        for i in range(4):
            btn = ttk.Button(self.example_frame, text=f"Draw Example {i + 1}",
                             command=lambda idx=i + 1: self.load_example(idx))
            btn.pack(fill=tk.X, padx=5, pady=2)
            self.draw_buttons.append(btn)

        # Sekce pro vlastní nastavení L-systému
        self.custom_frame = ttk.LabelFrame(self.right_frame, text="Custom L-System")
        self.custom_frame.pack(fill=tk.X, padx=5, pady=10)

        # Ovládací prvky pro nastavení počáteční pozice a úhlu
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

        # Ovládací prvky pro parametry L-systému
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

        # Tlačítka pro vykreslení a vyčištění plátna
        btn_frame = ttk.Frame(self.right_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.draw_custom_btn = ttk.Button(btn_frame, text="Draw Custom", command=self.draw_custom)
        self.draw_custom_btn.pack(fill=tk.X, padx=5, pady=2)

        self.clear_btn = ttk.Button(btn_frame, text="Clear Canvas", command=self.clear_canvas)
        self.clear_btn.pack(fill=tk.X, padx=5, pady=2)

        # Sekce pro ukládání a načítání L-systémů
        self.io_frame = ttk.LabelFrame(self.right_frame, text="Save & Load")
        self.io_frame.pack(fill=tk.X, padx=5, pady=10)

        # Tlačítka pro ukládání a načítání
        self.save_btn = ttk.Button(self.io_frame, text="Save as JSON", command=self.save_lsystem)
        self.save_btn.pack(fill=tk.X, padx=5, pady=2)

        self.load_btn = ttk.Button(self.io_frame, text="Load from JSON", command=self.load_lsystem)
        self.load_btn.pack(fill=tk.X, padx=5, pady=2)

        # Inicializace předdefinovaných příkladů L-systémů
        self.examples = {
            1: {
                # Příklad čtvercového fraktálu
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
                # Příklad trojúhelníkového fraktálu
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

        # Vytvoření adresáře pro ukládání L-systémů, pokud neexistuje
        self.systems_dir = "lsystems"
        if not os.path.exists(self.systems_dir):
            os.makedirs(self.systems_dir)

    def clear_canvas(self):
        # Vyčištění plátna
        self.canvas.delete("all")

    def load_example(self, example_num):
        # Načtení předdefinovaného příkladu L-systému
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
        # Vykreslení vlastního L-systému na základě zadaných parametrů
        axiom = self.axiom_var.get()
        rule = self.rule_var.get()
        angle_deg = self.angle_var.get()
        nesting = self.nesting_var.get()
        line_size = self.line_size_var.get()
        start_x = self.start_x_var.get()
        start_y = self.start_y_var.get()
        start_angle = self.start_angle_var.get()

        # Převod úhlu na radiány
        try:
            angle = math.radians(float(angle_deg)) if angle_deg else math.pi / 2
        except ValueError:
            angle = math.pi / 2

        # Nastavení výchozího úhlu
        try:
            start_angle_rad = math.radians(float(start_angle)) if start_angle else 0
        except ValueError:
            start_angle_rad = 0

        # Generování řetězce L-systému
        l_string = self.generate_l_string(axiom, rule, nesting)

        # Vykreslení řetězce L-systému
        self.draw_l_string(l_string, angle, line_size, start_x, start_y, start_angle_rad)

    def generate_l_string(self, axiom, rule, nesting):
        # Generování řetězce L-systému na základě axiómu a pravidla
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
        # Vykreslení řetězce L-systému na plátno
        x, y = start_x, start_y
        current_angle = start_angle
        stack = []

        for char in l_string:
            if char == 'F':
                # Kreslení čáry vpřed
                new_x = x + line_length * math.cos(current_angle)
                new_y = y + line_length * math.sin(current_angle)
                self.canvas.create_line(x, y, new_x, new_y, fill="black")
                x, y = new_x, new_y
            elif char == 'b':
                # Posun vpřed bez kreslení
                new_x = x + line_length * math.cos(current_angle)
                new_y = y + line_length * math.sin(current_angle)
                x, y = new_x, new_y
            elif char == '+':
                # Otočení doprava
                current_angle += angle
            elif char == '-':
                # Otočení doleva
                current_angle -= angle
            elif char == '[':
                # Uložení aktuálního stavu na zásobník
                stack.append((x, y, current_angle))
            elif char == ']':
                # Obnovení stavu ze zásobníku
                if stack:
                    x, y, current_angle = stack.pop()

    def save_lsystem(self):
        """Uloží aktuální L-systém jako JSON do adresáře lsystems s automaticky generovaným názvem"""

        # Získání dat L-systému
        lsystem_data = {
            "axiom": self.axiom_var.get(),
            "rule": self.rule_var.get(),
            "angle": self.angle_var.get(),
            "nesting": self.nesting_var.get(),
            "line_size": self.line_size_var.get(),
            "start_x": self.start_x_var.get(),
            "start_y": self.start_y_var.get(),
            "start_angle": self.start_angle_var.get()
        }

        # Kontrola zda jsou vyplněna základní data
        axiom = lsystem_data["axiom"]
        rule = lsystem_data["rule"]

        if not axiom or not rule:
            messagebox.showerror("Chyba", "Vyplňte alespoň axiom a pravidlo")
            return

        # Vytvoření základního názvu z prvních znaků pravidla a axiomu
        base_name = f"ls_{axiom[:3]}_{rule.split('->')[0]}"
        base_name = base_name.replace("->", "").replace("[", "").replace("]", "")
        base_name = base_name.replace("+", "p").replace("-", "m")

        # Zjištění nejvyššího existujícího čísla pro daný základ názvu
        count = 1
        while os.path.exists(os.path.join(self.systems_dir, f"{base_name}_{count}.json")):
            count += 1

        # Vytvoření konečného názvu souboru
        filename = f"{base_name}_{count}.json"
        filepath = os.path.join(self.systems_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(lsystem_data, f, indent=2)
            messagebox.showinfo("Úspěch", f"L-systém byl uložen jako {filename}")
        except Exception as e:
            messagebox.showerror("Chyba při ukládání", f"Nastala chyba: {str(e)}")

    def load_lsystem(self):
        """Načte L-systém z JSON souboru"""
        filetypes = [("JSON files", "*.json"), ("All files", "*.*")]
        filepath = filedialog.askopenfilename(
            initialdir=self.systems_dir,
            title="Vyberte soubor L-systému",
            filetypes=filetypes
        )

        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                lsystem_data = json.load(f)

            # Nastavení hodnot do rozhraní
            self.axiom_var.set(lsystem_data.get("axiom", ""))
            self.rule_var.set(lsystem_data.get("rule", ""))
            self.angle_var.set(lsystem_data.get("angle", ""))
            self.nesting_var.set(lsystem_data.get("nesting", 3))
            self.line_size_var.set(lsystem_data.get("line_size", 5))
            self.start_x_var.set(lsystem_data.get("start_x", 400))
            self.start_y_var.set(lsystem_data.get("start_y", 300))
            self.start_angle_var.set(lsystem_data.get("start_angle", ""))

            # Nastavení názvu L-systému podle názvu souboru
            filename = os.path.basename(filepath)
            system_name = os.path.splitext(filename)[0]
            self.system_name_var.set(system_name)

            # Automatické vykreslení načteného L-systému
            self.draw_custom()

            messagebox.showinfo("Úspěch", f"L-systém {system_name} byl úspěšně načten")
        except Exception as e:
            messagebox.showerror("Chyba při načítání", f"Nastala chyba: {str(e)}")


if __name__ == "__main__":
    # Spuštění aplikace
    root = tk.Tk()
    app = LSystemApp(root)
    root.mainloop()
