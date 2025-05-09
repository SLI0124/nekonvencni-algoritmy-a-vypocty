import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk, ImageGrab
import matplotlib.pyplot as plt
import io
import imageio
import os

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_  # knihovna gym používá np.bool8, proto je třeba přidat alias


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        # Inicializace neuronové sítě pro Q-learning
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        # Dopředný průchod sítí
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        # Inicializace agenta pro DQN (Deep Q-Network)
        self.state_size = state_size  # Velikost vstupního stavu
        self.action_size = action_size  # Počet možných akcí
        self.memory = []  # Paměť pro ukládání zkušeností
        self.gamma = 0.95  # Discount faktor pro budoucí odměny
        self.epsilon = 1.0  # Míra průzkumu prostředí
        self.epsilon_min = 0.01  # Minimální míra průzkumu
        self.epsilon_decay = 0.995  # Rychlost snižování průzkumu
        self.learning_rate = 0.001  # Zvýšení rychlosti učení
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        # Vylepšená odměna: penalizace za úhel a pozici
        pole_angle = abs(state[2])
        position = abs(state[0])
        angle_penalty = 1.0 - (pole_angle / (np.pi / 4))  # větší penalizace za větší úhel
        position_penalty = 1.0 - (position / 2.4)  # penalizace za vzdálenost od středu
        reward += angle_penalty + position_penalty
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Výběr akce na základě stavu (epsilon-greedy strategie)
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        # Trénování modelu na základě náhodného vzorku zkušeností
        if len(self.memory) < batch_size:
            return

        # Výběr náhodné dávky zkušeností
        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        states, targets = [], []

        # Zpracování každé zkušenosti v dávce
        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)

            # Výpočet cílové hodnoty Q
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state)
            target_f = target_f.clone()  # Vytvoření kopie tensoru pro úpravy
            target_f[action] = torch.tensor(target, dtype=torch.float32)  # Zajištění správného typu
            states.append(state)
            targets.append(target_f)

        # Trénování modelu
        states = torch.stack(states)
        targets = torch.stack(targets)
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(states), targets)
        loss.backward()
        self.optimizer.step()

        # Snížení epsilon pro méně průzkumu v budoucnu
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PoleBalancingApp:
    def __init__(self, root):
        # Inicializace hlavní aplikace
        self.root = root
        self.root.title("Pole-Balancing Problem with Q-Learning")

        # Inicializace prostředí a agenta
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.gamma = 0.99  # Increase discount factor for long-term rewards
        self.learning_rate = 0.001  # Zvýšení rychlosti učení
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.agent.gamma = self.gamma
        self.agent.learning_rate = self.learning_rate

        # Konfigurační proměnné
        self.step_counter = None  # Počítadlo kroků
        self.episodes = 2000  # Počet epizod pro trénování
        self.batch_size = 64  # Zvýšení velikosti dávky
        self.is_training = False  # Indikátor, zda probíhá trénování
        self.is_visualizing = False  # Indikátor, zda probíhá vizualizace
        self.max_steps = 500  # Maximální počet kroků
        self.total_steps = 500  # Celkový počet kroků (nastavitelný)
        self.training_scores = []  # Seznam pro ukládání skóre během tréninku

        # Vytvoření adresářové struktury pro ukládání výsledků
        self.results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        self.graph_dir = os.path.join(self.results_dir, "graph")
        self.animation_dir = os.path.join(self.results_dir, "animation")

        # Vytvoření adresářů, pokud neexistují
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.graph_dir, exist_ok=True)
        os.makedirs(self.animation_dir, exist_ok=True)

        # GUI komponenty
        self.canvas = None  # Plátno pro vizualizaci
        self.graph_canvas = None  # Plátno pro graf
        self.status_text = None  # Textové pole pro zobrazení stavu
        self.visualize_btn = None  # Tlačítko pro vizualizaci
        self.stop_train_btn = None  # Tlačítko pro zastavení tréninku
        self.train_btn = None  # Tlačítko pro zahájení tréninku
        self.batch_entry = None  # Pole pro zadání velikosti dávky
        self.episodes_entry = None  # Pole pro zadání počtu epizod
        self.save_anim_btn = None  # Tlačítko pro uložení animace
        self.save_graph_btn = None  # Tlačítko pro uložení grafu

        # Vytvoření GUI
        self.create_widgets()

    def create_widgets(self):
        # Vytvoření a rozmístění prvků GUI

        # === Ovládací panel ===
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=(10, 5))
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Nastavení počtu epizod
        ttk.Label(control_frame, text="Training Episodes:").grid(row=0, column=0, sticky="w")
        self.episodes_entry = ttk.Entry(control_frame)
        self.episodes_entry.insert(0, str(self.episodes))
        self.episodes_entry.grid(row=0, column=1, sticky="ew")

        # Nastavení velikosti dávky
        ttk.Label(control_frame, text="Batch Size:").grid(row=1, column=0, sticky="w")
        self.batch_entry = ttk.Entry(control_frame)
        self.batch_entry.insert(0, "32")
        self.batch_entry.grid(row=1, column=1, sticky="ew")

        # Nastavení celkového počtu kroků
        ttk.Label(control_frame, text="Total Steps:").grid(row=2, column=0, sticky="w")
        self.total_steps_entry = ttk.Entry(control_frame)
        self.total_steps_entry.insert(0, str(self.total_steps))
        self.total_steps_entry.grid(row=2, column=1, sticky="ew")

        # Tlačítka pro ovládání
        self.train_btn = ttk.Button(control_frame, text="Start Training", command=self.start_training)
        self.train_btn.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")

        self.stop_train_btn = ttk.Button(control_frame, text="Stop Training", state="disabled",
                                         command=self.stop_training)
        self.stop_train_btn.grid(row=4, column=0, columnspan=2, pady=5, sticky="ew")

        self.visualize_btn = ttk.Button(control_frame, text="Visualize Solution", command=self.visualize_solution)
        self.visualize_btn.grid(row=5, column=0, columnspan=2, pady=5, sticky="ew")

        self.save_anim_btn = ttk.Button(control_frame, text="Save Animation", command=self.capture_animation)
        self.save_anim_btn.grid(row=6, column=0, columnspan=2, pady=5, sticky="ew")

        self.save_graph_btn = ttk.Button(control_frame, text="Save Training Graph", command=self.save_training_graph)
        self.save_graph_btn.grid(row=7, column=0, columnspan=2, pady=5, sticky="ew")

        # === Panel stavu ===
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=(10, 5))
        status_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Textové pole pro výpis stavu
        self.status_text = tk.Text(status_frame, height=10, width=50, state="disabled")
        self.status_text.grid(row=0, column=0, sticky="nsew")

        # Posuvník pro textové pole
        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.status_text.configure(yscrollcommand=scrollbar.set)

        # === Panel grafu ===
        graph_frame = ttk.LabelFrame(self.root, text="Training Graph", padding=(10, 5))
        graph_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Plátno pro graf
        self.graph_canvas = tk.Canvas(graph_frame, width=400, height=300, bg="white")
        self.graph_canvas.grid(row=0, column=0, sticky="nsew")

        # Přidání události pro změnu velikosti grafu
        self.graph_canvas.bind("<Configure>", self.on_canvas_resize)

        # === Panel vizualizace ===
        vis_frame = ttk.LabelFrame(self.root, text="Visualization", padding=(10, 5))
        vis_frame.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")

        # Plátno pro vizualizaci
        self.canvas = tk.Canvas(vis_frame, width=400, height=300, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Nastavení roztažitelnosti prvků
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        control_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        graph_frame.columnconfigure(0, weight=1)
        graph_frame.rowconfigure(0, weight=1)
        vis_frame.columnconfigure(0, weight=1)
        vis_frame.rowconfigure(0, weight=1)

    def log_message(self, message):
        # Zápis zprávy do textového pole stavu
        self.status_text.configure(state="normal")
        self.status_text.insert("end", message + "\n")
        self.status_text.configure(state="disabled")
        self.status_text.see("end")
        self.root.update()

    def start_training(self):
        # Spuštění trénování agenta
        if self.is_training:
            return

        # Načtení parametrů z uživatelského vstupu
        try:
            self.episodes = int(self.episodes_entry.get())
            self.batch_size = int(self.batch_entry.get())
            self.total_steps = int(self.total_steps_entry.get())
        except ValueError:
            self.log_message("Invalid input for episodes, batch size or total steps")
            return

        # Nastavení stavu aplikace na trénování
        self.is_training = True
        self.train_btn.config(state="disabled")
        self.stop_train_btn.config(state="normal")
        self.visualize_btn.config(state="disabled")

        # Inicializace nového agenta
        self.agent = DQNAgent(self.state_size, self.action_size)

        # Spuštění procesu trénování
        self.root.after(100, self.run_training)

    def stop_training(self):
        # Zastavení procesu trénování
        self.is_training = False
        self.train_btn.config(state="normal")
        self.stop_train_btn.config(state="disabled")
        self.visualize_btn.config(state="normal")

    def run_training(self):
        # Průběh trénování agenta
        scores = []
        self.training_scores = []  # Reset seznamu skóre

        # Iterace přes epizody
        for e in range(self.episodes):
            if not self.is_training:
                break

            # Reset prostředí na začátku epizody
            state = self.env.reset()[0]
            total_reward = 0
            self.step_counter = 0

            # Průběh jedné epizody
            while self.step_counter < self.total_steps:
                action = self.agent.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                # Vylepšená odměna
                pole_angle = abs(state[2])
                position = abs(state[0])
                angle_penalty = 1.0 - (pole_angle / (np.pi / 4))
                position_penalty = 1.0 - (position / 2.4)
                reward += angle_penalty + position_penalty
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += 1  # Skóre = počet kroků, kdy agent udržel tyč
                self.step_counter += 1
                if done:
                    break

            # Uložení skóre a učení agenta
            scores.append(total_reward)
            self.training_scores = scores.copy()  # Uložení kopie pro případné pozdější použití
            self.agent.replay(self.batch_size)

            # Výpis stavu trénování
            self.log_message(
                f"Episode: {e + 1}/{self.episodes}, Score: {total_reward}, Epsilon: {self.agent.epsilon:.2f}")

            # Průběžné vykreslování grafu
            if e % 10 == 0:
                self.plot_training_progress(scores)

            # Aktualizace GUI
            if e % 5 == 0:
                self.root.update()

        # Finální vykreslení grafu
        self.plot_training_progress(scores)
        self.stop_training()
        self.log_message("Training completed!")

    def plot_training_progress(self, scores):
        # Vykreslení grafu postupu trénování

        # Získání aktuální velikosti plátna pro graf
        canvas_width = self.graph_canvas.winfo_width() or 400
        canvas_height = self.graph_canvas.winfo_height() or 300

        # Přizpůsobení velikosti grafu podle velikosti plátna
        fig_width = max(4, canvas_width / 100)  # Převod pixelů na palce
        fig_height = max(3, canvas_height / 100)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        ax.plot(scores)
        ax.set_title('Training Progress')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.grid(True)

        # Přidání většího okraje pro popisky
        plt.tight_layout(pad=2.0)

        # Převod grafu na obrázek pro zobrazení na plátně
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)

        # Změna velikosti obrázku podle velikosti plátna
        img = img.resize((canvas_width, canvas_height), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.graph_canvas.delete("all")

        # Umístění obrázku na plátno
        self.graph_canvas.create_image(0, 0, anchor="nw", image=img)
        self.graph_canvas.image = img
        plt.close(fig)

    def save_training_graph(self):
        # Uložení trénovacího grafu do souboru
        if not self.training_scores:
            self.log_message("No training data available to save.")
            return

        try:
            # Zjištění počtu existujících grafů v adresáři
            existing_files = [f for f in os.listdir(self.graph_dir) if
                              f.startswith("training_graph_") and f.endswith(".png")]
            file_count = len(existing_files)

            # Vytvoření názvu souboru s pořadovým číslem
            file_path = os.path.join(self.graph_dir, f"training_graph_{file_count + 1}.png")

            # Vytvoření grafu ve vysokém rozlišení pro ukládání
            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            ax.plot(self.training_scores)
            ax.set_title('Training Progress')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Score')
            ax.grid(True)
            plt.tight_layout()

            plt.savefig(file_path)
            plt.close(fig)

            self.log_message(f"Training graph saved to {file_path}")
        except Exception as e:
            self.log_message(f"Error saving training graph: {str(e)}")

    def visualize_solution(self):
        # Spuštění vizualizace natrénovaného modelu
        if self.is_visualizing:
            return

        self.is_visualizing = True
        self.train_btn.config(state="disabled")
        self.visualize_btn.config(state="disabled")
        self.run_visualization(self.root, self.canvas)

    def draw_cartpole(self, canvas, state):
        # Vykreslení vozíku s tyčí na plátno
        cart_pos = state[0]  # Pozice vozíku
        pole_angle = state[2]  # Úhel tyče

        # Rozměry plátna a objektů
        canvas_width = canvas.winfo_width() or 400
        canvas_height = canvas.winfo_height() or 300
        cart_width = 50
        cart_height = 30
        pole_length = 100

        # Centrování scény na plátně
        # Mapování rozsahu pozice vozíku na vizuální prostor plátna
        scale_factor = 50
        max_cart_movement = 150  # Omezení maximálního posunu pro lepší viditelnost
        scaled_cart_pos = min(max(cart_pos * scale_factor, -max_cart_movement), max_cart_movement)

        # Výpočet pozice vozíku - centrovaná na plátně
        x_center = canvas_width / 2 + scaled_cart_pos
        y_cart = canvas_height / 2 + 50  # Umístění vozíku pod středem plátna

        cart_left = x_center - cart_width / 2
        cart_right = x_center + cart_width / 2
        cart_top = y_cart - cart_height / 2
        cart_bottom = y_cart + cart_height / 2

        # Vykreslení pozadí a základní čáry
        canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="white")
        canvas.create_line(0, y_cart + cart_height / 2, canvas_width, y_cart + cart_height / 2,
                           fill="gray", width=2, dash=(4, 2))

        # Vykreslení vozíku
        canvas.create_rectangle(cart_left, cart_top, cart_right, cart_bottom, fill="blue")

        # Výpočet koncových bodů tyče
        pole_start_x = x_center
        pole_start_y = cart_top
        pole_end_x = pole_start_x + pole_length * np.sin(pole_angle)
        pole_end_y = pole_start_y - pole_length * np.cos(pole_angle)

        # Vykreslení tyče
        canvas.create_line(pole_start_x, pole_start_y, pole_end_x, pole_end_y, width=5, fill="red")

    def run_visualization(self, window, canvas):
        # Spuštění vizualizace modelu v akci
        result = self.env.reset()
        state = result if not isinstance(result, tuple) else result[0]
        self.step_counter = 0
        self.log_message("Visualization started.")

        def visualize_step(state):
            # Funkce pro jeden krok vizualizace
            if self.step_counter >= self.total_steps:
                self.log_message(f"Visualization complete after {self.step_counter} out of {self.total_steps} steps.")
                self.is_visualizing = False
                self.train_btn.config(state="normal")
                self.visualize_btn.config(state="normal")
                return

            # Vykreslení aktuálního stavu
            canvas.delete("all")
            self.draw_cartpole(canvas, state)
            window.update()

            # Výběr akce a provedení kroku v prostředí
            action = self.agent.act(state)
            result = self.env.step(action)

            # Zpracování výsledku kroku (kompatibilita s různými verzemi gym)
            if isinstance(result, tuple):
                if len(result) >= 5:
                    next_state, reward, done, truncated, info = result
                else:
                    next_state, reward, done, info, _ = result
            else:
                next_state, reward, done, info, _ = result

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # Aktualizace počítadla kroků
            self.step_counter += 1
            self.log_message(f"Visualization progress: Step {self.step_counter} out of {self.total_steps}.")

            # Reset prostředí při ukončení epizody
            if done:
                self.log_message(f"Episode ended at step {self.step_counter}, resetting environment.")
                reset_result = self.env.reset()
                next_state = reset_result if not isinstance(reset_result, tuple) else reset_result[0]

            # Naplánování dalšího kroku vizualizace
            self.root.after(50, lambda: visualize_step(next_state))

        # Nastavení stavu aplikace a spuštění vizualizace
        self.is_visualizing = True
        self.train_btn.config(state="disabled")
        self.visualize_btn.config(state="disabled")
        visualize_step(state)

    def capture_animation(self):
        # Zachycení animace pro uložení do GIF souboru
        self.log_message("Capturing animation started.")
        frames = []
        result = self.env.reset()
        state = result if not isinstance(result, tuple) else result[0]
        self.step_counter = 0

        def capture_frame(curr_state):
            # Funkce pro zachycení jednoho snímku animace
            if self.step_counter >= self.total_steps:
                self.log_message(f"Captured {self.step_counter} frames out of {self.total_steps}. Saving animation...")
                self.save_animation(frames)
                self.log_message("Animation capture completed.")
                return

            # Vykreslení aktuálního stavu
            self.canvas.delete("all")
            self.draw_cartpole(self.canvas, curr_state)
            self.root.update()

            # Zachycení snímku plátna
            x = self.canvas.winfo_rootx()
            y = self.canvas.winfo_rooty()
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            frames.append(np.array(img))
            self.log_message(f"Frame {self.step_counter + 1} out of {self.total_steps} captured.")

            # Výběr akce a provedení kroku v prostředí
            action = self.agent.act(curr_state)
            result = self.env.step(action)

            # Zpracování výsledku kroku
            if isinstance(result, tuple):
                next_state, reward, done, _, _ = result[:5]
            else:
                next_state, reward, done, _, _ = result

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # Aktualizace počítadla kroků
            self.step_counter += 1

            # Reset prostředí při ukončení epizody
            if done:
                self.log_message(f"Episode ended at frame {self.step_counter}, resetting environment.")
                reset_result = self.env.reset()
                next_state = reset_result if not isinstance(reset_result, tuple) else reset_result[0]

            # Naplánování zachycení dalšího snímku
            self.root.after(50, lambda: capture_frame(next_state))

        # Spuštění zachytávání snímků
        capture_frame(state)

    def save_animation(self, frames):
        # Uložení zachycených snímků jako GIF animace
        try:
            total_frames = len(frames)
            self.log_message(f"Saving animation with {total_frames} frames out of {self.total_steps} steps...")

            # Zjištění počtu existujících animací v adresáři
            existing_files = [f for f in os.listdir(self.animation_dir) if
                              f.startswith("pole_balancing_") and f.endswith(".gif")]
            file_count = len(existing_files)

            # Vytvoření názvu souboru s pořadovým číslem
            file_path = os.path.join(self.animation_dir, f"pole_balancing_{file_count + 1}.gif")

            # Vytvoření GIF souboru
            with imageio.get_writer(file_path, mode='I', duration=0.1) as writer:
                for i, frame in enumerate(frames, start=1):
                    writer.append_data(frame)
                    if i % 10 == 0 or i == total_frames:
                        self.log_message(f"Saved frame {i}/{total_frames}.")

            self.log_message(f"Animation saved as '{file_path}'.")
        except Exception as e:
            self.log_message(f"Failed to save animation: {str(e)}")

    def on_canvas_resize(self, event):
        # Funkce volaná při změně velikosti plátna grafu
        if hasattr(self, 'training_scores') and self.training_scores:
            # Překreslení grafu s novou velikostí
            self.plot_training_progress(self.training_scores)


if __name__ == "__main__":
    root = tk.Tk()
    app = PoleBalancingApp(root)
    root.mainloop()
