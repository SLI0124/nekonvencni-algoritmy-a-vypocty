import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import io
import imageio

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_  # Patch to support gym's check for np.bool8


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = np.random.choice(len(self.memory), batch_size, replace=False)
        states = []
        targets = []

        for idx in minibatch:
            state, action, reward, next_state, done = self.memory[idx]
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)

            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state)
            target_f[action] = target

            states.append(state)
            targets.append(target_f)

        states = torch.stack(states)
        targets = torch.stack(targets)

        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PoleBalancingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pole-Balancing Problem with Q-Learning")

        # Initialize environment and agent
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = DQNAgent(self.state_size, self.action_size)

        # Training parameters
        self.episodes = 2000
        self.batch_size = 32
        self.is_training = False
        self.is_visualizing = False
        self.max_steps = 500

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        # Control frame
        control_frame = ttk.LabelFrame(self.root, text="Controls", padding=(10, 5))
        control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Training controls
        ttk.Label(control_frame, text="Training Episodes:").grid(row=0, column=0, sticky="w")
        self.episodes_entry = ttk.Entry(control_frame)
        self.episodes_entry.insert(0, str(self.episodes))  # changed default from "500" to self.episodes value
        self.episodes_entry.grid(row=0, column=1, sticky="ew")

        ttk.Label(control_frame, text="Batch Size:").grid(row=1, column=0, sticky="w")
        self.batch_entry = ttk.Entry(control_frame)
        self.batch_entry.insert(0, "32")
        self.batch_entry.grid(row=1, column=1, sticky="ew")

        self.train_btn = ttk.Button(control_frame, text="Start Training", command=self.start_training)
        self.train_btn.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

        self.stop_train_btn = ttk.Button(control_frame, text="Stop Training", state="disabled",
                                         command=self.stop_training)
        self.stop_train_btn.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")

        self.visualize_btn = ttk.Button(control_frame, text="Visualize Solution", command=self.visualize_solution)
        self.visualize_btn.grid(row=4, column=0, columnspan=2, pady=5, sticky="ew")

        # Status frame
        status_frame = ttk.LabelFrame(self.root, text="Status", padding=(10, 5))
        status_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.status_text = tk.Text(status_frame, height=10, width=50, state="disabled")
        self.status_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(status_frame, orient="vertical", command=self.status_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.status_text.configure(yscrollcommand=scrollbar.set)

        # Visualization frame
        vis_frame = ttk.LabelFrame(self.root, text="Visualization", padding=(10, 5))
        vis_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")

        self.canvas = tk.Canvas(vis_frame, width=400, height=300, bg="white")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        control_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)
        vis_frame.columnconfigure(0, weight=1)
        vis_frame.rowconfigure(0, weight=1)

    def log_message(self, message):
        self.status_text.configure(state="normal")
        self.status_text.insert("end", message + "\n")
        self.status_text.configure(state="disabled")
        self.status_text.see("end")
        self.root.update()

    def start_training(self):
        if self.is_training:
            return

        try:
            self.episodes = int(self.episodes_entry.get())
            self.batch_size = int(self.batch_entry.get())
        except ValueError:
            self.log_message("Invalid input for episodes or batch size")
            return

        self.is_training = True
        self.train_btn.config(state="disabled")
        self.stop_train_btn.config(state="normal")
        self.visualize_btn.config(state="disabled")

        # Reset agent for new training
        self.agent = DQNAgent(self.state_size, self.action_size)

        # Start training in a separate thread to keep GUI responsive
        self.root.after(100, self.run_training)

    def stop_training(self):
        self.is_training = False
        self.train_btn.config(state="normal")
        self.stop_train_btn.config(state="disabled")
        self.visualize_btn.config(state="normal")

    def run_training(self):
        scores = []

        for e in range(self.episodes):
            if not self.is_training:
                break

            state = self.env.reset()
            state = state[0]  # Extract the state array from the tuple
            total_reward = 0
            self.step_counter = 0  # inicializace počtu kroků

            while self.step_counter < self.max_steps:
                action = self.agent.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward

                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                self.step_counter += 1

                if done:
                    break

            scores.append(total_reward)
            self.agent.replay(self.batch_size)

            self.log_message(
                f"Episode: {e + 1}/{self.episodes}, Score: {total_reward}, Epsilon: {self.agent.epsilon:.2f}")

            if e % 10 == 0:
                self.plot_training_progress(scores)

            if e % 5 == 0:
                self.root.update()

        self.plot_training_progress(scores)
        self.stop_training()
        self.log_message("Training completed!")

    def plot_training_progress(self, scores):
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        ax.plot(scores)
        ax.set_title('Training Progress')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.grid(True)

        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img = ImageTk.PhotoImage(img)

        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=img)
        self.canvas.image = img  # Keep reference

        plt.close(fig)

    def visualize_solution(self):
        if self.is_visualizing:
            return

        self.is_visualizing = True
        self.train_btn.config(state="disabled")
        self.visualize_btn.config(state="disabled")

        # Use the existing canvas in the main window instead of creating a new one
        self.run_visualization(self.root, self.canvas)

    def draw_cartpole(self, canvas, state):
        # Draw cart and pole based on state: [cart_position, cart_velocity, pole_angle, angular_velocity]
        cart_pos = state[0]
        pole_angle = state[2]
        # Canvas settings
        canvas_width = 400
        canvas_height = 300
        cart_width = 50
        cart_height = 30
        pole_length = 100
        # Map cart position to canvas coordinate (scale factor: 50 pixels per unit)
        x_center = canvas_width / 2 + cart_pos * 50
        y_cart = canvas_height - 50  # fixed vertical position for cart
        # Draw cart (rectangle)
        cart_left = x_center - cart_width / 2
        cart_right = x_center + cart_width / 2
        cart_top = y_cart - cart_height / 2
        cart_bottom = y_cart + cart_height / 2
        canvas.create_rectangle(cart_left, cart_top, cart_right, cart_bottom, fill="blue")
        # Draw pole (line) starting from the top center of the cart
        pole_start_x = x_center
        pole_start_y = cart_top
        pole_end_x = pole_start_x + pole_length * np.sin(pole_angle)
        pole_end_y = pole_start_y - pole_length * np.cos(pole_angle)
        canvas.create_line(pole_start_x, pole_start_y, pole_end_x, pole_end_y, width=5, fill="red")

    def run_visualization(self, window, canvas):
        state = self.env.reset()
        state = state[0]  # Extract state
        self.step_counter = 0  # inicializace počtu kroků

        def update_visualization(curr_state):
            if not self.is_visualizing or self.step_counter >= self.max_steps:
                self.is_visualizing = False
                self.train_btn.config(state="normal")
                self.visualize_btn.config(state="normal")
                return

            canvas.delete("all")
            self.draw_cartpole(canvas, curr_state)
            window.update()

            action = self.agent.act(curr_state)
            next_state, _, done, _, _ = self.env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            self.step_counter += 1

            if done:
                self.is_visualizing = False
                self.train_btn.config(state="normal")
                self.visualize_btn.config(state="normal")
                return

            self.root.after(50, lambda: update_visualization(next_state))

        update_visualization(state)

    def save_animation(self, frames):
        try:
            with imageio.get_writer('pole_balancing.gif', mode='I', duration=0.1) as writer:
                for frame in frames:
                    writer.append_data(frame)
            self.log_message("Animation saved as 'pole_balancing.gif'")
        except Exception as e:
            self.log_message(f"Failed to save animation: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PoleBalancingApp(root)
    root.mainloop()
