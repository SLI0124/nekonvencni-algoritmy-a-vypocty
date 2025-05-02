import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import os

np.random.seed(42)

CRITICAL_POINTS = [3.0, 3.45, 3.57, 3.83]
CRITICAL_LABELS = ["Perioda 1→2", "Perioda 2→4", "Perioda 4→8", "Chaotické okno"]


def logistic_map(x, a):
    return a * x * (1 - x)


def generate_bifurcation_data(a_values, n_iterations=100, n_discard=100):
    """Generování dat pro bifurkační diagram"""
    x = np.zeros((len(a_values), n_iterations))
    print("Generuji bifurkační data...")

    for i, a in enumerate(a_values):
        x_val = 0.5

        # Zahození přechodových iterací
        for _ in range(n_discard):
            x_val = logistic_map(x_val, a)

        # Sběr stabilních hodnot
        for j in range(n_iterations):
            x_val = logistic_map(x_val, a)
            x[i, j] = x_val

        if (i + 1) % (len(a_values) // 20) == 0:
            print(f"  Průběh: {(i + 1) / len(a_values) * 100:.0f}%")

    return x


def add_critical_lines(ax, a_values):
    for cp, label in zip(CRITICAL_POINTS, CRITICAL_LABELS):
        if np.min(a_values) <= cp <= np.max(a_values):
            ax.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
            ax.text(cp, 0.02, label, rotation=90, fontsize=10, color='red')


def plot_combined_diagram(a_values, actual_data, predicted_data=None, title="Bifurkační diagram"):
    fig = plt.figure(figsize=(24, 12))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    fig.suptitle(title, fontsize=20)

    # Levý graf - skutečná data
    a_repeated = np.repeat(a_values, actual_data.shape[1])
    x_flattened = actual_data.flatten()
    ax1.scatter(a_repeated, x_flattened, s=0.05, c='black', alpha=0.7)
    ax1.set_xlabel("Parametr (a)", fontsize=14)
    ax1.set_ylabel("Hodnoty x", fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(np.min(a_values), np.max(a_values))
    ax1.grid(True, alpha=0.3)
    add_critical_lines(ax1, a_values)

    # Pravý graf - predikovaná data
    data_to_plot = predicted_data if predicted_data is not None else actual_data
    a_repeated = np.repeat(a_values, data_to_plot.shape[1])
    x_flattened = data_to_plot.flatten()
    ax2.scatter(a_repeated, x_flattened, s=0.05, c='blue', alpha=0.7)
    ax2.set_title("Predikovaná bifurkace", fontsize=16)
    ax2.set_xlabel("Parametr (a)", fontsize=14)
    ax2.set_ylabel("Hodnoty x", fontsize=14)
    ax2.set_ylim(0, 1)
    ax2.set_xlim(np.min(a_values), np.max(a_values))
    ax2.grid(True, alpha=0.3)
    add_critical_lines(ax2, a_values)

    plt.tight_layout()
    return fig


def train_neural_network(X, y):
    """Trénink neuronové sítě"""
    print("Trénuji neuronovou síť...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 200, 100),
        activation='relu',
        solver='adam',
        batch_size=32,
        learning_rate='adaptive',
        max_iter=1_000,
        early_stopping=False,
        n_iter_no_change=200,
        verbose=True,
        tol=1e-8,
        random_state=42
    )

    mlp.fit(X_train, y_train)
    return mlp, scaler


def analyze_prediction_quality(a_values, actual_data, predicted_data):
    """Analýza kvality predikce"""
    errors = np.zeros(len(a_values))

    for i in range(len(a_values)):
        actual_sorted = np.sort(actual_data[i])
        pred_sorted = np.sort(predicted_data[i])
        errors[i] = np.mean(np.abs(actual_sorted - pred_sorted))

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.plot(a_values, errors, 'r-', linewidth=1.5)
    ax.set_title("Chyba predikce v závislosti na hodnotě parametru", fontsize=18)
    ax.set_xlabel("Parametr (a)", fontsize=14)
    ax.set_ylabel("Průměrná absolutní chyba", fontsize=14)
    ax.grid(True, alpha=0.3)

    for cp, label in zip(CRITICAL_POINTS, CRITICAL_LABELS):
        ax.axvline(x=cp, color='blue', linestyle='--', alpha=0.5)
        ax.text(cp, 0.005, label, rotation=90, fontsize=10, color='blue')

    high_error_mask = errors > 0.1
    if np.any(high_error_mask):
        ax.fill_between(a_values, 0, errors, where=high_error_mask,
                        color='red', alpha=0.3, label='Oblast vysoké chyby')
        ax.legend(fontsize=12)

    ax.axhline(y=0.05, color='green', linestyle='--', alpha=0.7)
    ax.text(3.9, 0.055, "Práh přijatelné chyby (0.05)", fontsize=10, color='green')

    plt.tight_layout()
    return fig, errors


def save_figure(fig, filename, results_dir="results", dpi=300):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    fig.savefig(os.path.join(results_dir, filename), dpi=dpi)
    plt.close(fig)


def main():
    n_a_values = 1000
    a_values = np.linspace(2.5, 4.0, n_a_values)
    results_dir = "results"

    start_time = time.time()

    # Generování dat
    bifurcation_data = generate_bifurcation_data(a_values)

    # Vykreslení skutečného diagramu
    actual_fig = plot_combined_diagram(
        a_values,
        bifurcation_data,
        title="Skutečný bifurkační diagram logistického zobrazení"
    )
    save_figure(actual_fig, "actual_bifurcation.png", results_dir)

    # Trénink a predikce
    X = a_values.reshape(-1, 1)
    mlp, scaler = train_neural_network(X, bifurcation_data)
    predicted_bifurcation = mlp.predict(scaler.transform(X))

    # Porovnání diagramů
    comparison_fig = plot_combined_diagram(
        a_values,
        bifurcation_data,
        predicted_bifurcation,
        title="Porovnání skutečného a predikovaného bifurkačního diagramu"
    )
    save_figure(comparison_fig, "comparison.png", results_dir)

    # Analýza chyby
    error_fig, _ = analyze_prediction_quality(a_values, bifurcation_data, predicted_bifurcation)
    save_figure(error_fig, "error.png", results_dir)

    # Detail chaotické oblasti
    chaotic_mask = (a_values >= 3.5) & (a_values <= 4.0)
    detail_fig = plot_combined_diagram(
        a_values[chaotic_mask],
        bifurcation_data[chaotic_mask],
        predicted_bifurcation[chaotic_mask],
        title="Detailní porovnání: Chaotická oblast (3.5 - 4.0)"
    )
    save_figure(detail_fig, "chaotic_detail.png", results_dir)

    print(f"Dokončeno za {time.time() - start_time:.1f} sekund.")


if __name__ == "__main__":
    main()
