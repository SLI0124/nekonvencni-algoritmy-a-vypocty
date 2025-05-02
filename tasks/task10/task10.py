import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import os

np.random.seed(42)

# Parametry pro bifurkační diagram
n_a_values = 1000  # Počet hodnot parametru
a_values = np.linspace(2.5, 4.0, n_a_values)  # Rozsah parametru zaměřený na chaotickou oblast
n_iterations = 100  # Počet iterací pro každou hodnotu parametru
n_discard = 100  # Počet počátečních iterací k zahození (přechodový jev)


def logistic_map(x, a):
    """Logistické zobrazení - základní iterativní funkce"""
    return a * x * (1 - x)


def generate_bifurcation_data(a_values, n_iterations, n_discard):
    """Generování dat pro bifurkační diagram logistického zobrazení"""
    x = np.zeros((len(a_values), n_iterations))

    start_time = time.time()
    print("Generuji bifurkační data...")

    init_x = 0.5  # Počáteční podmínka
    for i, a in enumerate(a_values):
        x_val = init_x

        # Zahození přechodových iterací
        for _ in range(n_discard):
            x_val = logistic_map(x_val, a)

        # Sběr stabilních hodnot pro bifurkační diagram
        for j in range(n_iterations):
            x_val = logistic_map(x_val, a)
            x[i, j] = x_val

        # Indikátor průběhu
        if (i + 1) % (len(a_values) // 10) == 0:
            print(f"  Průběh: {(i + 1) / len(a_values) * 100:.1f}%")

    print(f"Hotovo! Čas výpočtu: {time.time() - start_time:.2f} sekund")
    return x


def plot_combined_diagram(a_values, actual_data, predicted_data=None, title="Bifurkační diagram"):
    """Vykreslení bifurkačního diagramu a případné porovnání se simulovanými daty"""
    # Vytvoření obrázku s odpovídající velikostí a rozložením
    if predicted_data is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(title, fontsize=20)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.set_title(title, fontsize=18)

    # Vykreslení skutečného bifurkačního diagramu
    a_repeated_actual = np.repeat(a_values, actual_data.shape[1])
    x_flattened_actual = actual_data.flatten()
    ax1.scatter(a_repeated_actual, x_flattened_actual, s=0.05, c='black', alpha=0.7)
    ax1.set_xlabel("Parametr (a)", fontsize=14)
    ax1.set_ylabel("Stabilní hodnoty x", fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.set_xlim(np.min(a_values), np.max(a_values))
    ax1.grid(True, alpha=0.3)

    # Značky pro kritické body bifurkace
    critical_points = [3.0, 3.45, 3.57, 3.83]
    labels = ["Perioda 1→2", "Perioda 2→4", "Perioda 4→8", "Chaotické okno"]

    for cp, label in zip(critical_points, labels):
        if np.min(a_values) <= cp <= np.max(a_values):
            ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
            ax1.text(cp, 0.02, label, rotation=90, fontsize=10, color='red')

    if predicted_data is not None:
        # Vykreslení predikovaného bifurkačního diagramu
        a_repeated_pred = np.repeat(a_values, predicted_data.shape[1])
        x_flattened_pred = predicted_data.flatten()
        ax2.scatter(a_repeated_pred, x_flattened_pred, s=0.05, c='blue', alpha=0.7)
        ax2.set_title("Predikovaná bifurkace", fontsize=16)
        ax2.set_xlabel("Parametr (a)", fontsize=14)
        ax2.set_ylabel("Predikované hodnoty x", fontsize=14)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(np.min(a_values), np.max(a_values))
        ax2.grid(True, alpha=0.3)

        # Stejné kritické body
        for cp, label in zip(critical_points, labels):
            if np.min(a_values) <= cp <= np.max(a_values):
                ax2.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
                ax2.text(cp, 0.02, label, rotation=90, fontsize=10, color='red')

    plt.tight_layout()
    return fig


def prepare_training_data(a_values, bifurcation_data):
    """Příprava dat pro trénink neuronové sítě - trénujeme na celých trajektoriích"""
    X = a_values.reshape(-1, 1)
    y = bifurcation_data  # Každý řádek je celá trajektorie pro danou hodnotu 'a'
    return X, y


def train_neural_network(X, y):
    """Trénink neuronové sítě pro predikci trajektorií logistického zobrazení"""
    # Škálování vstupních hodnot
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Rozdělení dat na trénovací a testovací množinu
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print("Trénuji neuronovou síť na celých trajektoriích...")
    start_time = time.time()

    # Vytvoření a trénink modelu MLP
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 200, 100),  # Komplexní architektura pro zachycení dynamiky
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=32,
        learning_rate='adaptive',
        max_iter=500,
        early_stopping=True,
        verbose=True,
        random_state=42
    )

    mlp.fit(X_train, y_train)

    # Vyhodnocení modelu
    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)

    # Výpočet MSE pro každou trajektorii
    train_mse = np.mean(np.mean((y_train - y_pred_train) ** 2, axis=1))
    test_mse = np.mean(np.mean((y_test - y_pred_test) ** 2, axis=1))

    print(f"Trénovací MSE trajektorií: {train_mse:.6f}")
    print(f"Testovací MSE trajektorií: {test_mse:.6f}")
    print(f"Doba tréninku: {time.time() - start_time:.2f} sekund")

    return mlp, scaler


def predict_bifurcation(mlp, scaler, a_values):
    """Použití natrénované neuronové sítě pro predikci trajektorií bifurkačního diagramu"""
    a_scaled = scaler.transform(a_values.reshape(-1, 1))
    predicted_trajectories = mlp.predict(a_scaled)
    return predicted_trajectories


def analyze_prediction_quality(a_values, actual_data, predicted_data):
    """Analýza a vizualizace kvality predikcí"""
    # Výpočet chybových metrik pro každou hodnotu parametru
    errors = np.zeros(len(a_values))

    for i in range(len(a_values)):
        # Získání distribucí bodů pro skutečná a predikovaná data
        actual_vals = actual_data[i]
        pred_vals = predicted_data[i]

        # Seřazení pro porovnání distribucí
        actual_sorted = np.sort(actual_vals)
        pred_sorted = np.sort(pred_vals)

        # Výpočet průměrné absolutní chyby pro tento parametr
        errors[i] = np.mean(np.abs(actual_sorted - pred_sorted))

    # Vizualizace chyby
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(a_values, errors, 'r-', linewidth=1.5)
    ax.set_title("Chyba predikce v závislosti na hodnotě parametru", fontsize=16)
    ax.set_xlabel("Parametr (a)", fontsize=14)
    ax.set_ylabel("Průměrná absolutní chyba", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Identifikace oblastí s vysokou chybou
    high_error = errors > 0.1
    if np.any(high_error):
        ax.fill_between(a_values, 0, errors, where=high_error,
                        color='red', alpha=0.3, label='Oblast vysoké chyby')
        ax.legend(fontsize=12)

    # Značky pro kritické body bifurkace
    critical_points = [3.0, 3.45, 3.57, 3.83]
    labels = ["Perioda 1→2", "Perioda 2→4", "Perioda 4→8", "Chaotické okno"]

    for cp, label in zip(critical_points, labels):
        ax.axvline(x=cp, color='blue', linestyle='--', alpha=0.5)
        ax.text(cp, 0.02, label, rotation=90, fontsize=10, color='blue')

    plt.tight_layout()
    return fig, errors


def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 1. Generování bifurkačních dat
    bifurcation_data = generate_bifurcation_data(a_values, n_iterations, n_discard)

    # 2. Vykreslení skutečného bifurkačního diagramu
    actual_fig = plot_combined_diagram(a_values, bifurcation_data,
                                       title="Skutečný bifurkační diagram logistického zobrazení")
    actual_fig.savefig(os.path.join(results_dir, "actual_bifurcation.png"), dpi=300, bbox_inches='tight')
    plt.close(actual_fig)

    # 3. Příprava dat pro neuronovou síť
    X, y = prepare_training_data(a_values, bifurcation_data)

    # 4. Trénink neuronové sítě
    mlp, scaler = train_neural_network(X, y)

    # 5. Generování predikovaného bifurkačního diagramu
    predicted_bifurcation = predict_bifurcation(mlp, scaler, a_values)

    # 6. Porovnání skutečného a predikovaného diagramu
    comparison_fig = plot_combined_diagram(a_values, bifurcation_data, predicted_bifurcation,
                                           title="Porovnání skutečného a predikovaného bifurkačního diagramu")
    comparison_fig.savefig(os.path.join(results_dir, "comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(comparison_fig)

    # 7. Analýza kvality predikce
    error_fig, errors = analyze_prediction_quality(a_values, bifurcation_data, predicted_bifurcation)
    error_fig.savefig(os.path.join(results_dir, "prediction_error.png"), dpi=300, bbox_inches='tight')
    plt.close(error_fig)

    # 8. Detailní porovnání pro chaotickou oblast (a = 3.5 až 4.0)
    interesting_region = (a_values >= 3.5) & (a_values <= 4.0)
    a_subset = a_values[interesting_region]
    actual_subset = bifurcation_data[interesting_region]
    predicted_subset = predicted_bifurcation[interesting_region]

    detail_fig = plot_combined_diagram(a_subset, actual_subset, predicted_subset,
                                       title="Detailní porovnání: Chaotická oblast (3.5 - 4.0)")
    detail_fig.savefig(os.path.join(results_dir, "detailed_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(detail_fig)

    print(f"Celkový počet hodnot parametru a: {len(a_values)}")
    print(f"Počet hodnot parametru a v chaotické oblasti: {len(a_subset)}")
    print(f"Počet bodů v chaotické bifurkaci: {len(a_subset) * n_iterations}")
    print("Proces dokončen. Výsledky uloženy v adresáři 'results'.")


if __name__ == "__main__":
    main()
