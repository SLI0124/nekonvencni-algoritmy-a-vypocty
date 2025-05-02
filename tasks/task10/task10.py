import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import os

np.random.seed(42)

# Parameters for the bifurcation diagram
n_a_values = 1000  # Number of parameter values to explore
a_values = np.linspace(2.5, 4.0, n_a_values)  # Parameter range (focusing on chaotic region)
n_iterations = 100  # Number of iterations per parameter value
n_discard = 100  # Number of initial iterations to discard (transient)


def logistic_map(x, a):
    """Compute one iteration of the logistic map."""
    return a * x * (1 - x)


def generate_bifurcation_data(a_values, n_iterations, n_discard):
    """Generate data for a bifurcation diagram of the logistic map."""
    x = np.zeros((len(a_values), n_iterations))

    start_time = time.time()
    print("Generating bifurcation data...")

    init_x = 0.5  # Initial condition
    for i, a in enumerate(a_values):
        x_val = init_x

        # Discard transient iterations
        for _ in range(n_discard):
            x_val = logistic_map(x_val, a)

        # Run the logistic map iterations and store results
        for j in range(n_iterations):
            x_val = logistic_map(x_val, a)
            x[i, j] = x_val

        # Progress indicator
        if (i + 1) % (len(a_values) // 10) == 0:
            print(f"  Progress: {(i + 1) / len(a_values) * 100:.1f}%")

    print(f"Done! Time elapsed: {time.time() - start_time:.2f} seconds")
    return x


def plot_combined_diagram(a_values, actual_data, predicted_data=None, title="Bifurcation Diagram"):
    """Plot the actual bifurcation diagram and optionally compare with predicted data."""
    # Create figure with appropriate size and subplots
    if predicted_data is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(title, fontsize=20)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 8))
        ax1.set_title(title, fontsize=18)

    # Plot actual bifurcation diagram
    a_repeated_actual = np.repeat(a_values, actual_data.shape[1])
    x_flattened_actual = actual_data.flatten()
    ax1.scatter(a_repeated_actual, x_flattened_actual, s=0.05, c='black', alpha=0.7)
    ax1.set_xlabel("Parameter (a)", fontsize=14)
    ax1.set_ylabel("Stable values of x", fontsize=14)
    ax1.set_ylim(0, 1)

    # Set x-axis limits based on the input data range
    ax1.set_xlim(np.min(a_values), np.max(a_values))
    ax1.grid(True, alpha=0.3)

    # Add markers for critical bifurcation points that are within the current range
    critical_points = [3.0, 3.45, 3.57, 3.83]
    labels = ["Period 1→2", "Period 2→4", "Period 4→8", "Chaos Window"]

    for cp, label in zip(critical_points, labels):
        if np.min(a_values) <= cp <= np.max(a_values):
            ax1.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
            ax1.text(cp, 0.02, label, rotation=90, fontsize=10, color='red')

    if predicted_data is not None:
        # Plot predicted bifurcation diagram
        a_repeated_pred = np.repeat(a_values, predicted_data.shape[1])
        x_flattened_pred = predicted_data.flatten()
        ax2.scatter(a_repeated_pred, x_flattened_pred, s=0.05, c='blue', alpha=0.7)
        ax2.set_title("Predicted Bifurcation", fontsize=16)
        ax2.set_xlabel("Parameter (a)", fontsize=14)
        ax2.set_ylabel("Predicted values of x", fontsize=14)
        ax2.set_ylim(0, 1)

        # Match x-axis limits with the first plot
        ax2.set_xlim(np.min(a_values), np.max(a_values))
        ax2.grid(True, alpha=0.3)

        # Add same critical points that are within range
        for cp, label in zip(critical_points, labels):
            if np.min(a_values) <= cp <= np.max(a_values):
                ax2.axvline(x=cp, color='red', linestyle='--', alpha=0.5)
                ax2.text(cp, 0.02, label, rotation=90, fontsize=10, color='red')

    plt.tight_layout()
    return fig


def prepare_training_data(a_values, bifurcation_data):
    """Prepare data for neural network training - train on full trajectories."""
    X = a_values.reshape(-1, 1)
    y = bifurcation_data  # Each row is a full trajectory for a given 'a'
    return X, y


def train_neural_network(X, y):
    """Train a neural network to predict full trajectories of the logistic map."""
    # Scale the input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print("Training neural network on full trajectories...")
    start_time = time.time()

    # Create and train the MLP model
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 200, 100),
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

    # Evaluate the model
    y_pred_train = mlp.predict(X_train)
    y_pred_test = mlp.predict(X_test)

    # Calculate MSE for each trajectory
    train_mse = np.mean(np.mean((y_train - y_pred_train) ** 2, axis=1))
    test_mse = np.mean(np.mean((y_test - y_pred_test) ** 2, axis=1))

    print(f"Training trajectory MSE: {train_mse:.6f}")
    print(f"Test trajectory MSE: {test_mse:.6f}")
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    return mlp, scaler


def predict_bifurcation(mlp, scaler, a_values):
    """Use the trained neural network to predict full trajectories for bifurcation diagram."""
    # Scale the input a values
    a_scaled = scaler.transform(a_values.reshape(-1, 1))

    # Predict trajectories for each a value
    predicted_trajectories = mlp.predict(a_scaled)

    return predicted_trajectories


def analyze_prediction_quality(a_values, actual_data, predicted_data):
    """Analyze and visualize the quality of predictions."""
    # Compute error metrics for each parameter value
    errors = np.zeros(len(a_values))

    for i in range(len(a_values)):
        # Get the distributions of points for actual and predicted
        actual_vals = actual_data[i]
        pred_vals = predicted_data[i]

        # Sort to compare distributions
        actual_sorted = np.sort(actual_vals)
        pred_sorted = np.sort(pred_vals)

        # Compute absolute error for this parameter
        errors[i] = np.mean(np.abs(actual_sorted - pred_sorted))

    # Create visualization of the error
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot error as a function of parameter a
    ax.plot(a_values, errors, 'r-', linewidth=1.5)
    ax.set_title("Prediction Error vs. Parameter Value", fontsize=16)
    ax.set_xlabel("Parameter (a)", fontsize=14)
    ax.set_ylabel("Mean Absolute Error", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Identify regions of high error
    high_error = errors > 0.1
    if np.any(high_error):
        ax.fill_between(a_values, 0, errors, where=high_error,
                        color='red', alpha=0.3, label='High Error Region')
        ax.legend(fontsize=12)

    # Add vertical lines at critical bifurcation points
    critical_points = [3.0, 3.45, 3.57, 3.83]
    labels = ["Period 1→2", "Period 2→4", "Period 4→8", "Chaos Window"]

    for cp, label in zip(critical_points, labels):
        ax.axvline(x=cp, color='blue', linestyle='--', alpha=0.5)
        ax.text(cp, 0.02, label, rotation=90, fontsize=10, color='blue')

    plt.tight_layout()
    return fig, errors


def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 1. Generate bifurcation data
    bifurcation_data = generate_bifurcation_data(a_values, n_iterations, n_discard)

    # 2. Plot the actual bifurcation diagram
    actual_fig = plot_combined_diagram(a_values, bifurcation_data,
                                       title="Actual Bifurcation Diagram of Logistic Map")
    actual_fig.savefig(os.path.join(results_dir, "actual_bifurcation.png"), dpi=300, bbox_inches='tight')
    plt.close(actual_fig)

    # 3. Prepare data for neural network
    X, y = prepare_training_data(a_values, bifurcation_data)

    # 4. Train neural network
    mlp, scaler = train_neural_network(X, y)

    # 5. Generate predicted bifurcation diagram
    predicted_bifurcation = predict_bifurcation(mlp, scaler, a_values)

    # 6. Compare actual vs predicted in one plot
    comparison_fig = plot_combined_diagram(a_values, bifurcation_data, predicted_bifurcation,
                                           title="Actual vs. Predicted Bifurcation Diagram")
    comparison_fig.savefig(os.path.join(results_dir, "comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(comparison_fig)

    # 7. Analyze prediction quality
    error_fig, errors = analyze_prediction_quality(a_values, bifurcation_data, predicted_bifurcation)
    error_fig.savefig(os.path.join(results_dir, "prediction_error.png"), dpi=300, bbox_inches='tight')
    plt.close(error_fig)

    # 8. Generate focused comparison for most interesting region (a = 3.5 to 4.0)
    interesting_region = (a_values >= 3.5) & (a_values <= 4.0)
    a_subset = a_values[interesting_region]
    actual_subset = bifurcation_data[interesting_region]
    predicted_subset = predicted_bifurcation[interesting_region]

    # Make sure we're passing proper sized arrays
    print(
        f"Subset shape check - a_subset: {a_subset.shape}, actual: {actual_subset.shape}, predicted: {predicted_subset.shape}")

    detail_fig = plot_combined_diagram(a_subset, actual_subset, predicted_subset,
                                       title="Detailed Comparison: Chaotic Region (3.5 - 4.0)")
    detail_fig.savefig(os.path.join(results_dir, "detailed_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(detail_fig)

    # Display a message summarizing the number of samples in each region
    print(f"Total number of a values: {len(a_values)}")
    print(f"Number of a values in chaotic region: {len(a_subset)}")
    print(f"Number of points in chaotic bifurcation: {len(a_subset) * n_iterations}")

    print("Process completed. Results saved in the 'results' directory.")


if __name__ == "__main__":
    main()
