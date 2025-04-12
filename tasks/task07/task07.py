import random
import matplotlib.pyplot as plt


def define_transformations():
    transformations1 = [
        {"a": 0.00, "b": 0.00, "c": 0.01, "d": 0.00, "e": 0.26, "f": 0.00, "g": 0.00, "h": 0.00, "i": 0.05, "j": 0.00,
         "k": 0.00, "l": 0.00},
        {"a": 0.20, "b": -0.26, "c": -0.01, "d": 0.23, "e": 0.22, "f": -0.07, "g": 0.07, "h": 0.00, "i": 0.24,
         "j": 0.00, "k": 0.80, "l": 0.00},
        {"a": -0.25, "b": 0.28, "c": 0.01, "d": 0.26, "e": 0.24, "f": -0.07, "g": 0.07, "h": 0.00, "i": 0.24, "j": 0.00,
         "k": 0.22, "l": 0.00},
        {"a": 0.85, "b": 0.04, "c": -0.01, "d": -0.04, "e": 0.85, "f": 0.09, "g": 0.00, "h": 0.08, "i": 0.84, "j": 0.00,
         "k": 0.80, "l": 0.00}
    ]

    transformations2 = [
        {"a": 0.05, "b": 0.00, "c": 0.00, "d": 0.00, "e": 0.60, "f": 0.00, "g": 0.00, "h": 0.00, "i": 0.05, "j": 0.00,
         "k": 0.00, "l": 0.00},
        {"a": 0.45, "b": -0.22, "c": 0.22, "d": 0.22, "e": 0.45, "f": 0.22, "g": -0.22, "h": 0.22, "i": -0.45,
         "j": 0.00, "k": 1.00, "l": 0.00},
        {"a": -0.45, "b": 0.22, "c": -0.22, "d": 0.22, "e": 0.45, "f": 0.22, "g": 0.22, "h": -0.22, "i": 0.45,
         "j": 0.00, "k": 1.25, "l": 0.00},
        {"a": 0.49, "b": -0.08, "c": 0.08, "d": 0.08, "e": 0.49, "f": 0.08, "g": 0.08, "h": -0.08, "i": 0.49, "j": 0.00,
         "k": 2.00, "l": 0.00}
    ]

    return [transformations1, transformations2]


def generate_points(transformations, iterations=10000):
    """Generate points using the given transformations."""
    histories = []

    for transformation in transformations:
        x, y, z = 0, 0, 0
        history = []

        for _ in range(iterations):
            random_row = random.randint(0, 3)
            t = transformation[random_row]  # získáme slovník transformace

            # Aplikace transformace s použitím názvů parametrů
            x_new = t["a"] * x + t["b"] * y + t["c"] * z + t["j"]
            y_new = t["d"] * x + t["e"] * y + t["f"] * z + t["k"]
            z_new = t["g"] * x + t["h"] * y + t["i"] * z + t["l"]

            x, y, z = x_new, y_new, z_new
            history.append((x, y, z))

        histories.append(history)
    return histories


def plot_model(history, title, color='b'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_vals, y_vals, z_vals = zip(*history)
    ax.scatter(x_vals, y_vals, z_vals, c=color, marker='.')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def main():
    transformations = define_transformations()

    histories = generate_points(transformations)

    plot_model(histories[0], 'First Model', 'b')
    plot_model(histories[1], 'Second Model', 'r')

    plt.show()


if __name__ == "__main__":
    main()
