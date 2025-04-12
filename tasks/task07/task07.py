import random
import matplotlib.pyplot as plt


def define_transformations():
    transformations1 = [
        [0.00, 0.00, 0.01, 0.00, 0.26, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
        [0.20, -0.26, -0.01, 0.23, 0.22, -0.07, 0.07, 0.00, 0.24, 0.00, 0.80, 0.00],
        [-0.25, 0.28, 0.01, 0.26, 0.24, -0.07, 0.07, 0.00, 0.24, 0.00, 0.22, 0.00],
        [0.85, 0.04, -0.01, -0.04, 0.85, 0.09, 0.00, 0.08, 0.84, 0.00, 0.80, 0.00]
    ]

    transformations2 = [
        [0.05, 0.00, 0.00, 0.00, 0.60, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00],
        [0.45, -0.22, 0.22, 0.22, 0.45, 0.22, -0.22, 0.22, -0.45, 0.00, 1.00, 0.00],
        [-0.45, 0.22, -0.22, 0.22, 0.45, 0.22, 0.22, -0.22, 0.45, 0.00, 1.25, 0.00],
        [0.49, -0.08, 0.08, 0.08, 0.49, 0.08, 0.08, -0.08, 0.49, 0.00, 2.00, 0.00]
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
            a, b, c, d, e, f, g, h, i, j, k, l = transformation[random_row]

            # Apply the transformation, update the coordinates and save the new point
            x_new = a * x + b * y + c * z + j
            y_new = d * x + e * y + f * z + k
            z_new = g * x + h * y + i * z + l

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
