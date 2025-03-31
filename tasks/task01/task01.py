import matplotlib.pyplot as plt
import random
import os


def line(x):
    """ Funkce reprezentující přímku y = 3x + 2. """
    return 3 * x + 2


def categorize_point(x, y):
    """ Kategorizuje bod podle toho, zda je nad nebo pod přímkou y = 3x + 2. """
    if y > line(x):
        return 1
    else:  # y <= line(x), rozhodnul jsem se klassifikovat body na přímce jako pod přímkou
        return -1


def generate_points(num_points=100, x_range=(-10, 10), y_range=(-10, 10)) -> list[list[float]]:
    """
    Vytvoříme náhodná data, kde každý bod má 2 souřadnice a label.
    :return: x-ová souřadnice, y-ová souřadnice, label (1 nebo -1)
    """
    points = []
    for _ in range(num_points):
        x = random.uniform(x_range[0], x_range[1])
        y = random.uniform(y_range[0], y_range[1])
        label = categorize_point(x, y)
        points.append([x, y, label])
    return points


class Perceptron:
    """ Perceptron pro binární klasifikaci. """

    def __init__(self, learning_rate=0.01):
        self.weights = []
        self.bias = []
        self.learning_rate = learning_rate
        self._randomize_weights_and_bias()  # inicializace vah a biasu pomocí náhodných hodnot

    def _randomize_weights_and_bias(self):
        """ Inicializuje váhy a bias náhodnými hodnotami. """
        self.weights = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.bias = random.uniform(-1, 1)

    def fit(self, data, n_iters=100):
        """ Trénování perceptronu na vstupních datech. """
        for _ in range(n_iters):
            for point in data:
                self._update_weights_and_bias(point)

    def _update_weights_and_bias(self, point):
        """ Pomocná funkce pro aktualizaci vah a biasu. """
        x, y, label = point[0], point[1], point[2]

        # spočítáme predikci
        prediction = self.weights[0] * x + self.weights[1] * y + self.bias
        if prediction >= 0:
            y_pred = 1
        else:
            y_pred = -1

        # upravíme váhy a bias podle chyby, pokud je predikce dobrá, nic se neděje, závorka bude nula a nic se nepřičte
        update = self.learning_rate * (label - y_pred)
        self.weights[0] += update * x
        self.weights[1] += update * y
        self.bias += update  # bias je také váha, pro kterou platí stejná pravidla jako pro váhy

    def predict(self, X):
        """ Predikce na vstupních datech. """
        predictions = []
        for point in X:
            predictions.append(self._predict_point(point))
        return predictions

    def _predict_point(self, point):
        """Pomocná funkce pro predikci jednoho bodu."""
        x, y = point[0], point[1]
        prediction = self.weights[0] * x + self.weights[1] * y + self.bias
        if prediction >= 0:
            return 1
        else:
            return -1


def plot_results(X, y, perceptron, predictions):
    plt.figure(figsize=(10, 6))

    x_values = list(range(-10, 11))
    y_values = [line(x) for x in x_values]
    plt.plot(x_values, y_values, label='y = 3x + 2', color='black')

    above_x, above_y, below_x, below_y = [], [], [], []
    for i in range(len(X)):
        if y[i] == 1:
            above_x.append(X[i][0])
            above_y.append(X[i][1])
        else:
            below_x.append(X[i][0])
            below_y.append(X[i][1])

    plt.scatter(above_x, above_y, color='blue', label='Nad přímkou')
    plt.scatter(below_x, below_y, color='red', label='Pod přímkou')

    # Vykreslení rozhodovací hranice, kterou perceptron naučil
    # https://thomascountz.com/2018/04/13/calculate-decision-boundary-of-perceptron
    boundary_x, boundary_y = [], []
    for x in x_values:
        boundary_x.append(x)
        angle_tilt = -perceptron.weights[0] / perceptron.weights[1]  # úhel který vytváří přímka s osou x
        y_offset = -perceptron.bias / perceptron.weights[1]  # posunutí od počátku na ose y
        boundary_y.append(angle_tilt * x + y_offset)  # y = ax + b

    plt.plot(boundary_x, boundary_y, label='Rozhodovací hranice perceptronu', linestyle='--', color='green')

    for i in range(len(X)):
        color = 'blue' if predictions[i] == 1 else 'red'
        plt.scatter(X[i][0], X[i][1], color=color)

    save_path = "results"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_count = len(os.listdir(save_path))
    file_name = f"plot_{file_count}.png"

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Klasifikace bodů pomocí Perceptronu')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, file_name))
    plt.show()


def main():
    data = generate_points()
    X = []  # x-ová a y-ová souřadnice
    y = []  # labely
    for point in data:
        X.append([point[0], point[1]])
        y.append(point[2])

    perceptron = Perceptron()
    perceptron.fit(data, n_iters=1000)

    predictions = perceptron.predict(X)
    plot_results(X, y, perceptron, predictions)


if __name__ == "__main__":
    main()
