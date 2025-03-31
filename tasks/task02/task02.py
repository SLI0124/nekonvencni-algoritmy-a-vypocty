import numpy as np
import matplotlib.pyplot as plt
import os


def sigmoid(x):
    """ Sigmoidní aktivační funkce, která mapuje vstup na hodnotu mezi 0 a 1. """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """ Derivace sigmoidní funkce, použitá pro zpětnou propagaci. """
    return x * (1 - x)


class NeuralNetwork:
    """ Třída reprezentující jednoduchou neuronovou síť s jednou skrytou vrstvou. """

    def __init__(self, input_size, hidden_size, output_size):
        # Velikosti vrstev
        self.input_size = input_size  # Počet vstupních neuronů
        self.hidden_size = hidden_size  # Počet neuronů ve skryté vrstvě
        self.output_size = output_size  # Počet výstupních neuronů

        # Inicializace vah a biasů náhodnými hodnotami
        self.weights_input_hidden = np.random.uniform(0.0, 1.0, size=(input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(0.0, 1.0, size=(hidden_size, output_size))
        self.bias_hidden = np.random.uniform(0.0, 1.0, size=(1, hidden_size))
        self.bias_output = np.random.uniform(0.0, 1.0, size=(1, output_size))

        # Proměnné pro ukládání mezivýsledků
        self.hidden_layer_output = None  # Výstup skryté vrstvy
        self.output = None  # Výstup sítě

    def forward_propagation(self, input_data):
        """ Provedení forward propagace: výpočet výstupu sítě na základě vstupních dat. """
        # Výpočet skryté vrstvy: kombinace vstupů, vah a biasu, následovaná aktivací
        hidden_layer_activation = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(hidden_layer_activation)

        # Výpočet výstupní vrstvy: kombinace výstupu skryté vrstvy, vah a biasu, následovaná aktivací
        output_layer_activation = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(output_layer_activation)

        return self.output

    def backward_propagation(self, input_data, target_output, learning_rate):
        """ Provedení zpětné propagace: aktualizace vah a biasů na základě chyby. """
        # Chyba výstupní vrstvy: rozdíl mezi očekávaným a skutečným výstupem
        output_error = target_output - self.output

        # Gradient výstupní vrstvy: chyba vynásobená derivací sigmoidní funkce
        output_gradient = output_error * sigmoid_derivative(self.output)

        # Chyba skryté vrstvy: zpětná propagace chyby pomocí vah mezi skrytou a výstupní vrstvou
        hidden_layer_error = np.dot(output_gradient, self.weights_hidden_output.T)

        # Gradient skryté vrstvy: chyba skryté vrstvy vynásobená derivací sigmoidní funkce
        hidden_layer_gradient = hidden_layer_error * sigmoid_derivative(self.hidden_layer_output)

        # Aktualizace vah a biasů pomocí gradientů a rychlosti učení
        self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_gradient) * learning_rate
        self.weights_input_hidden += np.dot(input_data.T, hidden_layer_gradient) * learning_rate
        self.bias_output += np.sum(output_gradient) * learning_rate
        self.bias_hidden += np.sum(hidden_layer_gradient) * learning_rate

    def train(self, input_data, target_output, learning_rate, epochs):
        """ Trénování sítě: opakovaná forward a backward propagace. """
        for epoch in range(epochs):
            self.forward_propagation(input_data)
            self.backward_propagation(input_data, target_output, learning_rate)

            # Výpis ztráty každých 1000 epoch pro sledování průběhu učení
            if epoch % 1_000 == 0:
                loss = np.mean(np.square(target_output - self.output))
                print(f"Epoch {epoch}, Loss: {loss:.5f}")

    def predict(self, input_data):
        """ Předpověď výstupu sítě pro daná vstupní data. """
        return self.forward_propagation(input_data)

    def print_weights_biases(self, phase):
        """ Výpis vah a biasů pro danou fázi (např. před nebo po učení). """
        print("\n" + "-" * 50)
        print(f"Váhy a biasy {phase}:")
        print(f"neuron_hidden1.weights {self.weights_input_hidden[0]}")
        print(f"neuron_hidden2.weights {self.weights_input_hidden[1]}")
        print(f"neuron_output.weights {self.weights_hidden_output.flatten()}")
        print(f"neuron_hidden1.bias {self.bias_hidden[0, 0]}")
        print(f"neuron_hidden2.bias {self.bias_hidden[0, 1]}")
        print(f"neuron_output.bias {self.bias_output[0, 0]}")
        print("-" * 50)


def plot_decision_boundary(network):
    """ Vykreslení rozhodovací hranice naučené neuronovou sítí. """
    # Vytvoření mřížky bodů přes vstupní prostor
    x_values = np.linspace(0, 1, 100)
    y_values = np.linspace(0, 1, 100)
    xx, yy = np.meshgrid(x_values, y_values)

    # Ruční vytvoření mřížky bodů pro předpověď
    grid = []
    for x in x_values:
        for y in y_values:
            grid.append([x, y])
    grid = np.array(grid)  # Tvar (10000, 2)

    # Předpověď výstupu pro každý bod v mřížce
    predictions = network.predict(grid)
    predictions = predictions.reshape(xx.shape)  # Přetvarování na (100, 100) pro vykreslení

    # Vykreslení rozhodovací hranice
    plt.figure(figsize=(8, 6))
    plt.contour(xx, yy, predictions, levels=[0.5], colors="black", linewidths=2)
    plt.scatter([0, 1], [0, 1], c="red", edgecolors="black", label="Třída 0", s=100)
    plt.scatter([0, 1], [1, 0], c="blue", edgecolors="black", label="Třída 1", s=100)
    plt.title("Rozhodovací hranice naučená neuronovou sítí")
    plt.xlabel("Vstup 1")
    plt.ylabel("Vstup 2")
    plt.legend()

    # Uložení obrázku do složky results
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = os.path.join(save_dir, "decision_boundary.png")
    plt.savefig(save_name)

    plt.show()


# Hlavní funkce
def main():
    # XOR tabulka jako vstupní data
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target_output = np.array([[0], [1], [1], [0]])

    # Inicializace neuronové sítě
    input_size = 2
    hidden_size = 2
    output_size = 1
    network = NeuralNetwork(input_size, hidden_size, output_size)

    # Výpis počátečních vah a biasů
    network.print_weights_biases("před fází učení")

    # Trénování sítě
    print("\nProbíhá učení..\n")
    learning_rate = 0.1
    epochs = 10_000
    network.train(input_data, target_output, learning_rate, epochs)

    # Výpis konečných vah a biasů
    network.print_weights_biases("po fázi učení")

    # Testování sítě
    print("\nProbíhá testování..\n")
    correct = 0
    total = len(input_data)
    for i in range(total):
        prediction = network.predict(input_data[i].reshape(1, -1))
        expected = target_output[i][0]
        is_correct = (round(prediction[0, 0]) == expected)
        correct += is_correct

        print(f"Hádaný výstup {prediction[0, 0]:.12f}   Očekávaný výstup {expected}   Je to správně? {is_correct}")

    # Výpočet a výpis přesnosti
    accuracy = (correct / total) * 100
    print(f"\nÚspěšnost je {accuracy:.1f} %")
    print("\n" + "-" * 50)

    # Vykreslení rozhodovací hranice
    plot_decision_boundary(network)


if __name__ == "__main__":
    main()
