import numpy as np


class SimpleNeuralNetwork:
    def __init__(self):
        # Инициализация весов случайными значениями
        self.weights = np.random.randn(3)
        self.bias = np.random.randn()


    def sigmoid(self, x):
        # Функция активации - сигмоида
        return 1 / (1 + np.exp(-x))
    

    def sigmoid_derivative(self, x):
        # Производная сигмоиды
        return x * (1 - x)
    
    
    def predict(self, inputs):
        # Прямой проход - вычисление выхода
        network_input = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(network_input)
    

    def train(self, training_inputs, training_outputs, epochs=1000, learning_rate=0.1):
        # Обучение с помощью градиентного спуска
        for _ in range(epochs):
            # Прямой проход
            outputs = self.predict(training_inputs)

            # Вычисление ошибки
            error = training_outputs - outputs

            # Градиентный спуск
            adjustments = error * self.sigmoid_derivative(outputs)
            self.weights += learning_rate * np.dot(training_inputs.T, adjustments)
            self.bias += learning_rate * np.sum(adjustments)


if __name__ == "__main__":
    nn = SimpleNeuralNetwork()

    # Обучающие данные, три входа, четре примера
    training_inputs = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1], 
    ])

    # Ожидаемые выходные значения
    training_outputs = np.array([0, 1, 1, 0])

    # Обучаем сеть
    nn.train(training_inputs, training_outputs, epochs=5000, learning_rate=0.1)

    # Тестируем сеть
    test_input = np.array([1, 0, 0])
    print("Предсказание для [1, 0, 0]", nn.predict(test_input))
