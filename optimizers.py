
# Stochastic Gradient Descent
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layers):
        for layer in layers:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db
