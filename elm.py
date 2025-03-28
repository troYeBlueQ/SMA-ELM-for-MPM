import numpy as np
from paper.model.integration.models.sma import SMA
from sklearn.metrics import roc_curve, auc, accuracy_score

class ELM:
    def __init__(self, input_dim, hidden_neurons):
        self.input_dim = input_dim
        self.hidden_neurons = hidden_neurons

    def initialize_random_weights(self):
        self.input_weights = np.random.uniform(-1, 1, (self.input_dim, self.hidden_neurons))
        self.biases = np.random.uniform(-1, 1, self.hidden_neurons)

    def initialize(self, input_weights, biases):
        self.input_weights = input_weights
        self.biases = biases

    def train(self, X, y):
        H = self._activation(X)
        self.output_weights = np.linalg.pinv(H).dot(y)

    def predict_proba(self, X):
        H = self._activation(X)
        raw_output = H.dot(self.output_weights)
        # 应用 sigmoid 激活函数将输出值映射到 [0, 1]
        return 1 / (1 + np.exp(-raw_output))

    def _activation(self, X):
        Z = X.dot(self.input_weights) + self.biases
        Z = np.clip(Z, -50, 50)
        return 1 / (1 + np.exp(-Z))

def initialize_sma_elm(X_train, X_test, y_train, y_test, population_size, max_iterations):
    def fitness_function(params):
        input_dim, hidden_neurons = X_train.shape[1], 50
        input_weights = params[:input_dim * hidden_neurons].reshape(input_dim, hidden_neurons)
        biases = params[input_dim * hidden_neurons:]
        elm = ELM(input_dim, hidden_neurons)
        elm.initialize(input_weights, biases)
        elm.train(X_train, y_train)
        proba = elm.predict_proba(X_train)
        return np.mean((y_train - proba) ** 2)

    bounds = [(-1, 1)] * (X_train.shape[1] * 50 + 50)
    best_params, _, _, vb_vc_history = SMA(fitness_function, bounds, population_size, max_iterations)
    input_weights = best_params[:X_train.shape[1] * 50].reshape(X_train.shape[1], 50)
    biases = best_params[X_train.shape[1] * 50:]
    sma_elm = ELM(X_train.shape[1], 50)
    sma_elm.initialize(input_weights, biases)
    sma_elm.train(X_train, y_train)
    proba = sma_elm.predict_proba(X_test)
    auc_value = auc(*roc_curve(y_test, proba)[:2])
    acc = accuracy_score(y_test, (proba >= 0.5).astype(int))
    return proba, auc_value, acc, vb_vc_history
