import numpy as np

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int): # n_inputs is # of cols of inputs
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons) # defining shape for matrix
        self.biases = np.zeros((1, n_neurons)) # defining shape for matrix
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
    def forward(self, inputs: np.ndarray) -> None:
        self.output = np.dot(inputs.astype(np.float32), self.weights + self.biases)
    def nudge_w_and_b(self, factor: float) -> None:
        self.weights = self.weights + factor * (np.random.randint(-1, 1, self.weights.shape[1]) * np.random.randn(self.weights.shape[1]))
        self.biases = self.biases +  0.8 * (factor) * (np.random.randint(-1, 1, self.biases.shape[1]) * np.random.randn(self.biases.shape[1]))