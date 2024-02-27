import numpy as np

class ReLUActivation:
    def forward(self, layer_outputs: np.ndarray) -> np.ndarray:
        self.output = np.maximum(0, layer_outputs) # if value is below zero make it a zero
    
class SoftMaxActivation:
    def forward(self, layer_outputs: np.ndarray):
        # outputs - max in that row to remove one of the 2 values (becomes 1), exponentiate other value
        exp_layer_outputs = np.exp(layer_outputs - np.max(layer_outputs, axis=1, keepdims=True)) 
        # take each expo'd value and divide it by the sum of the row for weight in row (probability)
        probabilities = exp_layer_outputs / np.sum(exp_layer_outputs, axis=1, keepdims=True)
        self.output: np.ndarray = probabilities