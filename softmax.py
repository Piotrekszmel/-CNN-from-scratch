import numpy as np 


class Softmax:
    """
    Fully-connected layer with softmax activation
    """
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)
        
    def forward(self, input):
        """
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input: array with any dimensions.
        """
        self.last_input_shape = input.shape
        
        input = input.flatten()
        self.last_input = input
        
        input_len, nodes = self.weights.shape
        
        scores = np.dot(input, self.weights) + self.biases
        self.last_totals = scores
        
        exp = np.exp(scores)
        return exp / np.sum(exp, axis=0)