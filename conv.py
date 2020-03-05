import numpy as np 


class Conv2d:
    """
    Conv2d takes 2d numpy array as an input. Use it only as first layer, because 
    Conv2d is not returning the loss gradient 
    """
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9
    
    def iterate_regions(self, input):
        """
        Generates all possible input regions using valid padding
        - input: 2d numpy array
        """
        h, w = input.shape
        
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                input_region = input[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield input_region, i, j
    
    def forward(self, input):
        """
        Performs forward pass of the conv layer using the given input
        Returns a 3d numpy array with dimensions (h, w, num_filters)
        - input: 2d numpy array
        """
        self.last_input = input
        
        h, w = input.shape
        feature_map = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))
        
        for input_region, i, j in self.iterate_regions(input):
            feature_map[i, j] = np.sum(input_region * self.filters, axis=(1, 2))
        
        return feature_map
    
    def backprop(self, d_L_d_out, lr):
        """
        Performs a backward pass of the conv layer
        - d_L_d_out is the loss gradient for this layer's outputs
        - lr: float
        """
        d_L_d_filters = np.zeros(self.filters.shape)
        
        for input_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * input_region
        
        #Update filters
        self.filters -= lr * d_L_d_filters