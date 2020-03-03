import numpy as np 


class Conv2d:
    """
    Conv2d takes 2d numpy array as an input. 
    """
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9
    
    def iterate_regions(self, input):
        """
        Generates all possible input regions using valid padding. 
        - input is a 2d numpy array.
        """
        h, w = input.shape
        
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                input_region = input[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield input_region, i, j
    
    def forward(self, input):
        """
        Performs forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        """
        self.last_input = input
        
        h, w = input.shape
        feature_map = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))
        
        for input_region, i, j in self.iterate_regions(input):
            feature_map = np.sum(input_region * self.num_filters, axis=(1, 2))
        
        return feature_map
    