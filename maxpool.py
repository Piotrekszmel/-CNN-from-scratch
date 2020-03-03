import numpy as np 


class MaxPooling2d:
    """
    Max pooling layer using a 2d pool of the given size
    Height and width are equal
    """
    def __init__(self, pool_size):
        """
        - pool_size: int
        """
        self.pool_size = pool_size
    
    def iterate_regions(self, input):
        """
        Generates non-overlapping (pool_size, pool_size) inputs regions to pool over
        - input: 2d numpy array 
        """
        
        h, w, _ = input.shape
        
        new_h = (h - self.pool_size) // 2
        new_w = (w - self.pool_size) // 2
        
        for i in range(new_h):
            for j in range(new_w):
                input_region = input[(i * self.pool_size):(i * self.pool_size + self.pool_size), 
                                  (j * self.pool_size):(j * self.pool_size + self.pool_size)]
                yield input_region, i, j
    
    def forward(self, input):
        """
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        """
        self.last_input = input
        
        h, w, num_filters = input.shape
        output = np.zeros(((h - self.pool_size) // 2, (w - self.pool_size) // 2, num_filters))
        
        for input_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(input_region, axis=(0, 1))
            
        return output
    
    def backprop(self, d_L_d_out):
        """
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        """
        
        
        

