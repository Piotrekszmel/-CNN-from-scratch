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
        Generates non-overlapping (2, 2) inputs regions to pool over
        - input: 2d numpy array 
        """
        
        h, w, _ = input.shape
        
        new_h = h // 2
        new_w = w // 2
        
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
        output = np.zeros((h // 2, w // 2, num_filters))
        
        for input_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(input_region, axis=(0, 1))
            
        return output
    
    def backprop(self, d_L_d_out):
        """
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        """
        d_L_d_input = np.zeros(self.last_input.shape)
        
        for input_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = input_region.shape
            amax = np.amax(input_region, axis=(0, 1))
            
            for i2 in range(h):
              for j2 in range(w):
                  for f2 in range(f):
                    # If this pixel was the max value, copy the gradient to it.
                    if input_region[i2, j2, f2] == amax[f2]:
                        d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]
        
        return d_L_d_input

