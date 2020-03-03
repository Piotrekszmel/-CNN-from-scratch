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
        
        h, w = input.shape
        
        new_h = (h - self.pool_size) // 2
        new_w = (w - self.pool_size) // 2
        
        for i in range(new_h):
            for j in range(new_w):
                im_region = input[(i * self.pool_size):(i * self.pool_size + self.pool_size), 
                                  (j * self.pool_size):(j * self.pool_size + self.pool_size)]
                yield im_region, i, j
    

