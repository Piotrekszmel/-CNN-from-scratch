# CNN-from-scratch
CNN implementation from scratch using numpy for grayscale images.

An example of using the convolutional neural network is provided in example.py

conv.py and maxpool.py contain:
    - iterate_regions -> iterate through pixels
    - forward -> forward propagation 
    - backprop -> backprogation

softmax.py contain:
  - forward -> forward propagation of softmax 
  - backprop -> backprogation of softmax

cnn.py :
  - forward -> forward propagation of cnn
  - train -> completes a full training step on the given image and label
  - training_loop -> completes train function N times
  - test -> perform forward step and calculate loss and accuracy
 

