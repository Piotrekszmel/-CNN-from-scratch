from keras.datasets import mnist
import numpy as np 

from conv import Conv2d
from maxpool import MaxPooling2d
from softmax import Softmax

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images[:1000]
train_labels = train_labels[:1000]

test_images = test_images[:1000]
test_labels = test_labels[:1000]

conv2d = Conv2d(8, 3)
maxpool2d = MaxPooling2d(2)
softmax = Softmax(13*13*8, 10)


def forward(input, label):
    """
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - input: 2d numpy array
    - label: digit (int)
    """
    output = conv2d.forward((input / 255) - 0.5)
    output = maxpool2d.forward(output)
    output = softmax.forward(output)

    loss = -np.log(output[label])
    acc = 1 if np.argmax(output) == label else 0

    return out, loss, acc