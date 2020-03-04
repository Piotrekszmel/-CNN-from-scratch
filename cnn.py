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

    return output, loss, acc


def train(input, label, lr=0.005):
    """
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    """
    # Forward
    output, loss, accuracy = forward(input, label)

    gradient = np.zeros(10)
    gradient[label] = -1 / output[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = maxpool2d.backprop(gradient)
    gradient = conv2d.backprop(gradient, lr)

    return loss, accuracy


def training_loop(num):
    for epoch in range(num):
        print("---- Epoch {} ----".format(epoch + 1))
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

        loss = 0
        correct = 0
        for i, (img, label) in enumerate(zip(train_images, train_labels)):
            if i % 100 == 99:
                print("[{}] Average Loss: {} | Accuracy: {}%"
                .format(i + 1, loss / i + 1, correct))
            
            l, acc = train(img, label)
            loss += l
            correct += acc