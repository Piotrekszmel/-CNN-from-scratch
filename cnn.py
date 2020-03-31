from keras.datasets import mnist
import numpy as np 


def forward(input, label, conv, maxpool, softmax):
    """
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - input: 2d numpy array
    - label: digit (int)
    """
    output = conv.forward((input / 255) - 0.5)
    output = maxpool.forward(output)
    output = softmax.forward(output)

    loss = -np.log(output[label])
    acc = 1 if np.argmax(output) == label else 0

    return output, loss, acc


def train(input, label, conv, maxpool, softmax, lr=0.005):
    """
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    """
    # Forward
    output, loss, accuracy = forward(input, label, conv, maxpool, softmax)

    gradient = np.zeros(10)
    gradient[label] = -1 / output[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = maxpool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, accuracy


def training_loop(num, train_images, train_labels, conv, maxpool, softmax, lr=0.005):
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
                
                correct = 0
                
            
            l, acc = train(img, label, conv, maxpool, softmax, lr)
            loss += l
            correct += acc


def test(test_images, test_labels, conv, maxpool, softmax):
    loss = 0
    correct = 0
    for img, label in zip(test_images, test_labels):
        _, l, acc = forward(img, label, conv, maxpool, softmax)
        loss += l
        correct += acc
    
    print("Test Loss: ", loss / len(test_images))
    print("Test Accuracy: ", correct / len(test_images))
