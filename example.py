from keras.datasets import mnist

from conv import Conv2d
from maxpool import MaxPooling2d
from softmax import Softmax
from cnn import training_loop, test

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images[:1000]
train_labels = train_labels[:1000]

test_images = test_images[:1000]
test_labels = test_labels[:1000]

conv2d = Conv2d(8, 3)
maxpool2d = MaxPooling2d(2)
softmax = Softmax(13*13*8, 10)


training_loop(3, train_images, train_labels, conv2d, maxpool2d, softmax)
test(test_images, test_labels, conv2d, maxpool2d, softmax)