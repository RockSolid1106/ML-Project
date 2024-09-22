import numpy as np
from mnist import MNIST
import cmath
from scipy.special import softmax
from matplotlib import pyplot as plt


def sigmoid(x):
    return 1/(1+cmath.exp(-x))
apply_sigmoid = np.vectorize(sigmoid)


mndata = MNIST("/Users/rock/Documents/ML/handwriting/machine_learning/dataset")
images, labels = mndata.load_testing()
t=int(input("Enter random number: "))

first_image = mndata.process_images_to_numpy(images[t])
first_image = first_image/255
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
print("displaying image now")
plt.show()

w1 = np.load("/Users/rock/Documents/ML/handwriting/machine_learning/w1_matrix.dat", allow_pickle=True)
w2 = np.load("/Users/rock/Documents/ML/handwriting/machine_learning/w2_matrix.dat", allow_pickle=True)
b1 = np.load("/Users/rock/Documents/ML/handwriting/machine_learning/b1_matrix.dat", allow_pickle=True)
b2 = np.load("/Users/rock/Documents/ML/handwriting/machine_learning/b2_matrix.dat", allow_pickle=True)

input = mndata.process_images_to_numpy(images[t])
input = input / 255

l1_neurons = np.dot(w1, input)
l1_neurons = np.c_[l1_neurons]
l1_neurons = l1_neurons + b1
l1_neurons = apply_sigmoid(l1_neurons)
l2_neurons = np.dot(w2, l1_neurons)
l2_neurons = l2_neurons + b2
l2_neurons = softmax(l2_neurons)
#print(l2_neurons.real)

max = l2_neurons[0][0]
loc = 0
for i in range(1,10):
    if max < l2_neurons[i][0]:
        loc = i
        max = l2_neurons[i][0]
        
print("This image is a "+str(loc))
print("Confidence: "+str(100*l2_neurons[loc][0].real)+"%")

