import sys
if "/kaggle/input" not in sys.path:
    sys.path.append('/kaggle/input')
import matplotlib as mpl
import numpy as np
from mnist import MNIST
import cmath
import os
from scipy.special import softmax
import requests

mndata = MNIST("/kaggle/input/mydataset")
images, labels = mndata.load_training()

w1= np.array([])
b1 = np.array([])
l1_neurons = np.array([])

w2 = np.array([])
b2 = np.array([])

l2_neurons = []

learning_rate = 0.008

epoch = 20

def sigmoid(x):
    #print("sigmoid: "+str(x))
    return 1/(1+cmath.exp(-x))
apply_sigmoid = np.vectorize(sigmoid)

def sigmoid_derivative(x):
    return cmath.exp(-x)/(1+cmath.exp(-x))**2
    
def init_params():
    global w1, b1, w2, b2
    w1 = np.random.rand(15, 784)*np.sqrt(1/784)
    b1 = np.random.rand(15, 1)*np.sqrt(1/784)
    w2 = np.random.rand(10, 15)*np.sqrt(1/15)
    b2 = np.random.rand(10, 1)*np.sqrt(1/15)
    #w1 = np.load("w1_matrix.dat", allow_pickle=True)
    #w2 = np.load("w2_matrix.dat", allow_pickle=True)
    #b1 = np.load("b1_matrix.dat", allow_pickle=True)
    #b2 = np.load("b2_matrix.dat", allow_pickle=True)
init_params()

def forward_prop(img_index):
    input_array = mndata.process_images_to_numpy(images[img_index]).T
    input_array = input_array/255
    
    l1_neurons = np.dot(w1, input_array)[:, None]
    l1_neurons = l1_neurons + b1
    l1_neurons = apply_sigmoid(l1_neurons)

    l2_neurons = np.dot(w2, l1_neurons)
    l2_neurons = l2_neurons + b2
    #avg_sum = 0
    #for i in range(10):
    #    avg_sum += l2_neurons[i][0]
    #for i in range(10):
    #    l2_neurons[i][0] = l2_neurons[i][0]/avg_sum
    l2_neurons = softmax(l2_neurons)
    return l1_neurons, l2_neurons
    

def back_prop(a, b):
    global input_array, w1, b1, w2, b2, learning_rate
    
    for q in range(epoch): #epoch
        correct = 0
        print("run: "+str(q + 1))
        for p in range(a,b): #a,b
            input_array = mndata.process_images_to_numpy(images[p])
            input_array = input_array/255
            input_array = input_array[:, None]
            expected_neurons = np.array([0,0,0,0,0,0,0,0,0,0])
            expected_neurons[labels[p]] = 1
            expected_neurons = np.c_[expected_neurons]
            l1_neurons, l2_neurons = forward_prop(p)
            max = 0
            loc = 0
            for i in range(10):
                if max < l2_neurons[i][0]:
                    loc = i
                    max = l2_neurons[i][0]
            if loc == (labels[p]):
                correct += 1
            expected_neurons = np.c_[expected_neurons]
            
            difference_matrix = expected_neurons - l2_neurons
    
            cost = 0
            for i in range(10):
                cost += (difference_matrix[i])


            del_w2 = learning_rate * np.dot(difference_matrix, l1_neurons.T)
            del_b2 = learning_rate * difference_matrix
            
            mat1 = np.dot(w2.T, difference_matrix)
            mat2 = []
            for i in range(15):
                mat2.append(sigmoid_derivative(l1_neurons[i][0]))
            mat2 = np.c_[mat2] #make list to column vector
            
            for i in range(15):
                mat1[i][0] = mat1[i][0] * mat2[i][0]
            delH = mat1
            
            del_w1 = learning_rate * np.dot(delH, input_array.T)
            
            del_b1 = learning_rate * delH
            
            w1 += del_w1.real
            w2 += del_w2.real
            b1 += del_b1.real
            b2 += del_b2.real
        print("accuracy: "+str(correct/3))
        w1.dump("w1_matrix.dat")
        w2.dump("w2_matrix.dat")
        b1.dump("b1_matrix.dat")
        b2.dump("b2_matrix.dat")
    print("Back propogation complete")

for x in range(200):
    s = 0
    t = 300
    back_prop(s, t)
    print("-----------------------")
    s+=300
    t+=300
    
