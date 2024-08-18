# ML-Project
My attempt at making an ML model as a complete beginner. The main goal was to understand the math involved and the basic idea behind Neural Networks. I still lack some necessary mathematical knowledge to fully understand the equations involved, but at least I tried. God I hate Jacobians. 

## An ML model to predict what the digit is in a 28x28 input image. 
The model was trained and tested using the [MNIST dataset](https://github.com/cvdfoundation/mnist). This dataset is a collection of $60,000$ $28\times28$ images of handwritten digits. The Python library `mnist-python` offers functions to easily read the data.

The network consists of $784(28\times28)$ input neurons, $15$ hidden layer neurons and $10$($1$ through $10$ for the output) output neurons.
The input neurons have values from the pixels, which range from $0$ to $255$ as "brightness" are linearly scaled down to $0$ to $1$ by dividing by $255$. These are stored as `input-array`($A_{784 \times 1}$)
Each of the hidden layer neurons is associated with a bias and a weight for each of the hidden layer neurons. These weights are stored in the `w1`($W_{15 \times 784}^1$) matrix. Similarly, the biases are stored in `b1`($B_{15 \times 1}^1$) matrix and this is similar for the output layer neurons as `w2`($W_{10 \times 1}^2$) and `b2`($B_{10 \times 1}^2$). Each of these is assigned a random value from the standard deviation. This is necessary to ensure the network doesn't end up stuck in a weird minima. 

The network is trained of course by back-propogation. You basically run an image through the network, compare the outputs to what was expected and tell the computer how horrible of a job it's done. The computer then does some clever math, and tweaks the values of the weights and biases to get the output closer to what was expected.
More specifically, we use a cost function. The cost function is the sum of the squares of the differences of each element of the output neurons and the expected output. It is a measure of how "wrong" the output of the network is. This is the function we need to minimize. Does that ring some calculus bells in your mind? 
Lets say the input image was a 3, and the output neurons were(note that the sum of the elements are 1 because this layer gets altered to show the values as probabilities):

$$N^2 = \begin{bmatrix}
0.12 \\
0.08 \\
0.09 \\
0.07 \\
0.11 \\
0.10 \\
0.05 \\
0.14 \\
0.07 \\
0.09
\end{bmatrix}$$

And the expected neurons are:

$$E = \begin{bmatrix}
0 \\
0 \\
1 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0 \\
0
\end{bmatrix}$$

Then $\Delta O$ is the difference matrix of these two matrices. The cost function is the square of all the elements of this matrix.

Now we get to the real math part, and I genuinely don't understand much of it. I'll try to explain as much as I do.
We need to minimize the cost function($C$), which can only be done by tweaking the values of the weights and biases. To do this, we calculate the derivative of the cost function with respect to a specific weight.

$$
\frac{\partial C}{\partial w_i}
$$

Then the weight is nudged by this derivative, times a "learning rate". Learning rate is a measure of how big nudges you want to make to the variables at every back propogation. Set this too low and your network will take forever to find a minima. Set it too high and it might just keep jumping over minimas. It's like the steering sensitivity in a racing game. Set it too low and you can barely turn, set it too high and you're kissing the barriers.

$$
w_i = w_i - LR \times \frac{\partial C}{\partial w_i}
$$

Of course, we don't compute this individually for each weight and bias. It's done by differentiating the cost function with respect to the weight matrix to obtain another matrix, each element of which denotes the derivative of the cost function with respect to that weight. This is where Jacobians are involved and I do not understand the math here as a high school student. Finally, you just subtract this matrix from initial weight matrix and that way all the weights are nudged appropriately. Here are all the necessary equations(note: $\sigma(x)$ is the sigmoid function given by $\sigma(x) = \frac{1}{1+e^{-x}}$):

$$
\begin{gather}
\Delta O = N^2 - E \\\\
\Delta W^2 = LR \cdot (\Delta O)(N^1)^T\\\\
\Delta B^2 = LR \cdot (\Delta O)\\\\
\Delta h_{15 \times 1} = [(W^2)^T(\Delta O)] \cdot \sigma'(N^1)\\\\
\Delta W^1 = LR \cdot (\Delta h)(A)^T\\\\
\Delta B^1 = LR \cdot (\Delta h)\\\\
W^1 -= \Delta W^1\\\\
W^2 -= \Delta W^2\\\\
B^1 -= \Delta B^1\\\\
B^2 -= \Delta B^2
\end{gather}
$$

The sigmoid(or softmax) function is applied to the hidden layer to add a bit of "complexity" to the network. If this isn't done, the entire network can be proven to be a linear combination of the input layer. Adding the sigmoid function into the mix allows the network some more "flexibility".

Helpful resources:

https://youtu.be/9RN2Wr8xvro?si=IpbPKrK555KBOZMy

https://youtu.be/hfMk-kjRv4c?si=vEerYDXOxS1oVu5h

https://youtu.be/Ilg3gGewQ5U?si=Pd8OqjDYKc9UAj8q

https://youtu.be/IHZwWFHWa-w?si=xnwGofzONRjMg1w6
