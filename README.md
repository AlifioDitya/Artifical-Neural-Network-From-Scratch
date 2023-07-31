# Artificial Neural Network: From Scratch!
This repository features a TensorFlow-style, Artificial Neural Network made from scratch using only NumPy. It is made as a personal project to further my understanding on the internal workings of neural networks. The custom library can be found on the `tensorfio` package inside `src`, featuring:
- Activation functions
- Sequential model
- Fully connected layer


For a more in-depth explanation of ANN, head over to the `docs` folder and read the `Questions and Answers.pdf` file.

## Usage
For demonstration, I provided `ANN.ipynb` that shows the usage of the model on the Breast Cancer Dataset.
You can also test around with any numerical tabular data in the provided `ANN_Sandbox.ipynb` notebook.

## What is an Artificial Neural Network?
Artificial Neural Network (ANN) is a predictive model that is inspired by the biological neural network. It is a collection of connected nodes called neurons. Each neuron has a set of weights and biases that are adjusted during the training process. The weights and biases are used to calculate the output of the neuron. The output of the neuron is then passed to the next layer of neurons. The process is repeated until the output layer is reached. The output layer is the final layer of the network and it is where the prediction is made.

## How does it work?
Any kind of Neural Network generally works in two main phases: the forward pass and the backward pass. The forward pass is where the input is fed into the network and the output is calculated. The backward pass is where the weights and biases are adjusted based on the error of the output. The process is repeated until the error is minimized. The error is calculated using a loss function. The loss function is a function that calculates the error of the output. The loss function is usually a function that is differentiable. The loss function is used to calculate the gradient of the error. The gradient is then used to adjust the weights and biases. For a Multi-Layer Perceptron (MLP), it algorithmically works as follows:
1. Initialize the weights and biases of the network with random values.
2. Feed the input into the network (Forward Pass).
3. Calculate the error of the output using a loss function.
4. Calculate the gradient of the error.
5. Adjust the weights and biases using the gradient (Backward Pass).
6. Repeat steps 2-5 until the error is minimized for a given number of epochs, or until the performance converges.

## Requirements
- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
