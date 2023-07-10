import numpy as np
from .Activation import sigmoid, relu, softmax, tanh

class Layer:
    '''
    Base class for layers in the network.

    Arguments:
        units: Number of neurons in the layer

    Attributes:
        units: Number of neurons in the layer
        built: Whether the layer has been built
        _build_input_shape: Shape of the input to the layer
    '''
    def __init__(self):
        self.built = False

    def build(self, input_shape):
        self._build_input_shape = input_shape
        self.built = True

    def call(self, inputs, *args, **kwargs):
        return inputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)
    
    def add_weight(self, shape):
        # Initializes to random normal distribution
        return np.random.normal(size=shape)

class Dense(Layer):
    '''
    A fully-connected layer.

    Methods:
        build(input_shape): Builds the layer by initializing weights and biases.
        call(inputs): Forward propagates inputs through this layer.

    Attributes:
        units: Number of neurons in this layer.
        activation: Activation function to use.
        use_bias: Whether to use a bias vector.
        input_shape: Shape of the input tensor.

        w: Weights of this layer.
        b: Biases of this layer.
        input: Input tensor.
        z: Weighted sum of inputs.
        a: Activation of weighted sum of inputs.
    '''
    def __init__(self, units, activation=None, use_bias=True, input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units

        # If activation is not a string, check if it is a function
        if type(activation) is not str:
            if callable(activation):
                self.activation = activation
            else:
                raise TypeError('activation must be a string or a function')

        if activation.lower() == 'sigmoid':
            self.activation = sigmoid
        elif activation.lower() == 'relu':
            self.activation = relu
        elif activation.lower() == 'softmax':
            self.activation = softmax
        elif activation.lower() == 'tanh':
            self.activation = tanh
        else:
            self.activation = None
        
        self.use_bias = use_bias
        self.input_shape = input_shape
    
    def build(self, input_shape):
        # Initialize weights and biases
        self.w = self.add_weight([input_shape[-1], self.units])
        if self.use_bias:
            self.b = self.add_weight([self.units])
        
        super().build(input_shape)
    
    def call(self, inputs):
        # Forward propagate inputs through this layer
        self.input = inputs
        y = np.matmul(inputs, self.w)
        if self.use_bias:
            y = y + self.b
        self.z = y
        if self.activation is not None:
            y = self.activation(y)
        self.a = y

        return y
    
class Conv2D(Layer):
    '''
    A 2D convolution layer.
    For simplicity, only supports the following:
    - Square kernels and square strides
    - Single colour channel
    - Weight tensor is treated as a linear layer with shape (filters, kernel_size[0] * kernel_size[1]) for easier gradient calculation
    - Same goes for bias vector, with shape (filters,)

    Methods:
        build(input_shape): Builds the layer by initializing weights and biases.
        call(inputs): Forward propagates inputs through this layer.

    Attributes:
        filters: Number of filters in this layer.
        kernel_size: Size of the convolution window.
        strides: Stride of the convolution window.
        activation: Activation function to use.
        use_bias: Whether to use a bias vector.
        input_shape: Shape of the input tensor.

        w: Weights of this layer.
        b: Biases of this layer.
        input: Input tensor.
        z: Weighted sum of inputs.
        a: Activation of weighted sum of inputs.
    '''
    def __init__(self, filters, kernel_size, strides=(1, 1), activation=None, use_bias=True, input_shape=None, flatten=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.flatten = flatten

        # If activation is not a string, check if it is a function
        if type(activation) is not str:
            if callable(activation):
                self.activation = activation
            else:
                raise TypeError('activation must be a string or a function')

        if activation.lower() == 'sigmoid':
            self.activation = sigmoid
        elif activation.lower() == 'relu':
            self.activation = relu
        elif activation.lower() == 'softmax':
            self.activation = softmax
        elif activation.lower() == 'tanh':
            self.activation = tanh
        else:
            self.activation = None
        
        self.use_bias = use_bias
        self.input_shape = input_shape
    
    def build(self, input_shape):
        # Only support single colour channel for now
        if input_shape[-1] != 1:
            raise NotImplementedError('Only single colour channel is supported.')

        # Initialize weights and biases
        # Weight tensor is treated as a linear layer with shape (kernel_size[0] * kernel_size[1], filters)
        self.w = self.add_weight([self.kernel_size[0] * self.kernel_size[1], self.filters])
        if self.use_bias:
            self.b = self.add_weight([self.filters])
        
        super().build(input_shape)
    
    def call(self, inputs): 
        # Forward propagate inputs through this layer
        self.input = inputs

        # Initialize output tensor, have shape (batch_size, output_height * output_width, filters)
        output_shape = (inputs.shape[0], inputs.shape[1] // self.strides[0] * inputs.shape[2] // self.strides[1], self.filters)
        y = np.zeros(output_shape)

        # Iterate through each filter
        for i in range(self.filters):
            # Iterate through each row of the input tensor
            for j in range(0, inputs.shape[1] - self.kernel_size[0] + 1, self.strides[0]):
                # Iterate through each column of the input tensor
                for k in range(0, inputs.shape[2] - self.kernel_size[1] + 1, self.strides[1]):
                    # Get the current window
                    window = inputs[:, j:j + self.kernel_size[0], k:k + self.kernel_size[1], :].reshape(inputs.shape[0], -1)
                    # Compute the weighted sum of the window
                    y[:, j // self.strides[0] * inputs.shape[2] // self.strides[1] + k // self.strides[1], i] = np.matmul(window, self.w[:, i])
                    if self.use_bias:
                        y[:, j // self.strides[0] * inputs.shape[2] // self.strides[1] + k // self.strides[1], i] = y[:, j // self.strides[0] * inputs.shape[2] // self.strides[1] + k // self.strides[1], i] + self.b[i]

                    # Apply activation function
                    if self.activation is not None:
                        y[:, j // self.strides[0] * inputs.shape[2] // self.strides[1] + k // self.strides[1], i] = self.activation(y[:, j // self.strides[0] * inputs.shape[2] // self.strides[1] + k // self.strides[1], i])

        if self.flatten:
            y = y.reshape(inputs.shape[0], -1)

        return y
    
    def gradient_descent(self, learning_rate, y):
        # Compute the gradient of the loss function with respect to the weighted sum of inputs
        if self.activation is None:
            dz = y
        elif self.activation == sigmoid:
            dz = y * (1 - y)
        elif self.activation == relu:
            dz = np.where(y > 0, 1, 0)
        elif self.activation == softmax:
            dz = y * (1 - y)
        elif self.activation == tanh:
            dz = 1 - y ** 2
        else:
            raise NotImplementedError('Only sigmoid, relu, softmax and tanh are supported.')

        # Compute the gradient of the loss function with respect to the weights
        dw = np.zeros(self.w.shape)
        for i in range(self.filters):
            for j in range(self.kernel_size[0] * self.kernel_size[1]):
                # Iterate through each row of the input tensor
                for k in range(0, self.input.shape[1] - self.kernel_size[0] + 1, self.strides[0]):
                    # Iterate through each column of the input tensor
                    for l in range(0, self.input.shape[2] - self.kernel_size[1] + 1, self.strides[1]):
                        # Get the current window
                        window = self.input[:, k:k + self.kernel_size[0], l:l + self.kernel_size[1], :].reshape(self.input.shape[0], -1)
                        # Compute the gradient of the loss function
                        dw[j, i] = dw[j, i] + np.sum(dz[:, k // self.strides[0] * self.input.shape[2] // self.strides[1] + l // self.strides[1], i] * window[:, j])

        # Compute the gradient of the loss function with respect to the biases
        if self.use_bias:
            db = np.zeros(self.b.shape)
            for i in range(self.filters):
                for j in range(0, self.input.shape[1] - self.kernel_size[0] + 1, self.strides[0]):
                    for k in range(0, self.input.shape[2] - self.kernel_size[1] + 1, self.strides[1]):
                        db[i] = db[i] + np.sum(dz[:, j // self.strides[0] * self.input.shape[2] // self.strides[1] + k // self.strides[1], i])

        # Compute the gradient of the loss function with respect to the inputs
        dx = np.zeros(self.input.shape)
        for i in range(self.filters):
            for j in range(self.kernel_size[0] * self.kernel_size[1]):
                # Iterate through each row of the input tensor
                for k in range(0, self.input.shape[1] - self.kernel_size[0] + 1, self.strides[0]):
                    # Iterate through each column of the input tensor
                    for l in range(0, self.input.shape[2] - self.kernel_size[1] + 1, self.strides[1]):
                        # Get the current window
                        window = self.input[:, k:k + self.kernel_size[0], l:l + self.kernel_size[1], :].reshape(self.input.shape[0], -1)
                        # Compute the gradient of the loss function
                        dx[:, k:k + self.kernel_size[0], l:l + self.kernel_size[1], :] = dx[:, k:k + self.kernel_size[0], l:l + self.kernel_size[1], :] + dz[:, k // self.strides[0] * self.input.shape[2] // self.strides[1] + l // self.strides[1], i].reshape(-1, 1) * self.w[j, i].reshape(1, -1)

        # Update the weights and biases
        self.w = self.w - learning_rate * dw

        if self.use_bias:
            self.b = self.b - learning_rate * db

        return dx
    
    def compute_output_shape(self, input_shape):
        # Input shape format: (batch_size, height, width)
        if self.flatten:
            return (input_shape[0], self.filters * (input_shape[1] - self.kernel_size[0] + 1) * (input_shape[2] - self.kernel_size[1] + 1))
        else:
            return (input_shape[0], (input_shape[1] - self.kernel_size[0] + 1) // self.strides[0], (input_shape[2] - self.kernel_size[1] + 1) // self.strides[1], self.filters)
    
class MaxPooling2D(Layer):
    '''
    A 2D max pooling layer.

    Methods:
        build(input_shape): Builds the layer by initializing weights and biases.
        call(inputs): Forward propagates inputs through this layer.

    Attributes:
        pool_size: Size of the pooling window.
        strides: Stride of the pooling window.
        input_shape: Shape of the input tensor.

        input: Input tensor.
        z: Weighted sum of inputs.
        a: Activation of weighted sum of inputs.
    '''
    def __init__(self, pool_size, strides=(1, 1), input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.input_shape = input_shape
    
    def build(self, input_shape):
        super().build(input_shape)
    
    def call(self, inputs):
        # Forward propagate inputs through this layer
        self.input = inputs
        output_height = int((inputs.shape[1] - self.pool_size[0]) / self.strides[0] + 1)
        output_width = int((inputs.shape[2] - self.pool_size[1]) / self.strides[1] + 1)
        y = np.zeros((inputs.shape[0], output_height, output_width, inputs.shape[3]))
        for i in range(inputs.shape[0]):
            for j in range(y.shape[1]):
                for k in range(y.shape[2]):
                    for l in range(y.shape[3]):
                        y[i, j, k, l] = np.max

        return y
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] - self.pool_size[0] + 1, input_shape[2] - self.pool_size[1] + 1, input_shape[3])