import numpy as np
import tensorflow as tf # For optimized tensor operations only

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
        self.activation = activation
        self.use_bias = use_bias
        self.input_shape = input_shape
    
    def build(self, input_shape):
        # Initialize weights and biases
        self.w = self.add_weight([input_shape[-1], self.units]) # Shape: (input_dim, output_dim)
        if self.use_bias:
            self.b = self.add_weight([self.units]) # Shape: (output_dim,)
        
        super().build(input_shape)
    
    def call(self, inputs):
        # Forward propagate inputs through this layer
        # y = x * w + b
        self.input = inputs
        y = tf.matmul(inputs, self.w)
        if self.use_bias:
            y = y + self.b
        self.z = y
        if self.activation is not None:
            y = self.activation(y)
        self.a = y

        return y