import numpy as np

class Sequential:
    '''
    A sequential model.

    Methods:
        add(layer): Adds a layer to the model.
        build(input_shape): Builds the model by initializing weights and biases.
        call(inputs): Forward propagates inputs through this model.
        summary(): Prints a summary of the model.
        compile(loss, optimizer): Compiles the model for training.
        compute_loss(y_true, y_prob): Computes the loss between the true labels and predictions.
        compute_metrics(y_true, y_pred): Computes the metrics for this model.
        forward_propagation(X, y): Forward propagates the inputs and computes the loss.
        backward_propagation(X, y, y_prob): Backward propagates the loss.
        gradient_descent(X, y, y_true, y_prob): Performs gradient descent.
        loss_gradient(y_true, y_prob): Computes the gradient of the loss function.
        update_weights(grad_w, grad_b): Updates the weights and biases.
        fit(X, y, epochs, batch_size): Trains the model.
        predict(X): Predicts the labels for the given data.
        evaluate(X, y): Evaluates the model on the given data.

    Attributes:
        layers: List of layers in this model.
        built: Whether the model is built or not.
        _build_input_shape: Shape of the input tensor.
        loss: Loss function to use.
        optimizer: Optimizer to use.
        metrics: Metrics to use.
        lr: Learning rate.
    '''
    def __init__(self, layers=np.array([]), name='Sequential', beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.built = False
        self.layers = layers
        self.name = name

        # Adam optimizer parameters
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # If input shape is known, build the model
        if len(self.layers) > 0 and layers[0].input_shape is not None:
            self.build(self.layers[0].input_shape)

    def add(self, layer):
        self.layers.append(layer)

        if self.built:
            self.build(self._build_input_shape)
    
    def build(self, input_shape):
        self._build_input_shape = input_shape
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)
        self.built = True
    
    def call(self, inputs):
        y = inputs
        for layer in self.layers:
            y = layer.call(y)
        return y
    
    def summary(self):
        if not self.built:
            raise ValueError('Model is not built yet')

        print(' Model Summary')
        for i in range(len(' Layer (type) | Output Shape | Param #')):
            print('-', end='')
        print()
        print(' Layer (type) | Output Shape | Param #')
        for i in range(len(' Layer (type) | Output Shape | Param #')):
            print('=', end='')
        print()
        total_params = 0
        for layer in self.layers:
            output_shape = layer.compute_output_shape(self._build_input_shape)
            param_count = 0
            for param in layer.__dict__:
                if param == 'w':
                    param_count += np.prod(layer.__dict__[param].shape)
                elif param == 'b':
                    param_count += layer.__dict__[param].shape[0]
            print(f' {layer.__class__.__name__}', end='')

            for i in range(len(' Layer (type) ') - len(layer.__class__.__name__) - 1):
                print(' ', end='')

            print(f'| {output_shape}', end='')
            
            for i in range(len(' Output Shape ') - len(str(output_shape)) - 1):
                print(' ', end='')
            
            print(f'| {param_count}', end='')

            for i in range(len(' Param # ') - len(str(param_count)) - 1):
                print(' ', end='')
            print()

            total_params += param_count
        print('======================================')
        print(f'Total params: {total_params}')

    def compile(self, optimizer, loss, metric):
        if optimizer not in ['sgd', 'adam']:
            raise ValueError('Optimizer not supported')
        if loss not in ['mse', 'crossentropy', 'binary_crossentropy', 'categorical_crossentropy']:
            raise ValueError('Loss not supported')
        if metric not in ['accuracy', 'mse']:
            raise ValueError('Metric not supported')

        self.optimizer = optimizer
        self.loss = loss
        self.metric = metric

    def compute_loss(self, y_true, y_prob):
        epsilon = 1e-10 # Error term to prevent division by zero
        if self.loss == 'mse':
            return np.mean((y_true - y_prob) ** 2)
        elif self.loss == 'crossentropy' or self.loss == 'binary_crossentropy':
            return np.mean(-y_true * np.log(y_prob + epsilon) - (1 - y_true) * np.log(1 - y_prob + epsilon))
        elif self.loss == 'categorical_crossentropy':
            return np.mean(-np.sum(y_true * np.log(y_prob + epsilon), axis=-1))
        else:
            raise ValueError('Loss not supported')

    def compute_metric(self, y_true, y_pred):
        if self.metric == 'accuracy':
            return np.mean(y_true == y_pred)
        elif self.metric == 'mse':
            return np.mean((y_true - y_pred) ** 2)
        else:
            raise ValueError('Metric not supported')

    def one_hot_encode(self, y):
        if len(y.shape) == 1:
            y_one_hot = np.array([[1 if y[i] == j else 0 for j in range(self.layers[-1].units)] for i in range(len(y))])
            return y_one_hot
        else:
            y_one_hot = np.zeros((len(y), self.layers[-1].units))
            y_one_hot[np.arange(len(y)), y] = 1
            return y_one_hot
    
    def forward_propagation(self, X, y):
        '''
        Algorithm to propagate the input forward through the network to compute the loss and metric.
        
        Parameters:
        X: Input data
        y: Labels
        '''

        X = np.array(X)
        y = np.array(y)
        
        # One-hot encode the labels
        y_true = self.one_hot_encode(y)
        
        # Forward propagation
        y_prob = self.call(X)

        if y_prob.shape[-1] == 1:
            y_pred = np.array([0 if y_prob[i] > 0.5 else 1 for i in range(len(y_prob))])
        else:
            y_pred = np.argmax(y_prob, axis=-1)

        # Compute loss
        loss = self.compute_loss(y_true, y_prob)

        # Compute metric
        metric = self.compute_metric(y, y_pred)

        return y_prob, y_pred, loss, metric
        
    def backward_propagation(self, X, y, y_prob):
        '''
        Algorithm to propagate the error backwards through the network, using the selected optimizer.
        
        Parameters:
        X: Input data
        y: Labels
        y_prob: Output of the last layer
        '''
        
        y = np.array(y)

        # One hot encode the labels
        y_true = self.one_hot_encode(y)
        
        if self.optimizer == 'sgd':
            # Compute gradient of loss with respect to weights and biases
            grad_w, grad_b = self.gradient_descent(X, y_true, y_prob)
        elif self.optimizer == 'adam':
            # Adam optimizer
            grad_w, grad_b = self.adam(X, y_true, y_prob)
        else:
            raise ValueError('Optimizer not supported')
        
        # Update weights and biases
        self.update_weights(grad_w, grad_b)
        
    def adam(self, X, y_true, y_prob):
        '''
        Adam optimizer
        
        Parameters:
        X: Input data
        y_true: One-hot encoded labels
        y_prob: Output of the last layer
        '''
        
        # Initialize the first and second moment estimates
        self.m_w = [np.zeros(layer.w.shape) for layer in self.layers]
        self.m_b = [np.zeros(layer.b.shape) for layer in self.layers]
        self.v_w = [np.zeros(layer.w.shape) for layer in self.layers]
        self.v_b = [np.zeros(layer.b.shape) for layer in self.layers]

        # Compute gradient of loss with respect to weights and biases
        grad_w, grad_b = self.gradient_descent(X, y_true, y_prob)

        # Update the iteration counter
        self.t += 1

        # Update the first and second moment estimates
        for i, layer in enumerate(self.layers):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * grad_w[i] ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_b[i] ** 2

        # Correct the bias of the first and second moment estimates
        m_w_corrected = [self.m_w[i] / (1 - self.beta1 ** self.t) for i in range(len(self.m_w))]
        m_b_corrected = [self.m_b[i] / (1 - self.beta1 ** self.t) for i in range(len(self.m_b))]

        v_w_corrected = [self.v_w[i] / (1 - self.beta2 ** self.t) for i in range(len(self.v_w))]
        v_b_corrected = [self.v_b[i] / (1 - self.beta2 ** self.t) for i in range(len(self.v_b))]

        # Compute the update
        grad_w = [self.lr * m_w_corrected[i] / (np.sqrt(v_w_corrected[i]) + self.epsilon) for i in range(len(m_w_corrected))]
        grad_b = [self.lr * m_b_corrected[i] / (np.sqrt(v_b_corrected[i]) + self.epsilon) for i in range(len(m_b_corrected))]

        return grad_w, grad_b

        
    def loss_gradient(self, y_true, y_prob):
        if self.loss == 'mse':
            return 2 * (y_prob - y_true)
        elif self.loss == 'crossentropy' or self.loss == 'binary_crossentropy' or self.loss == 'categorical_crossentropy':
            # Avoid vanishing gradients
            y_prob = np.where(y_prob == 0, 1e-10, y_prob)
            y_prob = np.where(y_prob == 1, 1 - 1e-10, y_prob)
            
            return (y_prob - y_true) / (y_prob * (1 - y_prob))
        else:
            raise ValueError('Loss not supported')
    
    def update_weights(self, grad_w, grad_b):
        for i, layer in enumerate(self.layers):
            layer.w = layer.w - self.lr * grad_w[i]
            layer.b = layer.b - self.lr * grad_b[i]
        
    def gradient_descent(self, X, y_true, y_prob):
        '''
        Parameters:
        y_true: One-hot encoded labels
        y_prob: Output of the last layer

        Steps to gradient descent:
        1. Initialize the gradients
        2. Compute the gradient of the error w.r.t the weights of the output layer, we call this ∂C0/∂w(L)
        3. Compute the gradient of the error w.r.t the biases of the output layer, we call this ∂C0/∂b(L) 
        4. Compute the gradient of the error w.r.t the weights of the previous layer, we call this ∂C0/∂w(L-1)
        5. Compute the gradient of the error w.r.t the biases of the previous layer, we call this ∂C0/∂b(L-1)
        6. Update the weights and biases of the output layer using the gradients computed in step 2 and 3
        7. Update the weights and biases of the previous layer using the gradients computed in step 4 and 5
        8. Repeat steps 2 to 8 until all layers are updated 

        Using chain rule:
        ∂C0/∂w(L) = ∂z(L)/∂w(L) * ∂a(L)/∂z(L) * ∂C0/∂a(L)
        ∂C0/∂b(L) = ∂z(L)/∂b(L) * ∂a(L)/∂z(L) * ∂C0/∂a(L)

        Note:
        ∂C0/∂w(L)     : Gradients of loss with respect to layer weights
        ∂z(L)/∂w(L)   : Gradients of layer output with respect to layer weights
        ∂a(L)/∂z(L)   : Gradients of layer activation with respect to layer output
        ∂C0/∂a(L)     : Gradients of loss with respect to layer activation
        ∂C0/∂b(L)     : Gradients of loss with respect to layer biases
        ∂z(L)/∂b(L)   : Gradients of layer output with respect to layer biases

        Simplified explanation:
        To calculate how much the cost changes with respect to the weights, we need to know
        1. How much the output changes with respect to the weights
        2. How much the activation changes with respect to the output
        3. How much the cost changes with respect to the activation
        '''

        # Initialize the gradients
        grad_w = []
        grad_b = []

        # Compute the gradient of the error w.r.t the weights of the output layer
        dCdA = self.loss_gradient(y_true, y_prob)
        dAdZ = self.layers[-1].activation(self.layers[-1].z, derivative=True)
        dZdW = X if len(self.layers) == 1 else self.layers[-2].a
        dCdW = np.dot(dZdW.T, dCdA * dAdZ)
        grad_w.append(dCdW)

        # Compute the gradient of the error w.r.t the biases of the output layer
        dCdB = np.sum(dAdZ * dCdA, axis=0)
        grad_b.append(dCdB)

        # Loop over the layers starting from the second-to-last layer
        for layer_idx in range(len(self.layers) - 2, -1, -1):

            # Compute the gradient of the error w.r.t the output of the previous layer
            dZdA = self.layers[layer_idx + 1].w
            # Propagate the gradient backwards by multiplying with the gradient of the activation function
            dCdA = np.dot(dCdA * dAdZ, dZdA.T)

            # Compute the gradient of the error w.r.t the weights of the previous layer
            dAdZ = self.layers[layer_idx].activation(self.layers[layer_idx].z, derivative=True)
            dZdW = X if layer_idx == 0 else self.layers[layer_idx-1].a
            dCdW = np.dot(dZdW.T, dCdA * dAdZ)
            grad_w.insert(0, dCdW)

            # Compute the gradient of the error w.r.t the biases of the previous layer
            dCdB = np.sum(dAdZ * dCdA, axis=0)
            grad_b.insert(0, dCdB)

        return grad_w, grad_b

    def fit(self, X, y, epochs=1, batch_size=32, lr=0.01, verbose=True, random_state=None, patience=None):
        self.lr = lr
        X = np.array(X)
        y = np.array(y)

        # Check if batch_size is valid
        if batch_size <= 0 or batch_size > len(X):
            raise ValueError(f'Invalid batch_size: {batch_size}')

        # Check if epochs is valid
        if epochs <= 0:
            raise ValueError(f'Invalid epochs: {epochs}')
        
        # Check if patience is valid
        if patience is not None and patience <= 0:
            raise ValueError(f'Invalid patience: {patience}')

        # Set the seed for reproducibility
        if random_state is not None:
            np.random.seed(random_state)

        # Initialize early stopping variables
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        # Initialize adam counter
        self.t = 0

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_metric = 0
            ctr = 0

            print(f'Epoch {epoch+1}/{epochs}')

            for i in range(0, len(X), batch_size):
                # Get batch
                size = min(batch_size, len(X) - i)

                # Forward propagation
                y_prob, _, loss, metric = self.forward_propagation(X[i:i+size], y[i:i+size])

                # Backward propagation
                self.backward_propagation(X[i:i+size], y[i:i+size], y_prob)

                # Update epoch loss and metric
                epoch_loss += loss
                epoch_metric += metric
                ctr += 1

                # Print progress bar
                progress = int(20 * (i + size) / len(X))
                progress_bar = '[' + '=' * progress + '>' + '-' * (29 - progress) + ']'
                if verbose:
                    print(f'{i+size}/{len(X)} {progress_bar} - loss: {loss:.4f} - {self.metric}: {metric:.4f}', end='\r')

            # Compute average epoch loss and metric
            epoch_loss /= ctr
            epoch_metric /= ctr

            if verbose:
                print(f'{len(X)}/{len(X)} [==============================] - loss: {epoch_loss:.4f} - {self.metric}: {epoch_metric:.4f}')

            # Check if validation loss improved
            if best_val_loss - epoch_loss > 1e-4:
                best_val_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Check early stopping condition
            if patience is not None and epochs_without_improvement >= patience:
                print(f'Early stopping, no improvement for {patience} epochs.')
                break

    def evaluate(self, X, y):
        X = np.array(X)
        y = np.array(y)

        _, _, loss, metric = self.forward_propagation(X, y)

        return loss, metric

    def predict(self, X):
        X = np.array(X)

        # Forward propagation
        y_prob = self.call(X)

        if y_prob.shape[-1] == 1:
            y_pred = np.array([0 if y_prob[i] > 0.5 else 1 for i in range(len(y_prob))])
        else:
            y_pred = np.argmax(y_prob, axis=-1)

        return y_pred
