import numpy as np 
import time

class Layer:
    """Layer of neurons: container that holds multiple Neuron objects.

    Special Handling for Softmax:
    - Softmax requires looking at ALL neurons in the layer to normalize (divide by sum).
    - Therefore, if activation is 'softmax', this class modifies the outputs after the neurons run.
    """

    def __init__(self, n_inputs, n_neurons, activation='sigmoid'):
        # Xavier/He initialization for better convergence
        if activation == 'relu':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        else:
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1.0 / n_inputs)
        self.bias = np.zeros((1, n_neurons))
        self.activation = activation
        self.output = None  # will hold the output array after forward pass

    def forward(self, inputs):
        # 1. Calculate linear combination (z = X*W + b) and store it
        z = np.dot(inputs, self.weights) + self.bias
        
        # 2. Apply activation function
        if self.activation == 'softmax':
            # Softmax Formula: e^z_i / sum(e^z_j)
            # Numeric Stability Trick: subtract max(z) to prevent e^z becoming infinite
            exp_values = np.exp(z - np.max(z, axis=-1, keepdims=True))
            self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        elif self.activation == 'relu':
            self.output = np.maximum(0, z)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-z))
        else:
            self.output = z  # linear activation

        return self.output


class NeuralNetwork:
    """Feedforward neural network built from Layer and Neuron objects.

    Changes for this version:
    - Supports Cross-Entropy Loss.
    - Supports Softmax output layer.
    """

    def __init__(self, layer_sizes, activations=None):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1

        if activations is None:
            # Default: ReLU for hidden, Softmax for output (standard for classification)
            activations = ['relu'] * (self.n_layers - 1) + ['softmax']

        self.layers = []
        for i in range(self.n_layers):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)

        self.training_loss = []
        self.training_accuracy = []

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, inputs, targets):
        """
        Calculates gradients in a SINGLE backward pass.
        Merges error calculation and gradient computation for efficiency.
        """
        # Forward pass is assumed to have been called already
        # (it's called in train() before backward())
        
        network_grads = []
        
        # Store errors for backpropagation
        next_layer_error = None

        # --- Single Loop: Iterate from Output Layer down to Input Layer ---
        for l in range(self.n_layers - 1, -1, -1):
            layer = self.layers[l]

            # --- A. Calculate Error (Delta) for this layer ---
            if l == self.n_layers - 1:
                # Output Layer (Softmax + CrossEntropy case)
                # The derivative of softmax + cross-entropy simplifies to: output - target
                error = layer.output - targets
            else:
                # Hidden Layer: backpropagate error from next layer
                next_layer = self.layers[l + 1]
                error = np.dot(next_layer_error, next_layer.weights.T)
                
                # Apply activation derivative
                if layer.activation == 'relu':
                    error = error * (layer.output > 0)  # ReLU derivative
                elif layer.activation == 'sigmoid':
                    sigmoid_deriv = layer.output * (1 - layer.output)
                    error = error * sigmoid_deriv
            
            # --- B. Compute Gradients IMMEDIATELY ---
            # Input to this layer is either network input (if l=0) or prev layer output
            if l == 0:
                layer_input = inputs.reshape(1, -1)  # Ensure 2D
            else:
                layer_input = self.layers[l-1].output.reshape(1, -1)
            
            # Gradient w.r.t Weight: outer product of input and error
            # Shape: (n_inputs, 1) @ (1, n_neurons) = (n_inputs, n_neurons)
            w_grad = np.dot(layer_input.T, error.reshape(1, -1))
            
            # Gradient w.r.t Bias: just the error
            b_grad = error.reshape(1, -1)
            
            # Store gradients for this layer (insert at beginning to maintain order)
            network_grads.insert(0, (w_grad, b_grad))
            
            # Update error for next iteration (going backwards)
            next_layer_error = error

        return network_grads

    def update_params(self, accumulated_grads, learning_rate, batch_size):
        """
        Updates weights and biases using the averaged gradients from the batch.
        """
        for l, layer in enumerate(self.layers):
            sum_w_grad, sum_b_grad = accumulated_grads[l]
            # Average the gradients
            avg_w_grad = sum_w_grad / batch_size
            avg_b_grad = sum_b_grad / batch_size
            # Update weights and bias
            layer.weights -= learning_rate * avg_w_grad
            layer.bias -= learning_rate * avg_b_grad

    def train(self, X, y, epochs=10, learning_rate=0.1, batch_size=32, X_val=None, y_val=None, verbose=True):
        """
        Train using Mini-Batch Gradient Descent.
        """
        n_samples = len(X)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_loss = 0
            correct = 0

            # --- Mini-Batch Loop ---
            for i in range(0, n_samples, batch_size):
                # Get batch slice
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                current_batch_size = len(X_batch) # Handle last batch if smaller

                # Initialize accumulator for gradients for this batch
                batch_accumulated_grads = [
                    (np.zeros_like(layer.weights), np.zeros_like(layer.bias))
                    for layer in self.layers
                ]

                # Process samples in batch
                for j in range(current_batch_size):
                    inputs = X_batch[j]
                    targets = y_batch[j]

                    # Forward pass
                    output = self.forward(inputs)

                    # 1. Get gradients for single sample
                    sample_grads = self.backward(inputs, targets)

                    # 2. Accumulate gradients
                    for l in range(len(self.layers)):
                        w_g, b_g = sample_grads[l]
                        batch_accumulated_grads[l] = (
                            batch_accumulated_grads[l][0] + w_g,
                            batch_accumulated_grads[l][1] + b_g
                        )

                    # Track loss and accuracy
                    epsilon = 1e-9
                    loss = -np.sum(targets * np.log(output + epsilon))
                    total_loss += loss
                    if np.argmax(output) == np.argmax(targets):
                        correct += 1

                # 3. Apply updates after the batch is finished
                self.update_params(batch_accumulated_grads, learning_rate, current_batch_size)

            # Epoch stats
            avg_loss = total_loss / n_samples
            train_acc = correct / n_samples * 100
            self.training_loss.append(avg_loss)
            self.training_accuracy.append(train_acc)

            val_acc = 0
            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)

            if verbose:
                print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                      f"Time: {time.time() - start_time:.1f}s")


    def predict(self, X):
        predictions = []
        for inputs in X:
            output = self.forward(inputs)
            predictions.append(output)
        return np.array(predictions)

    def evaluate(self, X, y):
        correct = 0
        predictions = self.predict(X)

        for i in range(len(X)):
            if np.argmax(predictions[i]) == np.argmax(y[i]):
                correct += 1

        return correct / len(X) * 100