import numpy as np 
import time

class Layer:
    """Fully vectorized layer that processes batches efficiently."""

    def __init__(self, n_inputs, n_neurons, activation='sigmoid'):
        # Xavier/He initialization for better convergence
        if activation == 'relu':
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2.0 / n_inputs)
        else:
            self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(1.0 / n_inputs)
        
        self.bias = np.zeros((1, n_neurons))
        self.activation = activation
        self.output = None
        self.input = None  # Cache input for backward pass

    def forward(self, inputs):
        """
        Forward pass for a BATCH of inputs.
        inputs shape: (batch_size, n_inputs)
        output shape: (batch_size, n_neurons)
        """
        self.input = inputs  # Cache for backward pass
        
        # Linear transformation: Z = X @ W + b
        # Shape: (batch_size, n_inputs) @ (n_inputs, n_neurons) = (batch_size, n_neurons)
        z = np.dot(inputs, self.weights) + self.bias
        
        # Apply activation function
        if self.activation == 'softmax':
            # Subtract max for numerical stability (per sample)
            exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
            self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        elif self.activation == 'relu':
            self.output = np.maximum(0, z)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip for stability
        else:
            self.output = z  # Linear

        return self.output


class NeuralNetwork:
    """Ultra-fast vectorized neural network with batch processing."""

    def __init__(self, layer_sizes, activations=None):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1

        if activations is None:
            activations = ['relu'] * (self.n_layers - 1) + ['softmax']

        self.layers = []
        for i in range(self.n_layers):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)

        self.training_loss = []
        self.training_accuracy = []

    def forward(self, inputs):
        """
        Forward pass for a BATCH of inputs.
        inputs shape: (batch_size, n_features)
        """
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward_batch(self, targets):
        """
        Backward pass for entire BATCH at once.
        No loops - pure vectorized operations!
        
        targets shape: (batch_size, n_classes)
        Returns: list of (weight_gradient, bias_gradient) tuples
        """
        batch_size = targets.shape[0]
        gradients = []
        
        # Start from output layer and work backwards
        delta = None
        
        for l in range(self.n_layers - 1, -1, -1):
            layer = self.layers[l]
            
            # Calculate error/delta for this layer
            if l == self.n_layers - 1:
                # Output layer: softmax + cross-entropy derivative = output - target
                # Shape: (batch_size, n_classes)
                delta = layer.output - targets
            else:
                # Hidden layer: backpropagate error
                next_layer = self.layers[l + 1]
                # Shape: (batch_size, n_neurons_next) @ (n_neurons_next, n_neurons_current)^T
                #      = (batch_size, n_neurons_current)
                delta = np.dot(delta, next_layer.weights.T)
                
                # Apply activation derivative
                if layer.activation == 'relu':
                    delta = delta * (layer.output > 0)
                elif layer.activation == 'sigmoid':
                    delta = delta * layer.output * (1 - layer.output)
            
            # Calculate gradients (averaged over batch automatically)
            # Weight gradient: input^T @ delta
            # Shape: (n_inputs, batch_size)^T @ (batch_size, n_neurons) = (n_inputs, n_neurons)
            w_grad = np.dot(layer.input.T, delta) / batch_size
            
            # Bias gradient: sum over batch dimension
            # Shape: sum over axis 0 of (batch_size, n_neurons) = (n_neurons,)
            b_grad = np.sum(delta, axis=0, keepdims=True) / batch_size
            
            # Insert at beginning to maintain layer order
            gradients.insert(0, (w_grad, b_grad))
        
        return gradients

    def train(self, X, y, epochs=10, learning_rate=0.1, batch_size=32, X_val=None, y_val=None, verbose=True):
        """
        Train using FULLY VECTORIZED Mini-Batch Gradient Descent.
        Each batch is processed as a single matrix operation - NO sample loops!
        """
        n_samples = len(X)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            epoch_correct = 0

            # Process each batch (NO inner sample loop!)
            for i in range(0, n_samples, batch_size):
                # Get batch
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]
                current_batch_size = len(X_batch)

                # 1. Forward pass for ENTIRE batch at once
                outputs = self.forward(X_batch)

                # 2. Calculate loss for entire batch (vectorized)
                epsilon = 1e-9
                # Cross-entropy loss: -sum(y * log(y_pred)) averaged over batch
                batch_loss = -np.sum(y_batch * np.log(outputs + epsilon)) / current_batch_size
                epoch_loss += batch_loss * current_batch_size

                # 3. Calculate accuracy for entire batch (vectorized)
                predictions = np.argmax(outputs, axis=1)
                targets_class = np.argmax(y_batch, axis=1)
                epoch_correct += np.sum(predictions == targets_class)

                # 4. Backward pass for ENTIRE batch at once
                gradients = self.backward_batch(y_batch)

                # 5. Update parameters
                for l, layer in enumerate(self.layers):
                    w_grad, b_grad = gradients[l]
                    layer.weights -= learning_rate * w_grad
                    layer.bias -= learning_rate * b_grad

            # Epoch statistics
            avg_loss = epoch_loss / n_samples
            train_acc = epoch_correct / n_samples * 100
            self.training_loss.append(avg_loss)
            self.training_accuracy.append(train_acc)

            val_acc = 0
            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)

            if verbose:
                print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | "
                      f"Time: {time.time() - start_time:.2f}s")

    def predict(self, X, batch_size=1000):
        """
        Vectorized prediction - process in batches for efficiency.
        """
        # Handle single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            return self.forward(X).squeeze()
        
        n_samples = len(X)
        predictions = []
        
        for i in range(0, n_samples, batch_size):
            X_batch = X[i : i + batch_size]
            batch_pred = self.forward(X_batch)
            predictions.append(batch_pred)
        
        return np.vstack(predictions)

    def evaluate(self, X, y, batch_size=1000):
        """
        Vectorized evaluation - much faster than looping.
        """
        predictions = self.predict(X, batch_size=batch_size)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)
        accuracy = np.mean(pred_classes == true_classes) * 100
        return accuracy