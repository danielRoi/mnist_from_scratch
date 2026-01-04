import numpy as np 
import time


class Optimizer:
    """Base optimizer class."""
    
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.t = 0  # Time step for optimizers that need it
    
    def update(self, layers, gradients):
        """Update layer parameters given gradients."""
        raise NotImplementedError
    
    def get_state(self):
        """Get optimizer state for saving."""
        return {'t': self.t, 'learning_rate': self.learning_rate}
    
    def set_state(self, state):
        """Set optimizer state for loading."""
        self.t = state.get('t', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)

class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay."""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = []  # First moment (mean)
        self.v = []  # Second moment (variance)
    
    def update(self, layers, gradients):
        # Initialize moments on first call
        if len(self.m) == 0:
            for layer in layers:
                self.m.append({
                    'w': np.zeros_like(layer.weights),
                    'b': np.zeros_like(layer.bias)
                })
                self.v.append({
                    'w': np.zeros_like(layer.weights),
                    'b': np.zeros_like(layer.bias)
                })
        
        self.t += 1
        
        for i, layer in enumerate(layers):
            w_grad, b_grad = gradients[i]
            
            # Update biased first moment estimate
            self.m[i]['w'] = self.beta1 * self.m[i]['w'] + (1 - self.beta1) * w_grad
            self.m[i]['b'] = self.beta1 * self.m[i]['b'] + (1 - self.beta1) * b_grad
            
            # Update biased second moment estimate
            self.v[i]['w'] = self.beta2 * self.v[i]['w'] + (1 - self.beta2) * w_grad**2
            self.v[i]['b'] = self.beta2 * self.v[i]['b'] + (1 - self.beta2) * b_grad**2
            
            # Compute bias-corrected moments
            m_hat_w = self.m[i]['w'] / (1 - self.beta1**self.t)
            m_hat_b = self.m[i]['b'] / (1 - self.beta1**self.t)
            v_hat_w = self.v[i]['w'] / (1 - self.beta2**self.t)
            v_hat_b = self.v[i]['b'] / (1 - self.beta2**self.t)
            
            # Update parameters with decoupled weight decay
            # Key difference from Adam: weight decay is applied directly to weights
            layer.weights = layer.weights * (1 - self.learning_rate * self.weight_decay) - \
                           self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            layer.bias -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
    
    def get_state(self):
        state = super().get_state()
        state.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'weight_decay': self.weight_decay,
            'm': self.m,
            'v': self.v
        })
        return state
    
    def set_state(self, state):
        super().set_state(state)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.epsilon = state.get('epsilon', self.epsilon)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        self.m = state.get('m', [])
        self.v = state.get('v', [])


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

    def __init__(self, layer_sizes, activations=None, optimizer='adam', learning_rate=0.001, **optimizer_kwargs):
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
        
        # Initialize optimizer
        self.optimizer_name = optimizer.lower()
        if self.optimizer_name == 'adamw':
            self.optimizer = AdamW(learning_rate=learning_rate, **optimizer_kwargs)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}.")

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

    def train(self, X, y, epochs=10, batch_size=32, X_val=None, y_val=None, verbose=True):
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

                # 5. Update parameters using optimizer
                self.optimizer.update(self.layers, gradients)

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

    def save(self, filepath):
        """
        Save complete model state to a .npz file (fast, compressed).
        Preserves all weights, biases, architecture, training history, and optimizer state.
        """
        save_dict = {
            'layer_sizes': self.layer_sizes,
            'training_loss': self.training_loss,
            'training_accuracy': self.training_accuracy,
            'optimizer_name': self.optimizer_name,
            'optimizer_state': self.optimizer.get_state(),
        }
        
        # Save each layer's weights, biases, and activation
        for i, layer in enumerate(self.layers):
            save_dict[f'layer_{i}_weights'] = layer.weights
            save_dict[f'layer_{i}_bias'] = layer.bias
            save_dict[f'layer_{i}_activation'] = layer.activation
        
        np.savez_compressed(filepath, **save_dict)

    def load(self, filepath):
        """
        Load complete model state from a .npz file.
        Restores exact model state including optimizer - can continue training seamlessly.
        """
        data = np.load(filepath, allow_pickle=True)
        
        # Restore architecture
        self.layer_sizes = data['layer_sizes'].tolist()
        self.n_layers = len(self.layer_sizes) - 1
        self.training_loss = data['training_loss'].tolist()
        self.training_accuracy = data['training_accuracy'].tolist()
        
        # Restore layers
        self.layers = []
        for i in range(self.n_layers):
            activation = str(data[f'layer_{i}_activation'])
            layer = Layer(self.layer_sizes[i], self.layer_sizes[i+1], activation)
            layer.weights = data[f'layer_{i}_weights']
            layer.bias = data[f'layer_{i}_bias']
            self.layers.append(layer)
        
        # Restore optimizer
        if 'optimizer_name' in data:
            self.optimizer_name = str(data['optimizer_name'])
            optimizer_state = data['optimizer_state'].item()
            
            # Recreate optimizer with saved state
            if self.optimizer_name == 'adamw':
                self.optimizer = AdamW()
            
            self.optimizer.set_state(optimizer_state)

    @staticmethod
    def load_model(filepath):
        """
        Static method to load a model without needing to create an instance first.
        Returns a fully restored NeuralNetwork ready to use.
        
        Usage: nn = NeuralNetwork.load_model('my_model.npz')
        """
        data = np.load(filepath, allow_pickle=True)
        layer_sizes = data['layer_sizes'].tolist()
        
        # Determine optimizer from saved state
        optimizer_name = str(data.get('optimizer_name', 'adam'))
        
        # Create new instance with loaded architecture
        nn = NeuralNetwork(layer_sizes, optimizer=optimizer_name)
        
        # Restore training history
        nn.training_loss = data['training_loss'].tolist()
        nn.training_accuracy = data['training_accuracy'].tolist()
        
        # Restore layers
        for i in range(nn.n_layers):
            activation = str(data[f'layer_{i}_activation'])
            nn.layers[i].activation = activation
            nn.layers[i].weights = data[f'layer_{i}_weights']
            nn.layers[i].bias = data[f'layer_{i}_bias']
        
        # Restore optimizer state
        if 'optimizer_state' in data:
            optimizer_state = data['optimizer_state'].item()
            nn.optimizer.set_state(optimizer_state)
        
        return nn