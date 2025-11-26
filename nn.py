import numpy as np 
import time

class Neuron:
    """Single neuron with weights, bias, and activation function.

    Detailed explanation:
    - A neuron computes z = w.x + b where w is weights vector, x is input vector, b is a scalar bias.
    - The activation function transforms z to produce the neuron's output.
    - We keep `input`, `z`, and `output` stored on the object so the backward pass can use them.

    Activation functions and their intuition (simple links):
    - Sigmoid: maps to (0,1). Good for hidden layers. https://en.wikipedia.org/wiki/Sigmoid_function
    - ReLU: returns 0 for negative inputs and identity for positives; fast learning. https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    - Softmax: (Special case handled in Layer class) maps a vector to probabilities summing to 1.
    """

    def __init__(self, n_inputs, activation='sigmoid'):
        # Xavier/He initialization for better convergence
        if activation == 'relu':
            self.weights = np.random.randn(n_inputs) * np.sqrt(2.0 / n_inputs)
        else:
            self.weights = np.random.randn(n_inputs) * np.sqrt(1.0 / n_inputs)

        self.bias = 0
        self.activation = activation  # activation function name stored as string

        self.input = None  # placeholder to save last input seen by this neuron (used in backprop)
        self.output = None  # placeholder to save last output of this neuron
        self.z = None  # placeholder to save last linear combination (w.x + b)

    def activate(self, z):
        # Apply the chosen activation function to the pre-activation value z.
        # NOTE: Softmax is a special case applied at the LAYER level, not individual neuron level.
        # So if self.activation is 'softmax', we pass z through (linear) and let Layer handle it.
        
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'softmax':
            return z  # Return linear z; Layer.forward will calculate the actual softmax probability
        else:
            return z

    def activate_derivative(self, z):
        # Return derivative of the activation function evaluated at z; used in backpropagation.
        # Note: Softmax derivative is handled specially in NeuralNetwork.backward.
        if self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation == 'relu':
            return (z > 0).astype(float)
        else:
            return 1

    def forward(self, inputs):
        # Compute the neuron's forward pass for a single input vector.
        self.input = inputs  # save input for gradient computation in backward pass
        self.z = np.dot(self.weights, inputs) + self.bias  # linear combination w.x + b
        self.output = self.activate(self.z)  # apply activation function
        return self.output  # return scalar output for this neuron


class Layer:
    """Layer of neurons: container that holds multiple Neuron objects.

    Special Handling for Softmax:
    - Softmax requires looking at ALL neurons in the layer to normalize (divide by sum).
    - Therefore, if activation is 'softmax', this class modifies the outputs after the neurons run.
    """

    def __init__(self, n_inputs, n_neurons, activation='sigmoid'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]
        self.n_neurons = n_neurons
        self.activation = activation
        self.output = None  # will hold the output array after forward pass

    def forward(self, inputs):
        # 1. Calculate individual neuron outputs (or raw z if softmax)
        raw_outputs = np.array([neuron.forward(inputs) for neuron in self.neurons])
        
        # 2. Apply Softmax logic if this is the output layer
        if self.activation == 'softmax':
            # Collect all 'z' values (linear outputs) from the neurons
            z_values = np.array([n.z for n in self.neurons])
            
            # Softmax Formula: e^z_i / sum(e^z_j)
            # Numeric Stability Trick: subtract max(z) to prevent e^z becoming infinite
            exp_values = np.exp(z_values - np.max(z_values)) 
            self.output = exp_values / np.sum(exp_values)
            
            # CRITICAL: Update the individual neurons with their new softmax probability output
            for i, neuron in enumerate(self.neurons):
                neuron.output = self.output[i]
        else:
            self.output = raw_outputs

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
        # 1. Forward pass to ensure current state is fresh
        self.forward(inputs)
        network_grads = [None] * self.n_layers
        
        # We need to store the errors of the "next" layer (which we process first because we go backwards)
        # to calculate the errors of the current layer.
        next_layer_errors = None

        # --- Single Loop: Iterate from Output Layer down to Input Layer ---
        for l in range(self.n_layers - 1, -1, -1):
            layer = self.layers[l]
            layer_errors = []

            # --- A. Calculate Error (Delta) for this layer ---
            if l == self.n_layers - 1:
                # Output Layer (Softmax + CrossEntropy case)
                for i, neuron in enumerate(layer.neurons):
                    error = neuron.output - targets[i]
                    layer_errors.append(error)
            else:
                # Hidden Layer
                next_layer = self.layers[l+1]
                for i, neuron in enumerate(layer.neurons):
                    error = 0.0
                    # Aggregate error from the next layer (which we already processed)
                    for j, next_neuron in enumerate(next_layer.neurons):
                        error += next_layer_errors[j] * next_neuron.weights[i]
                    
                    # Multiply by derivative
                    error *= neuron.activate_derivative(neuron.z)
                    layer_errors.append(error)
            
            # Convert to numpy array for the next iteration to use easily
            layer_errors = np.array(layer_errors)
            
            # --- B. Compute Gradients IMMEDIATELY ---
            # Input to this layer is either network input (if l=0) or prev layer output
            layer_input = inputs if l == 0 else self.layers[l-1].output
            
            layer_grads = []
            for i, neuron in enumerate(layer.neurons):
                # Gradient w.r.t Weight
                w_grad = layer_errors[i] * layer_input
                # Gradient w.r.t Bias
                b_grad = layer_errors[i]
                
                layer_grads.append((w_grad, b_grad))
            
            # Store gradients for this layer
            network_grads[l] = layer_grads
            
            # Update 'next_layer_errors' to be the current ones, for the next iteration (l-1)
            next_layer_errors = layer_errors

        return network_grads

    def update_params(self, accumulated_grads, learning_rate, batch_size):
        """
        Updates weights and biases using the averaged gradients from the batch.
        """
        for l, layer in enumerate(self.layers):
            for i, neuron in enumerate(layer.neurons):
                # Get sum of gradients for this neuron
                sum_w_grad, sum_b_grad = accumulated_grads[l][i]
                
                # Average the gradients
                avg_w_grad = sum_w_grad / batch_size
                avg_b_grad = sum_b_grad / batch_size

                # Update weights and bias
                neuron.weights -= learning_rate * avg_w_grad
                neuron.bias -= learning_rate * avg_b_grad

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
                # Structure matches 'network_grads' from backward()
                # We need zeros of correct shape. 
                # Helper structure: accumulator[layer][neuron] = [w_grad_sum, b_grad_sum]
                batch_accumulated_grads = []
                for layer in self.layers:
                    layer_acc = []
                    for neuron in layer.neurons:
                        layer_acc.append([np.zeros_like(neuron.weights), 0.0])
                    batch_accumulated_grads.append(layer_acc)

                # Process samples in batch
                for j in range(current_batch_size):
                    inputs = X_batch[j]
                    targets = y_batch[j]

                    # 1. Get gradients for single sample
                    sample_grads = self.backward(inputs, targets)

                    # 2. Accumulate gradients
                    for l in range(len(self.layers)):
                        for n in range(len(self.layers[l].neurons)):
                            w_g, b_g = sample_grads[l][n]
                            batch_accumulated_grads[l][n][0] += w_g
                            batch_accumulated_grads[l][n][1] += b_g

                    # Logging (Forward pass logic for loss tracking)
                    # Note: We rely on the forward pass inside backward() usually, 
                    # but we need the output here for stats.
                    output = self.layers[-1].output 
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
