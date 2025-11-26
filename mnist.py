"""
MNIST Digit Classification using the OOP Neural Network
(Updated: Softmax Output + Cross-Entropy Loss)

This file is a fully commented version of your original `mnist.py`.
Each original code line has an explanatory comment added. Functions have longer explanations
where the math/idea might be unfamiliar.

UPDATES IN THIS VERSION:
1. Output Layer: Uses 'softmax' activation (probabilities summing to 1).
2. Loss Function: Uses Categorical Cross-Entropy (standard for classification).
3. Math: Gradient calculation is simplified due to the Softmax+CrossEntropy combination.

Source (original file uploaded by you): see chat message reference. 
"""

import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras
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
        self.weights = np.random.randn(n_inputs) * 0.01  # small random initial weights (1D array)
        self.bias = np.random.randn() * 0.01  # small random initial bias (scalar)
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
        Calculates gradients for a single sample but DOES NOT update weights.
        Returns a structure containing gradients for every neuron.
        """
        # 1. Forward pass to ensure current state (z, outputs) matches inputs
        self.forward(inputs)

        output_layer = self.layers[-1]
        errors = [None] * self.n_layers

        # --- Calculate Output Layer Error ---
        output_error = []
        if output_layer.activation == 'softmax':
            # Softmax + CrossEntropy derivative is simply (prediction - target)
            for i, neuron in enumerate(output_layer.neurons):
                error = neuron.output - targets[i]
                output_error.append(error)
        errors[-1] = np.array(output_error)

        # --- Backpropagate to Hidden Layers ---
        for l in range(self.n_layers - 2, -1, -1):
            layer_error = []
            for i, neuron in enumerate(self.layers[l].neurons):
                error = 0
                for j, next_neuron in enumerate(self.layers[l+1].neurons):
                    error += errors[l+1][j] * next_neuron.weights[i]
                
                error *= neuron.activate_derivative(neuron.z)
                layer_error.append(error)
            errors[l] = np.array(layer_error)

        # --- Compute Gradients (Don't update yet) ---
        # Structure: network_grads[layer_index][neuron_index] = (weight_grad, bias_grad)
        network_grads = []

        for l, layer in enumerate(self.layers):
            layer_grads = []
            layer_input = inputs if l == 0 else self.layers[l-1].output
            
            for i, neuron in enumerate(layer.neurons):
                # Gradient of Cost w.r.t Weight = error * input
                w_grad = errors[l][i] * layer_input
                # Gradient of Cost w.r.t Bias = error
                b_grad = errors[l][i]
                
                layer_grads.append((w_grad, b_grad))
            network_grads.append(layer_grads)

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


def load_and_preprocess_mnist(n_train=5000, n_test=1000):
    """Load and preprocess MNIST dataset."""
    print("Loading MNIST dataset...")
    (X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.mnist.load_data()

    # Subsample
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_test = X_test_full[:n_test]
    y_test = y_test_full[:n_test]

    # Flatten: 28x28 -> 784
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # One-hot encode
    y_train_onehot = np.zeros((len(y_train), 10))
    y_test_onehot = np.zeros((len(y_test), 10))

    for i in range(len(y_train)):
        y_train_onehot[i, y_train[i]] = 1
    for i in range(len(y_test)):
        y_test_onehot[i, y_test[i]] = 1

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    return X_train, y_train_onehot, X_test, y_test_onehot, y_test


def visualize_predictions(nn, X_test, y_test_labels, n_samples=20):
    """Visualize predictions with probabilities."""
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes = axes.ravel()

    indices = np.random.choice(len(X_test), n_samples, replace=False)

    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)

        output = nn.forward(X_test[idx])
        pred_label = np.argmax(output)
        true_label = y_test_labels[idx]
        
        # Confidence is real probability thanks to Softmax
        confidence = output[pred_label] * 100

        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                         color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=150, bbox_inches='tight')
    print("\nPredictions saved to 'mnist_predictions.png'")
    plt.show()


def plot_training_history(nn):
    """Plot training loss and accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(nn.training_loss)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (Cross-Entropy)')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(nn.training_accuracy)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training Accuracy')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("Training history saved to 'training_history.png'")
    plt.show()


def main():
    """Main function to run MNIST classification."""
    print("=" * 70)
    print("MNIST Digit Classification (Softmax + Cross-Entropy)")
    print("=" * 70)

    X_train, y_train, X_test, y_test, y_test_labels = load_and_preprocess_mnist(
        n_train=5000, n_test=1000
    )

    print("\nCreating Neural Network...")

    nn = NeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        activations=['relu', 'relu', 'softmax']
    )

    total_params = sum(len(n.weights) + 1 for l in nn.layers for n in l.neurons)
    print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 70)
    print("Training Network...")
    print("=" * 70)

    # NOTE: Learning rate can often be lower with Cross-Entropy (e.g. 0.01 or 0.05)

    nn.train_batch(
        X_train, y_train,
        epochs=5,
        learning_rate=0.05,
        batch_size=32,
        X_val=X_test,
        y_val=y_test,
        verbose=True
    )

    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    train_acc = nn.evaluate(X_train, y_train)
    test_acc = nn.evaluate(X_test, y_test)

    print(f"Final Training Accuracy: {train_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)

    plot_training_history(nn)
    visualize_predictions(nn, X_test, y_test_labels)

    print("\n" + "=" * 70)
    print("Classification Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()