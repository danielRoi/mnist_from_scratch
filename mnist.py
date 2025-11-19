"""
MNIST Digit Classification - Refactored Version

Key improvements:
1. Separated loss functions into a LossFunction class
2. Fixed gradient accumulation for proper mini-batch training
3. More modular code structure
4. Added gradient clipping for numerical stability
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import time


class LossFunction:
    """Container for different loss functions and their derivatives."""
    
    @staticmethod
    def mse(predictions, targets):
        """Mean Squared Error loss."""
        return np.mean((predictions - targets) ** 2)
    
    @staticmethod
    def mse_derivative(predictions, targets):
        """Derivative of MSE: 2 * (predictions - targets) / n."""
        n = len(targets)
        return 2 * (predictions - targets) / n
    
    @staticmethod
    def cross_entropy(predictions, targets):
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return -np.mean(targets * np.log(predictions) + (1-targets) * np.log(1-predictions))

    @staticmethod
    def cross_entropy_derivative(predictions, targets):
        """
        Derivative of Binary Cross Entropy w.r.t. the predictions (a).
        Formula: (a - y) / (a * (1 - a))
        """
        predictions = np.clip(predictions, 1e-15, 1 - 1e-15)
        return (predictions - targets) / (predictions * (1 - predictions))


class Neuron:
    """Single neuron with weights, bias, and activation function."""

    def __init__(self, n_inputs, activation='sigmoid'):
        # Xavier/He initialization for better convergence
        if activation == 'relu':
            self.weights = np.random.randn(n_inputs) * np.sqrt(2.0 / n_inputs)
        else:
            self.weights = np.random.randn(n_inputs) * np.sqrt(1.0 / n_inputs)
        self.bias = 0.0  # Initialize bias to zero
        self.activation = activation

        # Forward pass cache
        self.input = None
        self.output = None
        self.z = None
        
        # Gradient accumulation
        self.weight_gradients = np.zeros_like(self.weights)
        self.bias_gradient = 0.0

    def activate(self, z):
        """Apply activation function."""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'relu':
            return np.maximum(0, z)
        else:
            return z

    def activate_derivative(self, z):
        """Compute activation derivative."""
        if self.activation == 'sigmoid':
            s = self.activate(z)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation == 'relu':
            return (z > 0).astype(float)
        else:
            return 1

    def forward(self, inputs):
        """Forward pass computation."""
        self.input = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        self.output = self.activate(self.z)
        return self.output
    
    def zero_gradients(self):
        """Reset accumulated gradients to zero."""
        self.weight_gradients = np.zeros_like(self.weights)
        self.bias_gradient = 0.0
    
    def update_weights(self, learning_rate):
        """Apply accumulated gradients to update weights and bias."""
        self.weights -= learning_rate * self.weight_gradients
        self.bias -= learning_rate * self.bias_gradient


class Layer:
    """Layer of neurons."""

    def __init__(self, n_inputs, n_neurons, activation='sigmoid'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]
        self.n_neurons = n_neurons
        self.output = None

    def forward(self, inputs):
        """Compute layer output."""
        self.output = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.output
    
    def zero_gradients(self):
        """Reset gradients for all neurons in layer."""
        for neuron in self.neurons:
            neuron.zero_gradients()
    
    def update_weights(self, learning_rate):
        """Update weights for all neurons in layer."""
        for neuron in self.neurons:
            neuron.update_weights(learning_rate)


class NeuralNetwork:
    """Feedforward neural network."""

    def __init__(self, layer_sizes, activations=None, loss_function='mse'):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        if activations is None:
            activations = ['sigmoid'] * self.n_layers
        
        self.layers = []
        for i in range(self.n_layers):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)
        
        # Set loss function
        self.loss_function_name = loss_function
        if loss_function == 'mse':
            self.loss_fn = LossFunction.mse
            self.loss_derivative = LossFunction.mse_derivative
        elif loss_function == 'cross_entropy':
            self.loss_fn = LossFunction.cross_entropy
            self.loss_derivative = LossFunction.cross_entropy_derivative
        else:
            raise ValueError(f"Unknown loss function: {loss_function}")
        
        # Training metrics
        self.training_loss = []
        self.training_accuracy = []

    def forward(self, inputs):
        """Forward pass through network."""
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def compute_loss(self, predictions, targets):
        """Compute loss using configured loss function."""
        return self.loss_fn(predictions, targets)

    def backward_single(self, inputs, targets):
        """
        Backward pass for a single sample.
        Accumulates gradients but does NOT update weights.
        """
        # Ensure forward pass is done
        output = self.forward(inputs)
        
        # Compute output layer error using loss derivative
        output_layer = self.layers[-1]
        errors = [None] * self.n_layers
        
        output_error = []
        loss_grad = self.loss_derivative(output, targets)
        
        for i, neuron in enumerate(output_layer.neurons):
            error = loss_grad[i] * neuron.activate_derivative(neuron.z)
            output_error.append(error)
        errors[-1] = np.array(output_error)
        
        # Backpropagate errors
        for l in range(self.n_layers - 2, -1, -1):
            layer_error = []
            for i, neuron in enumerate(self.layers[l].neurons):
                error = 0
                for j, next_neuron in enumerate(self.layers[l+1].neurons):
                    error += errors[l+1][j] * next_neuron.weights[i]
                error *= neuron.activate_derivative(neuron.z)
                layer_error.append(error)
            errors[l] = np.array(layer_error)
        
        # Accumulate gradients (don't update yet)
        for l, layer in enumerate(self.layers):
            layer_input = inputs if l == 0 else self.layers[l-1].output
            for i, neuron in enumerate(layer.neurons):
                neuron.weight_gradients += errors[l][i] * layer_input
                neuron.bias_gradient += errors[l][i]
        
        return output

    def zero_gradients(self):
        """Reset all accumulated gradients to zero."""
        for layer in self.layers:
            layer.zero_gradients()

    def update_weights(self, learning_rate, batch_size):
        """Update all weights using accumulated gradients divided by batch size."""
        for layer in self.layers:
            for neuron in layer.neurons:
                # Average gradients over batch
                neuron.weight_gradients /= batch_size
                neuron.bias_gradient /= batch_size
                
                # Optional: gradient clipping
                neuron.weight_gradients = np.clip(neuron.weight_gradients, -5, 5)
                neuron.bias_gradient = np.clip(neuron.bias_gradient, -5, 5)
                
            layer.update_weights(learning_rate)

    def train_batch(self, X, y, epochs=10, learning_rate=0.1, batch_size=32,
                   X_val=None, y_val=None, verbose=True):
        """Train with proper mini-batch gradient descent."""
        n_samples = len(X)

        for epoch in range(epochs):
            start_time = time.time()

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            correct = 0

            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                actual_batch_size = len(batch_X)
                
                # Zero gradients before batch
                self.zero_gradients()
                
                # Accumulate gradients over batch
                batch_loss = 0
                for j in range(actual_batch_size):
                    output = self.backward_single(batch_X[j], batch_y[j])
                    
                    # Track loss and accuracy
                    loss = self.compute_loss(output, batch_y[j])
                    batch_loss += loss
                    
                    if np.argmax(output) == np.argmax(batch_y[j]):
                        correct += 1
                
                # Update weights once per batch
                self.update_weights(learning_rate, actual_batch_size)
                epoch_loss += batch_loss

            # Calculate metrics
            avg_loss = epoch_loss / n_samples
            train_acc = correct / n_samples * 100

            self.training_loss.append(avg_loss)
            self.training_accuracy.append(train_acc)

            # Validation
            val_acc = None
            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)

            epoch_time = time.time() - start_time

            if verbose:
                if val_acc is not None:
                    print(f"Epoch {epoch+1:2d}/{epochs} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Train Acc: {train_acc:.2f}% | "
                          f"Val Acc: {val_acc:.2f}% | "
                          f"Time: {epoch_time:.1f}s")
                else:
                    print(f"Epoch {epoch+1:2d}/{epochs} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Train Acc: {train_acc:.2f}% | "
                          f"Time: {epoch_time:.1f}s")

    def predict(self, X):
        """Generate predictions for dataset."""
        predictions = []
        for inputs in X:
            output = self.forward(inputs)
            predictions.append(output)
        return np.array(predictions)

    def evaluate(self, X, y):
        """Evaluate accuracy on dataset."""
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

    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_test = X_test_full[:n_test]
    y_test = y_test_full[:n_test]

    # Flatten images
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
    print(f"Image size: {X_train.shape[1]} pixels")

    return X_train, y_train_onehot, X_test, y_test_onehot, y_test


def visualize_predictions(nn, X_test, y_test_labels, n_samples=20):
    """Visualize predictions."""
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes = axes.ravel()

    indices = np.random.choice(len(X_test), n_samples, replace=False)

    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)
        output = nn.forward(X_test[idx])
        pred_label = np.argmax(output)
        true_label = y_test_labels[idx]
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
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(nn.training_loss)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel(f'Loss ({nn.loss_function_name.upper()})')
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
    """Main function."""
    print("=" * 70)
    print("MNIST Digit Classification - Refactored")
    print("=" * 70)

    # Load data
    X_train, y_train, X_test, y_test, y_test_labels = load_and_preprocess_mnist(
        n_train=5000, n_test=1000
    )

    # Create network
    print("\nCreating Neural Network...")
    print("Architecture: 784 -> 128 -> 64 -> 10")

    nn = NeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        activations=['relu', 'relu', 'sigmoid'],
        loss_function='cross_entropy'  # Can change to 'cross_entropy'
    )

    total_params = sum(len(n.weights) + 1 for l in nn.layers for n in l.neurons)
    print(f"Total parameters: {total_params:,}")
    print(f"Loss function: {nn.loss_function_name.upper()}")

    # Train
    print("\n" + "=" * 70)
    print("Training Network...")
    print("=" * 70)

    nn.train_batch(
        X_train, y_train,
        epochs=5,
        learning_rate=0.1,
        batch_size=32,
        X_val=X_test,
        y_val=y_test,
        verbose=True
    )

    # Evaluate
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    train_acc = nn.evaluate(X_train, y_train)
    test_acc = nn.evaluate(X_test, y_test)

    print(f"Final Training Accuracy: {train_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # Per-class accuracy
    print("\n" + "=" * 70)
    print("Per-Class Accuracy")
    print("=" * 70)

    predictions = nn.predict(X_test)
    pred_labels = np.argmax(predictions, axis=1)

    for digit in range(10):
        digit_mask = y_test_labels == digit
        digit_acc = np.mean(pred_labels[digit_mask] == digit) * 100
        digit_count = np.sum(digit_mask)
        print(f"Digit {digit}: {digit_acc:.2f}% ({digit_count} samples)")

    # Visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)

    plot_training_history(nn)
    visualize_predictions(nn, X_test, y_test_labels)

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()