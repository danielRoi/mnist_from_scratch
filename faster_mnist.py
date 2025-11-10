"""
Optimized MNIST Neural Network with Vectorized Operations and Softmax
Performance improvements: ~50-100x faster than neuron-by-neuron approach
Improved for better probability distributions and hand-drawn digit recognition
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import time


class OptimizedLayer:
    """Vectorized layer using matrix operations"""
    
    def __init__(self, n_inputs, n_neurons, activation='sigmoid'):
        # Initialize weights as a matrix: (n_neurons, n_inputs)
        self.weights = np.random.randn(n_neurons, n_inputs) * np.sqrt(2.0 / n_inputs)  # He initialization
        self.bias = np.zeros((n_neurons, 1))
        self.activation = activation
        
        # Cache for backpropagation
        self.input = None
        self.z = None
        self.output = None
    
    def activate(self, z):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'softmax':
            # Softmax for output layer (batch-wise)
            z_clipped = np.clip(z, -500, 500)
            exp_z = np.exp(z_clipped - np.max(z_clipped, axis=0, keepdims=True))
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        else:
            return z
    
    def activate_derivative(self, z):
        if self.activation == 'sigmoid':
            s = self.activate(z)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'softmax':
            # For softmax with cross-entropy, derivative is handled differently
            return np.ones_like(z)
        else:
            return np.ones_like(z)
    
    def forward(self, inputs):
        """
        Forward pass for a batch
        inputs: (n_inputs, batch_size)
        returns: (n_neurons, batch_size)
        """
        self.input = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        self.output = self.activate(self.z)
        return self.output


class OptimizedNeuralNetwork:
    """Vectorized feedforward neural network with softmax output"""
    
    def __init__(self, layer_sizes, activations=None):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        if activations is None:
            activations = ['relu'] * (self.n_layers - 1) + ['softmax']
        
        self.layers = []
        for i in range(self.n_layers):
            layer = OptimizedLayer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)
        
        self.training_loss = []
        self.training_accuracy = []
    
    def forward(self, inputs):
        """
        Forward pass through network
        inputs: (n_inputs, batch_size) or (n_inputs,) for single sample
        """
        # Handle single sample
        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward_batch(self, inputs, targets, learning_rate=0.1):
        """
        Vectorized backpropagation for a batch with cross-entropy loss
        inputs: (n_inputs, batch_size)
        targets: (n_outputs, batch_size)
        """
        batch_size = inputs.shape[1]
        
        # Forward pass
        self.forward(inputs)
        
        # Compute output layer error
        # For softmax + cross-entropy, the gradient simplifies to (output - target)
        output_layer = self.layers[-1]
        if output_layer.activation == 'softmax':
            delta = output_layer.output - targets
        else:
            delta = (output_layer.output - targets) * output_layer.activate_derivative(output_layer.z)
        
        # Backpropagate through layers
        deltas = [delta]
        
        for l in range(self.n_layers - 2, -1, -1):
            delta = np.dot(self.layers[l + 1].weights.T, delta) * \
                    self.layers[l].activate_derivative(self.layers[l].z)
            deltas.insert(0, delta)
        
        # Update weights and biases
        for l, layer in enumerate(self.layers):
            layer_input = inputs if l == 0 else self.layers[l - 1].output
            
            # Gradient averaging over batch
            dW = np.dot(deltas[l], layer_input.T) / batch_size
            db = np.sum(deltas[l], axis=1, keepdims=True) / batch_size
            
            # Update parameters
            layer.weights -= learning_rate * dW
            layer.bias -= learning_rate * db
    
    def cross_entropy_loss(self, predictions, targets):
        """Calculate cross-entropy loss (better for classification)"""
        # Clip predictions to avoid log(0)
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        # Cross-entropy: -sum(target * log(prediction))
        loss = -np.sum(targets * np.log(predictions)) / predictions.shape[1]
        return loss
    
    def train_batch(self, X, y, epochs=10, learning_rate=0.1, batch_size=32,
                   X_val=None, y_val=None, verbose=True):
        """Train with vectorized mini-batch gradient descent"""
        n_samples = len(X)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            correct = 0
            
            # Mini-batch training with vectorized operations
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Transpose for matrix operations: (batch_size, features) -> (features, batch_size)
                batch_X_T = batch_X.T
                batch_y_T = batch_y.T
                
                # Single backward pass for entire batch
                self.backward_batch(batch_X_T, batch_y_T, learning_rate)
                
                # Compute loss and accuracy for this batch
                output = self.forward(batch_X_T)
                loss = self.cross_entropy_loss(output, batch_y_T)
                total_loss += loss * len(batch_X)
                
                # Count correct predictions
                predictions = np.argmax(output, axis=0)
                true_labels = np.argmax(batch_y, axis=1)
                correct += np.sum(predictions == true_labels)
            
            avg_loss = total_loss / n_samples
            train_acc = correct / n_samples * 100
            
            self.training_loss.append(avg_loss)
            self.training_accuracy.append(train_acc)
            
            # Validation
            val_acc = 0
            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)
            
            epoch_time = time.time() - start_time
            
            if verbose:
                if X_val is not None:
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
    
    def predict(self, X, batch_size=256):
        """Predict in batches for efficiency"""
        n_samples = len(X)
        all_predictions = []
        
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size].T
            output = self.forward(batch_X)
            all_predictions.append(output.T)
        
        return np.vstack(all_predictions)
    
    def evaluate(self, X, y, batch_size=256):
        """Evaluate accuracy on dataset"""
        predictions = self.predict(X, batch_size)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(pred_labels == true_labels) * 100
        return accuracy


def load_and_preprocess_mnist(n_train=5000, n_test=1000, binarize=True, threshold=0.5):
    """
    Load and preprocess MNIST dataset
    
    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        binarize: If True, convert grayscale to pure black/white
        threshold: Threshold for binarization (0-1)
    """
    print("Loading MNIST dataset...")
    (X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.mnist.load_data()
    
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_test = X_test_full[:n_test]
    y_test = y_test_full[:n_test]
    
    # Flatten and normalize
    X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
    
    # Binarize if requested (converts to pure black/white like hand-drawn)
    if binarize:
        print(f"Binarizing images with threshold {threshold}...")
        X_train = (X_train > threshold).astype('float32')
        X_test = (X_test > threshold).astype('float32')
    
    # One-hot encode
    y_train_onehot = np.eye(10)[y_train]
    y_test_onehot = np.eye(10)[y_test]
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    if binarize:
        print("Images binarized to pure black/white")
    
    return X_train, y_train_onehot, X_test, y_test_onehot, y_test


def visualize_predictions(nn, X_test, y_test_labels, n_samples=20):
    """Visualize some predictions"""
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes = axes.ravel()
    
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)
        output = nn.forward(X_test[idx])
        pred_label = np.argmax(output)
        true_label = y_test_labels[idx]
        confidence = output.ravel()[pred_label] * 100
        
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
    """Plot training metrics"""
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
    """Main function"""
    print("=" * 70)
    print("MNIST Digit Classification with Improved Neural Network")
    print("=" * 70)
    
    # Load data with binarization
    X_train, y_train, X_test, y_test, y_test_labels = load_and_preprocess_mnist(
        n_train=5000, 
        n_test=10000,
        binarize=True,  # Convert to pure black/white
        threshold=0.5
    )
    
    # Create network with softmax output
    print("\nCreating Neural Network...")
    print("Architecture: 784 (input) -> 128 (hidden) -> 64 (hidden) -> 10 (output)")
    print("Activations: ReLU -> ReLU -> Softmax")
    
    nn = OptimizedNeuralNetwork(
        layer_sizes=[784, 64, 64, 10],
        activations=['relu', 'relu', 'softmax']
    )
    
    total_params = sum(layer.weights.size + layer.bias.size for layer in nn.layers)
    print(f"Total parameters: {total_params:,}")
    
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
    
    # Final evaluation
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
    
    # Show probability distribution quality
    print("\n" + "=" * 70)
    print("Probability Distribution Analysis")
    print("=" * 70)
    
    sample_preds = predictions[:100]
    max_probs = np.max(sample_preds, axis=1)
    print(f"Average maximum probability: {np.mean(max_probs)*100:.1f}%")
    print(f"Median maximum probability: {np.median(max_probs)*100:.1f}%")
    print(f"Min/Max probability: {np.min(max_probs)*100:.1f}% / {np.max(max_probs)*100:.1f}%")
    
    # Visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)
    
    plot_training_history(nn)
    visualize_predictions(nn, X_test, y_test_labels)
    
    print("\n" + "=" * 70)
    print("MNIST Classification Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()