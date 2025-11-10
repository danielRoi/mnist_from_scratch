import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """Single neuron with weights, bias, and activation function"""
    
    def __init__(self, n_inputs, activation='sigmoid'):
        self.weights = np.random.randn(n_inputs) * 0.5
        self.bias = np.random.randn() * 0.5
        self.activation = activation
        
        # Cache for backpropagation
        self.input = None
        self.output = None
        self.z = None
    
    def activate(self, z):
        """Apply activation function"""
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'relu':
            return np.maximum(0, z)
        else:
            return z
    
    def activate_derivative(self, z):
        """Derivative of activation function"""
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
        """Forward pass through neuron"""
        self.input = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        self.output = self.activate(self.z)
        return self.output
    
    def __repr__(self):
        return f"Neuron(weights={self.weights}, bias={self.bias:.3f})"


class Layer:
    """Layer of neurons"""
    
    def __init__(self, n_inputs, n_neurons, activation='sigmoid'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]
        self.n_neurons = n_neurons
        self.output = None
    
    def forward(self, inputs):
        """Forward pass through layer"""
        self.output = np.array([neuron.forward(inputs) for neuron in self.neurons])
        return self.output
    
    def get_weights_matrix(self):
        """Get weights as matrix for visualization"""
        return np.array([neuron.weights for neuron in self.neurons])
    
    def get_biases(self):
        """Get biases as vector"""
        return np.array([neuron.bias for neuron in self.neurons])
    
    def __repr__(self):
        return f"Layer({self.n_neurons} neurons)"


class NeuralNetwork:
    """Feedforward neural network"""
    
    def __init__(self, layer_sizes, activations=None):
        """
        Initialize network
        
        Args:
            layer_sizes: list of integers [input_size, hidden1, hidden2, ..., output_size]
            activations: list of activation functions for each layer (default: all sigmoid)
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        if activations is None:
            activations = ['sigmoid'] * self.n_layers
        
        self.layers = []
        for i in range(self.n_layers):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)
        
        self.training_loss = []
    
    def forward(self, inputs):
        """Forward propagation through network"""
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, inputs, targets, learning_rate=0.1):
        """Backpropagation algorithm"""
        # Forward pass
        self.forward(inputs)
        
        # Compute output layer error
        output_layer = self.layers[-1]
        errors = [None] * self.n_layers
        
        # Output layer error: (output - target) * activation_derivative
        output_error = []
        for i, neuron in enumerate(output_layer.neurons):
            error = (neuron.output - targets[i]) * neuron.activate_derivative(neuron.z)
            output_error.append(error)
        errors[-1] = np.array(output_error)
        
        # Backpropagate error through hidden layers
        for l in range(self.n_layers - 2, -1, -1):
            layer_error = []
            for i, neuron in enumerate(self.layers[l].neurons):
                # Sum of weighted errors from next layer
                error = 0
                for j, next_neuron in enumerate(self.layers[l+1].neurons):
                    error += errors[l+1][j] * next_neuron.weights[i]
                error *= neuron.activate_derivative(neuron.z)
                layer_error.append(error)
            errors[l] = np.array(layer_error)
        
        # Update weights and biases
        for l, layer in enumerate(self.layers):
            layer_input = inputs if l == 0 else self.layers[l-1].output
            for i, neuron in enumerate(layer.neurons):
                # Update weights
                neuron.weights -= learning_rate * errors[l][i] * layer_input
                # Update bias
                neuron.bias -= learning_rate * errors[l][i]
    
    def train(self, X, y, epochs=10000, learning_rate=0.5, verbose=True):
        """
        Train the network
        
        Args:
            X: input data (n_samples x n_features)
            y: target data (n_samples x n_outputs)
            epochs: number of training epochs
            learning_rate: learning rate
            verbose: print progress
        """
        self.training_loss = []
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Train on each sample
            for i in range(len(X)):
                self.backward(X[i], y[i], learning_rate)
                
                # Calculate loss (MSE)
                output = self.forward(X[i])
                loss = np.mean((output - y[i]) ** 2)
                total_loss += loss
            
            avg_loss = total_loss / len(X)
            self.training_loss.append(avg_loss)
            
            if verbose and (epoch % 1000 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:5d} | Loss: {avg_loss:.6f}")
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for inputs in X:
            output = self.forward(inputs)
            predictions.append(output)
        return np.array(predictions)
    
    def __repr__(self):
        layers_info = " -> ".join([str(size) for size in self.layer_sizes])
        return f"NeuralNetwork({layers_info})"


# XOR Problem Simulation
def simulate_xor():
    """Simulate XOR problem using neural network"""
    
    print("=" * 60)
    print("XOR Problem Simulation with Neural Network")
    print("=" * 60)
    
    # XOR training data
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    y = np.array([
        [0],  # 0 XOR 0 = 0
        [1],  # 0 XOR 1 = 1
        [1],  # 1 XOR 0 = 1
        [2]   # 1 XOR 1 = 0
    ])
    
    print("\nTraining Data:")
    print("Input | Expected Output")
    print("-" * 25)
    for i in range(len(X)):
        print(f"{X[i]} | {y[i][0]}")
    
    # Create neural network: 2 inputs -> 4 hidden -> 1 output
    nn = NeuralNetwork([2, 4, 1], activations=['sigmoid', 'sigmoid'])
    print(f"\n{nn}")
    print(f"Total parameters: {sum(len(n.weights) + 1 for l in nn.layers for n in l.neurons)}")
    
    # Train the network
    print("\nTraining...")
    print("-" * 60)
    nn.train(X, y, epochs=10000, learning_rate=0.5, verbose=True)
    
    # Test the network
    print("\n" + "=" * 60)
    print("Testing Results:")
    print("=" * 60)
    print("Input | Expected | Predicted | Rounded")
    print("-" * 50)
    
    predictions = nn.predict(X)
    for i in range(len(X)):
        pred_val = predictions[i][0]
        rounded = int(pred_val > 0.5)
        correct = "✓" if rounded == y[i][0] else "✗"
        print(f"{X[i]} | {y[i][0]:8.0f} | {pred_val:9.6f} | {rounded:7d} {correct}")
    
    # Calculate accuracy
    accuracy = np.mean((predictions > 0.5).astype(int) == y) * 100
    print(f"\nAccuracy: {accuracy:.1f}%")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(nn.training_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    
    # Plot decision boundary
    plt.subplot(1, 2, 2)
    
    # Create mesh
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict for each point in mesh
    mesh_predictions = []
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            pred = nn.forward(np.array([xx[i, j], yy[i, j]]))[0]
            mesh_predictions.append(pred)
    
    Z = np.array(mesh_predictions).reshape(xx.shape)
    
    # Plot contour
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='Output')
    
    # Plot training points
    colors = ['blue' if yi[0] == 0 else 'red' for yi in y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidths=2)
    
    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Decision Boundary')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)


# Run the simulation
if __name__ == "__main__":
    simulate_xor()