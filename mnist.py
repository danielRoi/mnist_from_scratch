"""
MNIST Digit Classification using the OOP Neural Network

This file is a fully commented version of your original `mnist.py`.
Each original code line has an explanatory comment added. Functions have longer explanations
where the math/idea might be unfamiliar. Links to accessible, high-school-level resources
are provided next to lines that benefit from external explanation (activation functions, one-hot,
flattening images, gradient descent, etc.). Links chosen are simple and assume at most
high-school algebra knowledge.

Source (original file uploaded by you): see chat message reference. 
"""

import numpy as np  # import NumPy for numerical arrays and vectorized operations
import matplotlib.pyplot as plt  # import matplotlib for plotting and saving images
from tensorflow import keras  # import Keras (bundled in TensorFlow) to load MNIST dataset
import time  # import time to measure training epoch durations

# If you separated classes into another file, you would import them like this:
# from neural_network import Neuron, Layer, NeuralNetwork
# (left commented because the classes are included below for a self-contained example)


class Neuron:
    """Single neuron with weights, bias, and activation function.

    Detailed explanation:
    - A neuron computes z = w.x + b where w is weights vector, x is input vector, b is a scalar bias.
    - The activation function (sigmoid/tanh/relu) then transforms z to produce the neuron's output.
    - We keep `input`, `z`, and `output` stored on the object so the backward pass can use them.

    Activation functions and their intuition (simple links):
    - Sigmoid: maps numbers to (0,1). Good for probabilities. https://en.wikipedia.org/wiki/Sigmoid_function
    - Tanh: maps to (-1,1). Centered around 0 which can speed learning a bit. https://en.wikipedia.org/wiki/Hyperbolic_function
    - ReLU: returns 0 for negative inputs and identity for positives; simple and effective. https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
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
        if self.activation == 'sigmoid':
            # Sigmoid: 1 / (1 + e^-z). We clip z to avoid overflow in exp for very large magnitudes.
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clipping is a numeric safety measure
        elif self.activation == 'tanh':
            return np.tanh(z)  # hyperbolic tangent: outputs between -1 and 1
        elif self.activation == 'relu':
            return np.maximum(0, z)  # ReLU: zero for negatives, identity for positives
        else:
            return z  # linear activation: return z unchanged (useful for regression or output before softmax)

    def activate_derivative(self, z):
        # Return derivative of the activation function evaluated at z; used in backpropagation.
        if self.activation == 'sigmoid':
            s = self.activate(z)  # sigmoid(z)
            # derivative of sigmoid is s * (1 - s)
            return s * (1 - s)  # efficient reuse of s
        elif self.activation == 'tanh':
            # derivative of tanh is 1 - tanh(z)^2
            return 1 - np.tanh(z) ** 2
        elif self.activation == 'relu':
            # derivative of ReLU is 1 where z>0 else 0. We cast to float for numeric operations.
            return (z > 0).astype(float)
        else:
            return 1  # derivative of linear activation is 1

    def forward(self, inputs):
        # Compute the neuron's forward pass for a single input vector.
        self.input = inputs  # save input for gradient computation in backward pass
        self.z = np.dot(self.weights, inputs) + self.bias  # linear combination w.x + b
        self.output = self.activate(self.z)  # apply activation function
        return self.output  # return scalar output for this neuron


class Layer:
    """Layer of neurons: container that holds multiple Neuron objects and computes their outputs.

    Notes:
    - Each Layer is responsible for creating `n_neurons` Neuron objects.
    - The `forward` method applies each neuron to the same input vector and returns a 1D numpy array
      with the outputs of all neurons in this layer.
    """

    def __init__(self, n_inputs, n_neurons, activation='sigmoid'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_neurons)]
        # create a list of Neuron objects; each neuron expects `n_inputs` features
        self.n_neurons = n_neurons  # store number of neurons for convenience
        self.output = None  # will hold the output array after forward pass

    def forward(self, inputs):
        # Compute forward pass for the whole layer: gather outputs from each neuron
        self.output = np.array([neuron.forward(inputs) for neuron in self.neurons])
        # The result is a 1D array of length `n_neurons` containing each neuron's output.
        return self.output


class NeuralNetwork:
    """Feedforward neural network built from Layer and Neuron objects.

    Structure and usage:
    - `layer_sizes` is a list like [784, 128, 64, 10] meaning input size 784, two hidden layers,
      and output size 10.
    - `activations` is a list of activation names for each hidden/output layer (length = n_layers).

    Training internals provided here are simple and educational (not optimized like modern frameworks).
    """

    def __init__(self, layer_sizes, activations=None):
        self.layer_sizes = layer_sizes  # keep the architecture sizes
        self.n_layers = len(layer_sizes) - 1  # number of layers with parameters (excluding input layer)

        if activations is None:
            activations = ['sigmoid'] * self.n_layers  # default: sigmoid for every layer

        self.layers = []  # list to hold constructed Layer objects
        for i in range(self.n_layers):
            layer = Layer(layer_sizes[i], layer_sizes[i+1], activations[i])
            # create each layer using size of previous layer as n_inputs
            self.layers.append(layer)

        # track training metrics for plotting later
        self.training_loss = []
        self.training_accuracy = []

    def forward(self, inputs):
        # Compute the network's forward pass through all layers.
        output = inputs  # start with raw input vector
        for layer in self.layers:
            output = layer.forward(output)  # feed output of previous layer into next
        return output  # final output is e.g., a length-10 vector for MNIST

    def backward(self, inputs, targets, learning_rate=0.1):
        # Single-sample backward pass using mean-squared error derivative.
        # NOTE: This is a pedagogical implementation of backprop and uses element-wise formulas.

        self.forward(inputs)  # run forward to ensure .output, .z saved on neurons

        output_layer = self.layers[-1]  # reference to last layer (output layer)
        errors = [None] * self.n_layers  # placeholder for error signals per layer

        # Compute errors for output layer neurons
        output_error = []
        for i, neuron in enumerate(output_layer.neurons):
            # (neuron.output - target) is gradient of MSE w.r.t. neuron's output
            # multiply by derivative of activation (chain rule)
            error = (neuron.output - targets[i]) * neuron.activate_derivative(neuron.z)
            output_error.append(error)
        errors[-1] = np.array(output_error)  # store array of errors for output layer

        # Backpropagate errors to earlier layers
        for l in range(self.n_layers - 2, -1, -1):
            # iterate layers from second-last down to first parameterized layer
            layer_error = []
            for i, neuron in enumerate(self.layers[l].neurons):
                # compute error contribution to neuron i in layer l from all neurons in l+1
                error = 0
                for j, next_neuron in enumerate(self.layers[l+1].neurons):
                    # next_neuron.weights[i] is weight connecting neuron i (current layer) to neuron j (next layer)
                    error += errors[l+1][j] * next_neuron.weights[i]
                # multiply aggregated error by derivative of activation at neuron's z
                error *= neuron.activate_derivative(neuron.z)
                layer_error.append(error)
            errors[l] = np.array(layer_error)  # store error vector for layer l

        # Update weights and biases using the computed errors and the saved inputs/outputs
        for l, layer in enumerate(self.layers):
            # Determine input to this layer: the raw network inputs for the first layer,
            # otherwise the output of previous layer.
            layer_input = inputs if l == 0 else self.layers[l-1].output
            for i, neuron in enumerate(layer.neurons):
                # Gradient for weights: error * input (note inputs is a vector)
                # Here we rely on numpy broadcasting: neuron.weights and layer_input have same length
                neuron.weights -= learning_rate * errors[l][i] * layer_input
                # Bias gradient is just the error term (since d/d b of (w.x + b) is 1)
                neuron.bias -= learning_rate * errors[l][i]

    def train_batch(self, X, y, epochs=10, learning_rate=0.1, batch_size=32, 
                   X_val=None, y_val=None, verbose=True):
        """Train with mini-batch gradient descent.

        Explanation and notes:
        - This method implements basic mini-batch gradient descent: the dataset is shuffled and
          processed in chunks (batches). Inside each batch it performs a full backward pass for each sample.
        - This is NOT vectorized for speed; it's written for clarity and teaching.
        - For larger datasets, frameworks use vectorized operations and compute gradient averages per batch.

        Helpful link about mini-batch gradient descent (simple):
        https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html
        """
        n_samples = len(X)  # number of training examples

        for epoch in range(epochs):
            start_time = time.time()  # time epoch duration

            # Shuffle data each epoch to reduce bias and improve convergence
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_loss = 0  # accumulate loss across all samples to compute average later
            correct = 0  # count correct predictions for accuracy

            # Mini-batch training loop
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]  # slice a batch of inputs
                batch_y = y_shuffled[i:i+batch_size]  # corresponding batch of labels

                for j in range(len(batch_X)):
                    # Update network parameters using one sample from the batch
                    self.backward(batch_X[j], batch_y[j], learning_rate)

                    # After update, compute output to evaluate loss and accuracy tracking
                    output = self.forward(batch_X[j])
                    loss = np.mean((output - batch_y[j]) ** 2)  # mean squared error of this sample
                    total_loss += loss  # add to running total loss

                    # If the index with max output equals index with max target, it's a correct classification
                    if np.argmax(output) == np.argmax(batch_y[j]):
                        correct += 1

            avg_loss = total_loss / n_samples  # average loss per sample in epoch
            train_acc = correct / n_samples * 100  # training accuracy as percentage

            # Save metrics for plotting later
            self.training_loss.append(avg_loss)
            self.training_accuracy.append(train_acc)

            # Validation accuracy if validation set provided
            val_acc = 0
            if X_val is not None and y_val is not None:
                val_acc = self.evaluate(X_val, y_val)

            epoch_time = time.time() - start_time  # time taken for epoch

            if verbose:
                if X_val is not None:
                    # print training progress with validation accuracy included
                    print(f"Epoch {epoch+1:2d}/{epochs} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Train Acc: {train_acc:.2f}% | "
                          f"Val Acc: {val_acc:.2f}% | "
                          f"Time: {epoch_time:.1f}s")
                else:
                    # print training progress without validation accuracy
                    print(f"Epoch {epoch+1:2d}/{epochs} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"Train Acc: {train_acc:.2f}% | "
                          f"Time: {epoch_time:.1f}s")

    def predict(self, X):
        # Run forward pass for each input sample and collect raw outputs
        predictions = []
        for inputs in X:
            output = self.forward(inputs)
            predictions.append(output)
        return np.array(predictions)  # return 2D array shape (n_samples, n_outputs)

    def evaluate(self, X, y):
        """Evaluate accuracy on dataset.

        This uses predict() and a simple argmax comparison to compute percentage correct.
        """
        correct = 0
        predictions = self.predict(X)  # raw network outputs for each sample

        for i in range(len(X)):
            if np.argmax(predictions[i]) == np.argmax(y[i]):
                correct += 1

        return correct / len(X) * 100  # percent accuracy


def load_and_preprocess_mnist(n_train=5000, n_test=1000):
    """Load and preprocess MNIST dataset.

    Steps performed:
    1. Load using Keras helper (downloads dataset the first time).
    2. Optionally subset to `n_train` / `n_test` for faster experiments.
    3. Flatten 28x28 images to 1D arrays of length 784. Explanation: neural nets here expect vectors.
       Simple link explaining flattening/why: https://www.khanacademy.org/math/precalculus
       (flattening is simply listing pixels row-by-row into one long vector; no advanced math needed).
    4. Normalize pixel values from [0,255] to [0,1] for numerical stability.
    5. One-hot encode labels: transform a scalar label like 3 into a vector with a 1 at index 3.
       Simple one-hot explanation: https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
    """
    print("Loading MNIST dataset...")  # notify user about dataset download/load
    (X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.mnist.load_data()
    # keras.datasets.mnist comes with 60k training and 10k test images by default

    # Use subset for faster training (user-specified sizes)
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_test = X_test_full[:n_test]
    y_test = y_test_full[:n_test]

    # Flatten images: 28x28 -> 784 (row-major order). This converts 2D images into 1D feature vectors.
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Normalize: convert integers 0-255 to floats 0.0-1.0 to help optimization convergence
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # One-hot encode labels into vectors of length 10 (for digits 0-9)
    y_train_onehot = np.zeros((len(y_train), 10))
    y_test_onehot = np.zeros((len(y_test), 10))

    for i in range(len(y_train)):
        # set the column corresponding to the digit label to 1
        y_train_onehot[i, y_train[i]] = 1

    for i in range(len(y_test)):
        y_test_onehot[i, y_test[i]] = 1

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Image size: {X_train.shape[1]} pixels (28x28 flattened)")

    # return processed arrays plus original scalar test labels for display convenience
    return X_train, y_train_onehot, X_test, y_test_onehot, y_test


def visualize_predictions(nn, X_test, y_test_labels, n_samples=20):
    """Visualize some predictions and save a PNG.

    - Selects `n_samples` random examples from X_test.
    - Plots the image, predicted label, true label, and the model's confidence.
    - Confidence here is the raw output for the predicted class (not a calibrated probability).

    Note on confidence: this network outputs values that are not passed through a softmax,
    so treating the highest output as a probability is only an approximation. For proper
    calibrated probabilities, apply a softmax to the final outputs. Simple softmax intro:
    https://www.khanacademy.org/math/statistics-probability/probability-library
    (softmax is a way to convert raw scores into positive numbers that sum to 1).
    """
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))  # create a grid of subplots 4x5
    axes = axes.ravel()  # flatten axes array to iterate easily

    indices = np.random.choice(len(X_test), n_samples, replace=False)  # sample unique indices

    for i, idx in enumerate(indices):
        # Reshape flattened vector back to 28x28 for display as image
        img = X_test[idx].reshape(28, 28)

        # Get prediction vector and derive predicted label
        output = nn.forward(X_test[idx])
        pred_label = np.argmax(output)  # index of maximum output component
        true_label = y_test_labels[idx]  # true scalar label from original dataset
        confidence = output[pred_label] * 100  # naive 'confidence' as percentage of raw output

        # Plot image
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')  # hide axis ticks

        # color title green for correct, red for incorrect
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)', 
                         color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig('mnist_predictions.png', dpi=150, bbox_inches='tight')  # save the figure to file
    print("\nPredictions saved to 'mnist_predictions.png'")
    plt.show()  # display the figure in an interactive environment


def plot_training_history(nn):
    """Plot training loss and accuracy saved during train_batch."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(nn.training_loss)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)  # add light grid to make trends easier to read

    # Accuracy plot
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
    """Main function to run MNIST classification end-to-end.

    Steps:
    1. Load and preprocess data
    2. Create network architecture
    3. Train with train_batch
    4. Evaluate and show per-class accuracy
    5. Plot and save visualizations
    """
    print("=" * 70)
    print("MNIST Digit Classification with OOP Neural Network")
    print("=" * 70)

    # Load data (subsampled for speed by default)
    X_train, y_train, X_test, y_test, y_test_labels = load_and_preprocess_mnist(
        n_train=5000, n_test=1000
    )

    # Create neural network: 784 -> 128 -> 64 -> 10
    print("\nCreating Neural Network...")
    print("Architecture: 784 (input) -> 128 (hidden) -> 64 (hidden) -> 10 (output)")

    nn = NeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        activations=['relu', 'relu', 'sigmoid']
    )

    # Count total parameters: for each neuron count its weights + bias
    total_params = sum(len(n.weights) + 1 for l in nn.layers for n in l.neurons)
    print(f"Total parameters: {total_params:,}")

    # Train the network with specified hyperparameters
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

    # Final evaluation on training and test sets
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)

    train_acc = nn.evaluate(X_train, y_train)
    test_acc = nn.evaluate(X_test, y_test)

    print(f"Final Training Accuracy: {train_acc:.2f}%")
    print(f"Final Test Accuracy: {test_acc:.2f}%")

    # Per-class accuracy breakdown
    print("\n" + "=" * 70)
    print("Per-Class Accuracy")
    print("=" * 70)

    predictions = nn.predict(X_test)
    pred_labels = np.argmax(predictions, axis=1)

    for digit in range(10):
        digit_mask = y_test_labels == digit  # mask for samples of this digit
        digit_acc = np.mean(pred_labels[digit_mask] == digit) * 100  # percent correct for this digit
        digit_count = np.sum(digit_mask)
        print(f"Digit {digit}: {digit_acc:.2f}% ({digit_count} samples)")

    # Visualizations
    print("\n" + "=" * 70)
    print("Generating Visualizations...")
    print("=" * 70)

    plot_training_history(nn)  # loss and accuracy plots
    visualize_predictions(nn, X_test, y_test_labels)  # sample image predictions

    print("\n" + "=" * 70)
    print("MNIST Classification Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
