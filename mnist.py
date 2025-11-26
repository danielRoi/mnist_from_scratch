from nn import NeuralNetwork
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

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

    nn.train(
        X_train, y_train,
        epochs=15,
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