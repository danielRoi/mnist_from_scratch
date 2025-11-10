"""
Interactive MNIST Digit Drawing and Recognition Application
Draw digits and get real-time predictions from the trained neural network
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import cv2

# Import from the optimized MNIST file
from faster_mnist import OptimizedNeuralNetwork, load_and_preprocess_mnist


class DigitRecognizerApp:
    """Interactive digit drawing and recognition application"""
    
    def __init__(self, model):
        self.model = model
        self.window = tk.Tk()
        self.window.title("MNIST Digit Recognizer - Draw and Predict")
        self.window.geometry("900x600")
        self.window.configure(bg='#2b2b2b')
        
        # Drawing canvas size (larger for easier drawing)
        self.canvas_size = 280  # 10x the MNIST 28x28
        self.mnist_size = 28
        
        # Create PIL image for drawing (white background)
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Setup UI
        self.setup_ui()
        
        # Mouse state
        self.last_x = None
        self.last_y = None
        self.is_drawing = False
        
    def setup_ui(self):
        """Create the user interface"""
        # Main container
        main_frame = tk.Frame(self.window, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left side - Drawing canvas
        left_frame = tk.Frame(main_frame, bg='#2b2b2b')
        left_frame.pack(side=tk.LEFT, padx=10)
        
        # Title
        title_label = tk.Label(
            left_frame, 
            text="Draw a Digit (0-9)", 
            font=('Arial', 16, 'bold'),
            bg='#2b2b2b',
            fg='white'
        )
        title_label.pack(pady=(0, 10))
        
        # Canvas frame with border
        canvas_frame = tk.Frame(left_frame, bg='white', bd=2, relief=tk.SOLID)
        canvas_frame.pack()
        
        # Drawing canvas
        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg='white',
            cursor='pencil'
        )
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        # Buttons
        button_frame = tk.Frame(left_frame, bg='#2b2b2b')
        button_frame.pack(pady=15)
        
        clear_btn = tk.Button(
            button_frame,
            text="Clear Canvas",
            command=self.clear_canvas,
            font=('Arial', 12),
            bg='#ff4444',
            fg='white',
            width=12,
            height=2,
            cursor='hand2'
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        predict_btn = tk.Button(
            button_frame,
            text="Predict Now",
            command=self.predict,
            font=('Arial', 12),
            bg='#4444ff',
            fg='white',
            width=12,
            height=2,
            cursor='hand2'
        )
        predict_btn.pack(side=tk.LEFT, padx=5)
        
        # Right side - Predictions and visualization
        right_frame = tk.Frame(main_frame, bg='#2b2b2b')
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Prediction display
        pred_label = tk.Label(
            right_frame,
            text="Prediction",
            font=('Arial', 16, 'bold'),
            bg='#2b2b2b',
            fg='white'
        )
        pred_label.pack(pady=(0, 10))
        
        self.prediction_label = tk.Label(
            right_frame,
            text="Draw a digit",
            font=('Arial', 48, 'bold'),
            bg='#3b3b3b',
            fg='#00ff00',
            width=8,
            height=2,
            relief=tk.RIDGE,
            bd=3
        )
        self.prediction_label.pack(pady=10)
        
        # Confidence label
        self.confidence_label = tk.Label(
            right_frame,
            text="Confidence: --",
            font=('Arial', 14),
            bg='#2b2b2b',
            fg='white'
        )
        self.confidence_label.pack(pady=5)
        
        # Matplotlib figure for probability distribution
        self.fig, self.ax = plt.subplots(figsize=(5, 3), facecolor='#2b2b2b')
        self.ax.set_facecolor('#3b3b3b')
        self.ax.set_xlabel('Digit', color='white', fontsize=12)
        self.ax.set_ylabel('Probability (%)', color='white', fontsize=12)
        self.ax.set_title('Prediction Probabilities', color='white', fontsize=14, pad=10)
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        # Embed matplotlib figure
        canvas_widget = FigureCanvasTkAgg(self.fig, master=right_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(pady=20)
        
        # Auto-predict checkbox
        self.auto_predict_var = tk.BooleanVar(value=True)
        auto_check = tk.Checkbutton(
            right_frame,
            text="Auto-predict while drawing",
            variable=self.auto_predict_var,
            font=('Arial', 11),
            bg='#2b2b2b',
            fg='white',
            selectcolor='#3b3b3b',
            activebackground='#2b2b2b',
            activeforeground='white'
        )
        auto_check.pack(pady=10)
        
        # Initialize empty bar chart
        self.update_probability_chart([0]*10)
        
    def start_draw(self, event):
        """Start drawing"""
        self.is_drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_line(self, event):
        """Draw on canvas"""
        if self.is_drawing:
            x, y = event.x, event.y
            
            # Draw on tkinter canvas (black line)
            if self.last_x and self.last_y:
                self.canvas.create_line(
                    self.last_x, self.last_y, x, y,
                    width=20, fill='black',
                    capstyle=tk.ROUND, smooth=True
                )
            
            # Draw on PIL image (black on white)
            if self.last_x and self.last_y:
                self.draw.line(
                    [self.last_x, self.last_y, x, y],
                    fill=0, width=20
                )
            
            self.last_x = x
            self.last_y = y
            
            # Auto-predict if enabled
            if self.auto_predict_var.get():
                self.predict()
    
    def stop_draw(self, event):
        """Stop drawing"""
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        
        # Final prediction
        if self.auto_predict_var.get():
            self.predict()
    
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 255)
        self.draw = ImageDraw.Draw(self.image)
        
        self.prediction_label.config(text="Draw a digit", fg='#00ff00')
        self.confidence_label.config(text="Confidence: --")
        self.update_probability_chart([0]*10)
    
    def preprocess_image(self):
        """Convert drawn image to MNIST format (28x28, inverted colors, centered)"""
        # Convert PIL image to numpy array
        img_array = np.array(self.image)
        
        # Check if image is empty
        if np.mean(img_array) > 250:
            return None
        
        # Invert colors (MNIST has white digits on black background)
        img_array = 255 - img_array
        
        # Resize to 28x28 using proper anti-aliasing
        img_resized = cv2.resize(img_array, (self.mnist_size, self.mnist_size), 
                                interpolation=cv2.INTER_AREA)
        
        # Find bounding box of the digit
        rows = np.any(img_resized > 30, axis=1)
        cols = np.any(img_resized > 30, axis=0)
        
        if not rows.any() or not cols.any():
            return None
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Extract digit with some padding
        digit = img_resized[rmin:rmax+1, cmin:cmax+1]
        
        # Calculate aspect ratio preserving resize
        digit_h, digit_w = digit.shape
        
        # Fit into 20x20 box (leaving 4 pixels border like MNIST)
        max_dim = max(digit_h, digit_w)
        scale = 20.0 / max_dim
        new_h = int(digit_h * scale)
        new_w = int(digit_w * scale)
        
        digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center in 28x28 image
        result = np.zeros((self.mnist_size, self.mnist_size), dtype=np.float32)
        y_offset = (self.mnist_size - new_h) // 2
        x_offset = (self.mnist_size - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
        
        # Normalize to [0, 1]
        result = result.astype('float32') / 255.0
        
        # Flatten
        result = result.reshape(-1)
        
        return result
    
    def update_probability_chart(self, probabilities):
        """Update the bar chart with prediction probabilities"""
        self.ax.clear()
        
        digits = list(range(10))
        colors = ['#00ff00' if p == max(probabilities) and p > 0 else '#4444ff' 
                  for p in probabilities]
        
        bars = self.ax.bar(digits, np.array(probabilities) * 100, color=colors, alpha=0.8)
        
        self.ax.set_xlabel('Digit', color='white', fontsize=12)
        self.ax.set_ylabel('Probability (%)', color='white', fontsize=12)
        self.ax.set_title('Prediction Probabilities', color='white', fontsize=14, pad=10)
        self.ax.set_xticks(digits)
        self.ax.set_ylim([0, 105])
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['left'].set_color('white')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.grid(axis='y', alpha=0.3, color='white')
        
        self.fig.canvas.draw()
    
    def softmax(self, x):
        """Apply softmax to convert outputs to probabilities"""
        # Clip to prevent overflow
        x_clipped = np.clip(x, -500, 500)
        exp_x = np.exp(x_clipped - np.max(x_clipped))
        return exp_x / np.sum(exp_x)
    
    def predict(self):
        """Predict the drawn digit"""
        # Preprocess image
        processed_image = self.preprocess_image()
        
        if processed_image is None:
            self.prediction_label.config(text="Draw more", fg='#ffaa00')
            self.confidence_label.config(text="Confidence: --")
            self.update_probability_chart([0]*10)
            return
        
        # Get raw output (logits before final activation)
        output = self.model.forward(processed_image)
        raw_output = output.ravel()
        
        # Apply softmax to get proper probabilities
        probabilities = self.softmax(raw_output)
        
        predicted_digit = np.argmax(probabilities)
        confidence = probabilities[predicted_digit] * 100
        
        # Update UI
        self.prediction_label.config(
            text=str(predicted_digit),
            fg='#00ff00' if confidence > 50 else '#ffaa00'
        )
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Update chart
        self.update_probability_chart(probabilities)
    
    def run(self):
        """Start the application"""
        self.window.mainloop()


def train_model():
    """Train a neural network for digit recognition"""
    print("=" * 70)
    print("Training Neural Network for Digit Recognition")
    print("=" * 70)
    
    # Load data
    X_train, y_train, X_test, y_test, y_test_labels = load_and_preprocess_mnist(
        n_train=50000, n_test=10000
    )
    
    # Create network
    print("\nCreating Neural Network...")
    print("Architecture: 784 -> 128 -> 64 -> 10")
    
    nn = OptimizedNeuralNetwork(
        layer_sizes=[784, 128, 64, 10],
        activations=['relu', 'relu', 'relu']  # Use relu for output, apply softmax in prediction
    )
    
    # Train
    print("\nTraining...")
    nn.train_batch(
        X_train, y_train,
        epochs=15,
        learning_rate=0.1,
        batch_size=32,
        X_val=X_test,
        y_val=y_test,
        verbose=True
    )
    
    # Evaluate
    test_acc = nn.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    print("=" * 70)
    
    return nn


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("MNIST Interactive Digit Recognizer")
    print("=" * 70 + "\n")
    
    # Train the model
    model = train_model()
    
    # Launch interactive application
    print("\nLaunching interactive drawing application...")
    print("Instructions:")
    print("  - Draw a digit (0-9) on the canvas")
    print("  - The model will predict in real-time (if auto-predict is on)")
    print("  - Use 'Clear Canvas' to start over")
    print("  - Try to draw similar to MNIST style (white on black)")
    print("=" * 70 + "\n")
    
    app = DigitRecognizerApp(model)
    app.run()


if __name__ == "__main__":
    main()