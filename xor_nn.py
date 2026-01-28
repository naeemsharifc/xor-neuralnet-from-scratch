import numpy as np
import matplotlib.pyplot as plt

# Set random seed for consistent results on each program run
np.random.seed(42)

class XORNeuralNet:
    def __init__(self):
        # Initialize weights and biases
        
        self.Wxh = np.random.randn(2, 5)
        self.bh = np.zeros((1, 5))
        self.Why = np.random.randn(5, 1)
        self.by = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def train(self, X, y, iterations=10000, alpha=0.1):
        cost_history = []
        for i in range(iterations):
            # Forward Prop
            z1 = np.dot(X, self.Wxh) + self.bh
            a1 = self.sigmoid(z1)
            z2 = np.dot(a1, self.Why) + self.by
            y_hat = self.sigmoid(z2)

            # Compute Cost (Mean Squared Error)
            cost = 0.5 * np.sum((y - y_hat)**2)
            cost_history.append(cost)

            # Backward Prop
            delta2 = (y_hat - y) * self.sigmoid_derivative(z2)
            delta1 = np.dot(delta2, self.Why.T) * self.sigmoid_derivative(z1)

            # Updates (weights and biases)
            self.Why -= alpha * np.dot(a1.T, delta2)
            self.by  -= alpha * np.sum(delta2, axis=0, keepdims=True)
            self.Wxh -= alpha * np.dot(X.T, delta1)
            self.bh  -= alpha * np.sum(delta1, axis=0, keepdims=True)
            
        return cost_history # Return history for plotting

    def predict(self, x1, x2):
        input_data = np.array([[x1, x2]])
        z1 = np.dot(input_data, self.Wxh) + self.bh
        a1 = self.sigmoid(z1)
        z2 = np.dot(a1, self.Why) + self.by
        y_hat = self.sigmoid(z2)
        return y_hat[0][0]

# --- Visualization of Convergence ---

def plot_convergence(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history, color='royalblue', linewidth=2)
    plt.title('Model Convergence: Cost vs. Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# --- Execution ---

# 1. Define Training Data
X_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([[0],   [1],   [1],   [0]])

# 2. Create and Train the Model
nn = XORNeuralNet()
print("Training the neural network...")
history = nn.train(X_train, y_train)
print("Training complete!\n")

# 3. Plot the convergence results
plot_convergence(history)

# 4. Custom Input by the User
print("\n--- XOR Predictor ---")
try:
    user_x1 = int(input("Enter x1 (0 or 1): "))
    user_x2 = int(input("Enter x2 (0 or 1): "))
    
    prediction = nn.predict(user_x1, user_x2)
    
    print(f"\nRaw NN Output: {prediction:.4f}")
    print(f"Final Decision (Rounded): {round(prediction)}")
except ValueError:
    print("Please enter valid integers (0 or 1).")