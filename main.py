from model import NeuralNetwork
from layers import Dense
from activation import ReLU, Softmax
from utils import load_data, accuracy

def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Build a simple neural network
    nn = NeuralNetwork()
    nn.add(Dense(64, 128, activation=ReLU()))  # Input layer to hidden layer
    nn.add(Dense(128, 10, activation=Softmax()))  # Hidden layer to output layer

    # Train the neural network
    nn.train(X_train, y_train, epochs=50)

    # Evaluate the model
    test_accuracy = accuracy(X_test, y_test, nn)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
