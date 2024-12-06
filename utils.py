from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def load_data():
    # Load and prepare the dataset (digits dataset as an example)
    data = load_digits()
    X = data.data
    y = data.target

    # One-hot encoding the target labels
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, X_test, y_train, y_test

def accuracy(X, y, model):
    y_pred = model.forward(X)
    y_pred_class = np.argmax(y_pred, axis=1)
    y_true_class = np.argmax(y, axis=1)
    return np.mean(y_pred_class == y_true_class)
