from keras.datasets import mnist

def load_mnist_data():
    _, (X_test, y_test) = mnist.load_data()
    return X_test, y_test

def load_explanations(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
