from keras.datasets import mnist
import os

def load_mnist_data():
    _, (X_test, y_test) = mnist.load_data()
    return X_test, y_test

def load_explanations(filename):
    project_root = os.path.dirname(os.path.abspath(__file__))
    explanations_md_path = os.path.join(project_root, filename)
    
    with open(explanations_md_path, 'r', encoding='utf-8') as file:
        return file.read()
