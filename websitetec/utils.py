from keras.datasets import mnist
import os
import streamlit as st

def load_mnist_data():
    _, (X_test, y_test) = mnist.load_data()
    return X_test, y_test

def load_explanations(filename):
    project_root = os.path.dirname(os.path.abspath(__file__))
    explanations_md_path = os.path.join(project_root, filename)
    
    with open(explanations_md_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_css(filename):
    project_root = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(project_root, filename)
    
    with open(css_path, 'r', encoding='utf-8') as file:
        st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)
