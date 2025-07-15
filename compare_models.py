import tensorflow as tf
from accuracy import evaluate_models
import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy_comparison():
    print("Starting model evaluation...")
    evaluate_models()
    
if __name__ == "__main__":
    # Enable memory growth to avoid GPU memory issues
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        print("No GPU devices found. Running on CPU.")
    
    plot_accuracy_comparison() 