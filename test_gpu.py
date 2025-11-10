import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU detected: {len(gpus)} GPU(s) available")
    for gpu in gpus:
        print("   •", gpu)
else:
    print("No GPU detected.")