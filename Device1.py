#plots
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import re
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt  # Import Matplotlib for plotting

# Define the mapping from 4-bit sequences to current values with logarithmic scaling
n = 0  # Systematic error percentage as standard deviation

# Define a function to apply multiplicative random error then log transform
def apply_error_then_log(current, error_percentage):
    error_factor = 1 + np.random.normal(0, error_percentage / 100)
    return np.log(current * error_factor)

# Current values before logging
base_current_map = {
    '1100': 1.23e-06,
    '1011': 5.75e-06,
    '0100': 4.23e-07,
    '0110': 2.48e-06,
    '0111': 5.85e-06,
    '0011': 4.59e-06,
    '1101': 5.51e-06,
    '1110': 3.13e-06,
    '0101': 4.41e-06,
    '0000': 8.10e-08,
    '1111': 9.89e-06,
    '1010': 2.10e-06,
    '1001': 4.36e-06,
    '1000': 2.72e-07,
    '0010': 6.90e-07,
    '0001': 2.15e-06
}


def read_and_transform_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    currents = [apply_error_then_log(base_current_map[line.strip()], n)
                for line in lines if line.strip() in base_current_map]
    return currents

def load_data(directory):
    features = []
    labels = []
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory, file_name)
            currents = read_and_transform_data(file_path)
            label = int(re.search(r'label_(\d+)_', file_name).group(1))
            features.append(currents)
            labels.append(label)
    return np.array(features), np.array(labels)

X, y = load_data(r'C:\Users\97843\Desktop\MNIST\Processed_50')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = Sequential([
    Input(shape=(196,)),
    Dense(10, activation='softmax', kernel_regularizer=l2(0.001))
])

model.compile(optimizer=Adam(learning_rate=6e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model and store the history
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
