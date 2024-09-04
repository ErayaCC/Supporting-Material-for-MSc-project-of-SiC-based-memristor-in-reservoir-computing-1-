import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import re
import matplotlib.pyplot as plt

# Define the mapping from 4-bit sequences to current values
current_map = {
    '0000': 9.74e-08, '1000': 3.23e-07, '0100': 4.96e-07, '0010': 8.36e-07,
    '1100': 1.68e-06, '1010': 2.36e-06, '0001': 2.74e-06, '0110': 2.81e-06,
    '1110': 3.74e-06, '1001': 5.42e-06, '0101': 5.44e-06, '0011': 5.74e-06,
    '1101': 6.09e-06, '1011': 6.25e-06, '0111': 6.72e-06, '1111': 9.32e-06
}

def read_and_transform_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    currents = [current_map[line.strip()] for line in lines if line.strip() in current_map]
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

# Load your data
X, y = load_data(r'C:\Users\97843\Desktop\MNIST\Processed_20')

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the model with an explicit Input layer
model = Sequential([
    Input(shape=(196,)),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")

# Plotting the training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
