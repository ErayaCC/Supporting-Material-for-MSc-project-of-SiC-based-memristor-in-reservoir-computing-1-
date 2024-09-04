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

n = 2  # Systematic error percentage as standard deviation

# Define a function to apply multiplicative random error
def apply_error(current, error_percentage):
    error_factor = 1 + np.random.normal(0, error_percentage / 100)
    return current * error_factor

base_current_map = {
    '0000': 9.74e-08, '1000': 3.23e-07, '0100': 4.96e-07, '0010': 8.36e-07,
    '1100': 1.68e-06, '1010': 2.36e-06, '0001': 2.74e-06, '0110': 2.81e-06,
    '1110': 3.74e-06, '1001': 5.42e-06, '0101': 5.44e-06, '0011': 5.74e-06,
    '1101': 6.09e-06, '1011': 6.25e-06, '0111': 6.72e-06, '1111': 9.32e-06
}

def read_and_transform_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    currents = [apply_error(base_current_map[line.strip()], n)
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

X, y = load_data(r'C:\Users\97843\Desktop\MNIST\Processed_20')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = Sequential([
    Input(shape=(196,)),
    Dense(10, activation='softmax', kernel_regularizer=l2(0.001))
])

model.compile(optimizer=Adam(learning_rate=2e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy * 100:.2f}%")
