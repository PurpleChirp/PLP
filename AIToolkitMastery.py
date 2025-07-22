# Task 1: Classical ML with Scikit-learn

from textblob import TextBlob
import spacy
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Load and prepare the dataset
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Encode labels (already numeric but shown for clarity)
data['target'] = LabelEncoder().fit_transform(data['target'])

# Simulate missing values and handle them
data.iloc[0, 0] = np.nan
data.fillna(data.mean(), inplace=True)

# Train/Test Split
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict and Evaluate
y_pred = clf.predict(X_test)
print("Iris Classifier Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='macro'))
print("Recall:", recall_score(y_test, y_pred, average='macro'))


# Deep Learning with TensorFlow


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print("MNIST Test Accuracy:", test_acc)

# Visualize 5 predictions
predictions = model.predict(x_test[:5])
for i in range(5):
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {np.argmax(predictions[i])}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.show()


# NLP with spaCy


nlp = spacy.load("en_core_web_sm")

# Sample Amazon review
review = "The new Sony WH-1000XM5 headphones have amazing sound quality and the battery life is unbeatable!"

# Named Entity Recognition
doc = nlp(review)
print("Named Entities:")
for ent in doc.ents:
    print(ent.text, "->", ent.label_)

# Rule-based Sentiment (using TextBlob for simplicity)
sentiment = TextBlob(review).sentiment.polarity
print("Sentiment:", "Positive" if sentiment >
      0 else "Negative" if sentiment < 0 else "Neutral")


# =============== Part 3: Ethics & Troubleshooting ===============

# Ethical Reflection
print("\nEthical Concern Example:")
print("Potential bias in MNIST if trained only on balanced digits; Amazon reviews may show brand or product sentiment bias.")
print("To mitigate this, use tools like TensorFlow Fairness Indicators or rule-based analysis in spaCy to flag bias-prone samples.")

# Bug Fixing Example (hypothetical):
# Before: model.compile(loss='mse', metrics=['accuracy']) -> Incorrect for classification
# After:
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Streamlit Web App
# # import streamlit as st
# st.title("MNIST Digit Classifier")
# uploaded_image = st.file_uploader("Upload a digit image")
# if uploaded_image:
#     prediction = model.predict(processed_image)
#     st.write(f"Predicted Digit: {np.argmax(prediction)}")
