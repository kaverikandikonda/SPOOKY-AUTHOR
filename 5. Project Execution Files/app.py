from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd

# Load your pre-trained model and tokenizer 
model = tf.keras.models.load_model("author_prediction_model.h5")
tokenizer = tf.keras.preprocessing.text.Tokenizer()

app = Flask(__name__)

# Define route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for predict page
@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from user
    input_text = request.form['text']

    # Preprocess input text (adjust as needed)
    input_text = input_text.lower()  # Convert to lowercase
    input_text = input_text.strip()  # Remove leading/trailing whitespace

    # Convert input text to sequence of integer IDs
    input_text_id = tokenizer.texts_to_sequences([input_text])

    # Pad input sequence with zeros
    input_text_id_padded = tf.keras.preprocessing.sequence.pad_sequences(input_text_id, maxlen=300)

    # Make prediction using the model
    y_pred = model.predict(np.asarray(input_text_id_padded))

    # Map predicted probabilities to author names (replace with your actual mapping)
    author_mapping = {0: 'Edgar Allan Poe', 1: 'HP Lovecraft', 2: 'Mary Shelley'}
    y_pred_author = author_mapping[y_pred.argmax()]

    # Return result to author page
    return render_template('author.html', input_text=input_text, author=y_pred_author)

if __name__ == '_main_':
    app.run(debug=True)