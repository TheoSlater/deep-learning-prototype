import tensorflow as tf
import numpy as np
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.utils import to_categorical
import pickle
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_LENGTH = 20

# Global variables
tokenizer = None
model = None
NUM_CLASSES = None

def initialize_resources():
    global tokenizer, model, NUM_CLASSES

    tokenizer = Tokenizer(oov_token='<OOV>')
    NUM_CLASSES = 1000  # Initial value, adjust according to the vocabulary size

    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=(None, MAX_LENGTH))
    model.summary()

def build_model():
    global NUM_CLASSES
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=NUM_CLASSES, output_dim=64, input_length=MAX_LENGTH),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    ])

def save_model(filepath):
    model.save(filepath + '.keras')
    logger.info(f"Model saved to {filepath}.keras")

def load_model(filepath):
    global model
    model = tf.keras.models.load_model(filepath + '.keras')
    logger.info(f"Model loaded from {filepath}.keras")

def save_tokenizer(filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(tokenizer, file)
    logger.info(f"Tokenizer saved to {filepath}")

def load_tokenizer(filepath):
    global tokenizer
    with open(filepath, 'rb') as file:
        tokenizer = pickle.load(file)
    logger.info(f"Tokenizer loaded from {filepath}")

def preprocess_text(text):
    if not text:
        logger.error("Empty or None text provided")
        return np.zeros((1, MAX_LENGTH))  # Return a zero-padded sequence

    sequences = tokenizer.texts_to_sequences([text])
    logger.info(f"Sequences before padding: {sequences}")

    if sequences is None or not isinstance(sequences, list) or not sequences[0]:
        logger.error("Tokenization resulted in None or invalid format")
        return np.zeros((1, MAX_LENGTH))  # Return a zero-padded sequence

    sequences = sequences[0]

    # Replace None values with 0 (or another default token index)
    sequences = [0 if v is None else v for v in sequences]
    logger.info(f"Sequences after replacing None values: {sequences}")

    if not sequences or not all(isinstance(i, int) for i in sequences):
        logger.error("Sequences are empty or contain non-integer values")
        return np.zeros((1, MAX_LENGTH))  # Return a zero-padded sequence

    padded_sequences = pad_sequences([sequences], maxlen=MAX_LENGTH, padding='post')
    return padded_sequences

def postprocess_text(tokens):
    index_word = tokenizer.index_word
    return ' '.join([index_word.get(token, '') for token in tokens if token > 0])

def train_model(input_text, target_text):
    input_sequence = preprocess_text(input_text)
    target_sequence = preprocess_text(target_text)

    target_sequence_one_hot = to_categorical(target_sequence, num_classes=NUM_CLASSES)
    model.fit(input_sequence, target_sequence_one_hot, epochs=150 , verbose=1)

def generate_response(input_text):
    input_sequence = preprocess_text(input_text)
    prediction = model.predict(input_sequence)
    response_sequence = np.argmax(prediction[0], axis=-1)
    return postprocess_text(response_sequence)

def add_word_to_tokenizer(new_word):
    global tokenizer, NUM_CLASSES, model

    # Update tokenizer with the new word
    tokenizer.fit_on_texts([new_word])
    NUM_CLASSES = len(tokenizer.word_index) + 1  # +1 to account for padding token

    # Rebuild model with updated NUM_CLASSES
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=(None, MAX_LENGTH))
    logger.info(f"Model rebuilt with NUM_CLASSES: {NUM_CLASSES}")

def main():
    initialize_resources()
    
    model_filepath = 'model'
    tokenizer_filepath = 'tokenizer.pkl'
    
    if os.path.exists(model_filepath + '.keras'):
        load_model(model_filepath)
    if os.path.exists(tokenizer_filepath):
        load_tokenizer(tokenizer_filepath)
    else:
        logger.info("No pre-trained tokenizer found, starting with a new tokenizer.")
    
    while True:
        user_text = input("You: ")
        if user_text.lower() == 'exit':
            break

        response = generate_response(user_text)
        print(f"Bot: {response}")

        if response.strip() == "":
            expected_response = input("Correct response: ")
            # Add new word(s) to the tokenizer
            add_word_to_tokenizer(user_text)
            add_word_to_tokenizer(expected_response)
            # Re-train the model with the new words
            train_model(user_text, expected_response)
            save_model(model_filepath)
            save_tokenizer(tokenizer_filepath)

if __name__ == "__main__":
    main()
