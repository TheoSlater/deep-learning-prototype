import speech_recognition as sr
import tensorflow as tf
import numpy as np
from gtts import gTTS
import pygame
import time
from io import BytesIO
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer
import logging
import os
import pickle

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the recognizer
recognizer = sr.Recognizer()

# Global variables
tokenizer = None
model = None
max_length = 20
num_classes = 10000  # Ensure this matches the number of classes in your one-hot encoding

def initialize_resources():
    global tokenizer, model, max_length, num_classes

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=num_classes, output_dim=64, input_length=max_length),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))
    ])
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Build the model with a dummy input shape
    model.build(input_shape=(None, max_length))  # Specify input shape

    # Print model summary to check the architecture
    model.summary()

    tokenizer = Tokenizer(num_words=num_classes)
    max_length = 20

def save_model(filepath):
    model.save(filepath)
    logger.info(f"Model saved to {filepath}")

def load_model(filepath):
    global model
    model = tf.keras.models.load_model(filepath)
    logger.info(f"Model loaded from {filepath}")

def save_tokenizer(filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(tokenizer, file)
    logger.info(f"Tokenizer saved to {filepath}")

def load_tokenizer(filepath):
    global tokenizer
    with open(filepath, 'rb') as file:
        tokenizer = pickle.load(file)
    logger.info(f"Tokenizer loaded from {filepath}")

def proper_tokenizer(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    return padded_sequences[0]

def simple_detokenizer(tokens):
    if isinstance(tokens, np.ndarray):
        tokens = tokens.tolist()
    if not isinstance(tokens, list):
        tokens = [tokens]
    
    try:
        return ' '.join([tokenizer.index_word.get(t, '') for t in tokens if t > 0])
    except ValueError:
        logger.error("Error decoding tokens: %s", tokens)
        return ""

def listen():
    with sr.Microphone() as source:
        logger.info("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            logger.info(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            logger.warning("Could not understand audio")
            return ""
        except sr.RequestError:
            logger.error("Could not request results; check your network connection")
            return ""

def generate_response(input_tokens):
    input_tokens = np.array(input_tokens).reshape((1, -1))
    prediction = model.predict(input_tokens)
    logger.info(f"Raw prediction: {prediction}")

    # Check if prediction is giving non-zero values
    if np.all(prediction == 0):
        logger.warning("Prediction output is all zeros")
        return "I didn't understand that"

    # Ensure that prediction is properly handled
    response_tokens = np.argmax(prediction[0], axis=-1)
    logger.info(f"Raw response tokens: {response_tokens}")

    if not np.any(response_tokens):
        return "No valid response generated"

    response = simple_detokenizer(response_tokens)
    logger.info(f"Decoded response: {response}")

    return response



def text_to_speech(text):
    if not text.strip():
        print("No text to speak")
        return
    
    try:
        tts = gTTS(text=text, lang='en')
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)

        pygame.mixer.quit()
        pygame.mixer.init()
        
        pygame.mixer.music.stop()
        
        try:
            pygame.mixer.music.load(audio_fp, 'mp3')
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            print("Played text-to-speech message")
        except pygame.error as e:
            print(f"Error playing MP3 data: {e}")
    except Exception as e:
        print(f"Error with gTTS: {e}")

def update_tokenizer(new_texts):
    global tokenizer
    tokenizer.fit_on_texts(new_texts)
    logger.info(f"Updated tokenizer with texts: {new_texts}")

def print_vocab():
    global tokenizer
    vocab_size = len(tokenizer.word_index)
    vocab = tokenizer.word_index
    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Vocabulary: {vocab}")

def train_on_interaction(input_text, target_text):
    update_tokenizer([input_text, target_text])
    
    input_tokens = proper_tokenizer(input_text)
    target_tokens = proper_tokenizer(target_text)
    
    input_tokens = np.array(input_tokens).reshape((1, -1))
    target_tokens = np.array(target_tokens).reshape((1, -1))
    
    if target_tokens.shape[1] != max_length:
        target_tokens = pad_sequences(target_tokens, maxlen=max_length)
    
    # Convert target tokens to one-hot encoding
    target_tokens_one_hot = tf.keras.utils.to_categorical(target_tokens, num_classes=num_classes)
    
    # Ensure the target shape matches the model output shape
    assert target_tokens_one_hot.shape[2] == num_classes, "Target shape does not match the model's output shape"
    
    logger.info(f"Training input tokens: {input_tokens.shape}")
    logger.info(f"Training target tokens: {target_tokens_one_hot.shape}")

    model.fit(input_tokens, target_tokens_one_hot, epochs=100, verbose=1)

def main():
    initialize_resources()
    
    model_filepath = 'trained_model.keras'
    tokenizer_filepath = 'tokenizer.pkl'
    
    if os.path.exists(model_filepath):
        load_model(model_filepath)
    else:
        logger.info("No pre-trained model found, starting with a new model.")

    if os.path.exists(tokenizer_filepath):
        load_tokenizer(tokenizer_filepath)
    else:
        logger.info("No pre-trained tokenizer found, starting with a new tokenizer.")

    print_vocab()

    while True:
        user_text = listen()
        if user_text:
            logger.info(f"Received user text: {user_text}")
            input_tokens = proper_tokenizer(user_text)
            logger.info(f"Tokenized input: {input_tokens}")
            response = generate_response(input_tokens)
            logger.info(f"Generated response: {response}")
            text_to_speech(response)
            
            expected_response = input("Correct response: ")
            logger.info(f"Expected response: {expected_response}")
            train_on_interaction(user_text, expected_response)
            
            save_model(model_filepath)
            save_tokenizer(tokenizer_filepath)

if __name__ == "__main__":
    main()
