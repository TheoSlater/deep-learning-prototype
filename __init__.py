import sounddevice as sd
import numpy as np
import speech_recognition as sr
import time
import os
from gtts import gTTS
import pygame
from io import BytesIO
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import json
import pandas as pd
import wavio

# Initialize or load dataset
def load_dataset():
    if os.path.exists('dataset.json'):
        with open('dataset.json', 'r') as file:
            data = json.load(file)
        return pd.DataFrame(data['data'])
    else:
        return pd.DataFrame(columns=['text', 'label'])

def save_dataset(df):
    data = {'data': df.to_dict(orient='records')}
    with open('dataset.json', 'w') as file:
        json.dump(data, file, indent=4)

def preprocess_data(df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    
    max_length = max(len(seq) for seq in sequences) if sequences else 1
    X = pad_sequences(sequences, maxlen=max_length)
    
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    return tokenizer, le, X, y, max_length

def create_model(num_classes, vocab_size, embedding_dim=64):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def initialize_resources(df):
    global model, tokenizer, le, X, y, max_length
    tokenizer, le, X, y, max_length = preprocess_data(df)
    
    vocab_size = len(tokenizer.word_index)
    
    if len(df) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = create_model(len(le.classes_), vocab_size)
        history = model.fit(X_train, y_train, epochs=100, batch_size=50, validation_data=(X_test, y_test))
        print("Training History:", history.history)
    else:
        model = create_model(len(le.classes_), vocab_size)
    
    model.save('intent_model.keras')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('label_encoder.pickle', 'wb') as handle:
        pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('max_length.pickle', 'wb') as handle:
        pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_resources():
    global model, tokenizer, le, max_length
    model = tf.keras.models.load_model('intent_model.keras')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as handle:
        le = pickle.load(handle)
    with open('max_length.pickle', 'rb') as handle:
        max_length = pickle.load(handle)
    
    df = load_dataset()
    sequences = tokenizer.texts_to_sequences(df['text'])
    X = pad_sequences(sequences, maxlen=max_length)

def listen():
    samplerate = 44100
    duration = 5
    print("I am listening...")
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()
    wavio.write("temp.wav", recording, samplerate, sampwidth=2)
    
    r = sr.Recognizer()
    with sr.AudioFile('temp.wav') as source:
        audio = r.record(source)

    data = ""
    try:
        data = r.recognize_google(audio)
        print("You said: " + data)
    except sr.UnknownValueError:
        print("Google Speech Recognition did not understand audio")
    except sr.RequestError as e:
        print("Request Failed; {0}".format(e))
    return data

def respond(audioString):
    print(audioString)
    tts = gTTS(text=audioString, lang='en')
    audio_fp = BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    
    pygame.mixer.init()
    pygame.mixer.music.load(audio_fp, 'mp3')
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    prediction = model.predict(padded_sequence)
    
    label_index = prediction.argmax()
    print(f"Predicted label index: {label_index}")  # Debug print
    label = le.inverse_transform([label_index])[0]
    print(f"Predicted label: {label}")  # Debug print
    
    return label

def digital_assistant(data):
    user_intent = predict_intent(data.lower())
    
    # Debugging print to show what was predicted
    print(f"Predicted user intent: {user_intent}")

    # Define command responses
    commands = {
        '0': "You said 'hi'. This is a greeting.",
        '1': "You said 'hello'. This is a greeting.",
        '2': "You said 'goodbye'. This is a farewell.",
        # Add other possible labels if necessary
    }
    
    # Debugging print to show the available commands
    print(f"Available commands: {commands}")
    
    if user_intent == "unknown":
        respond("I did not understand what you said. Please provide a label for this new command.")
        return False
    else:
        response = commands.get(user_intent, "I did not understand what you said.")
        respond(response)

def update_dataset(text, label):
    df = load_dataset()
    new_data = pd.DataFrame({'text': [text], 'label': [label]})
    df = pd.concat([df, new_data], ignore_index=True)
    save_dataset(df)

def retrain_model():
    df = load_dataset()
    initialize_resources(df)

def main():
    if os.path.exists('intent_model.keras'):
        load_resources()
    else:
        df = load_dataset()
        initialize_resources(df)

    time.sleep(2)
    respond("Hi, I am learning. Please help me learn by talking to me!")
    listening = True
    while listening:
        data = listen()
        if data:
            user_intent = predict_intent(data.lower())
            if user_intent == "unknown":
                respond("I did not understand what you said. Please provide a label for this new command.")
                new_command = input("Please provide a label for this new command (0-2): ")
                if new_command.isdigit():
                    label = int(new_command)
                    update_dataset(data, label)
                    retrain_model()
                    respond("I have learned a new command!")
            else:
                listening = digital_assistant(data)

if __name__ == "__main__":
    main()
