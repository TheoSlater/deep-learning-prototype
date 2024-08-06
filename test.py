import numpy as np
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM, Embedding
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras._tf_keras.keras.preprocessing.text import Tokenizer
import pickle

# Constants
MAX_SEQUENCE_LENGTH = 10
VOCAB_SIZE = 10  # Adjust this based on your vocabulary size
EMBEDDING_DIM = 8
NUM_CLASSES = 3  # Number of classes (words) in the output
EPOCHS = 100

# Sample training data
texts = [
    "hi hello",  # Example input sequences
    "hello hi",
]

# Create labels (next words in sequence)
labels = [
    "hello",  # Corresponding labels for the inputs
    "hi",
]

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts + labels)  # Fit on both texts and labels

# Prepare sequences
text_sequences = tokenizer.texts_to_sequences(texts)
label_sequences = tokenizer.texts_to_sequences(labels)

# Pad sequences
X = pad_sequences(text_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
y = pad_sequences(label_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Convert labels to one-hot encoding
y = np.array([np.eye(NUM_CLASSES)[seq] for seq in y])

# Define the model
model = Sequential()
model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
model.add(LSTM(50, return_sequences=True))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=EPOCHS, verbose=1)

# Save the model
model.save('trained_model.keras')

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle)

print("Model and tokenizer saved.")
