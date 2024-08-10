import torch
import json
from nltk.tokenize import word_tokenize
from model import Seq2SeqLSTM
import gtts
import pygame
import io

# Initialize pygame mixer
pygame.mixer.init()

# Load vocab and model
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

input_dim = len(vocab) + 1
hidden_dim = 128
output_dim = len(vocab)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2SeqLSTM(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(torch.load('chatbot_model.pth'))
model.eval()

# Define response_dict based on your model's output
response_dict = {
    0: "Hello!",
    1: "How can I help you?",
    2: "Goodbye!",
    3: "I don't understand.",
    # Add more mappings based on your model's output
}

def predict_response(model, input_sentence):
    model.eval()
    with torch.no_grad():
        # Tokenize input and convert to tensor
        tokens = word_tokenize(input_sentence.lower())
        src = torch.tensor([vocab.get(token, 0) for token in tokens], dtype=torch.long).unsqueeze(0)
        src = src.to(device)

        # Get model output
        output = model(src, lengths=[src.size(1)])
        
        # Print for debugging
        print(f"Tokenized input: {tokens}")
        print(f"Model output: {output}")

        # Get the most probable index
        _, predicted_idx = torch.max(output, 1)
        
        # Map index to response
        response = response_dict.get(predicted_idx.item(), "Sorry, I don't understand.")
        return response

def speak_response(response):
    # Generate speech using gTTS
    tts = gtts.gTTS(text=response, lang='en')
    
    # Save to a bytes buffer
    buffer = io.BytesIO()
    tts.write_to_fp(buffer)
    buffer.seek(0)
    
    # Load buffer into pygame mixer
    pygame.mixer.music.load(buffer, 'mp3')
    pygame.mixer.music.play()
    
    # Wait until the audio is done playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # Check every 10ms

def generate_response(input_sentence):
    return predict_response(model, input_sentence)

print("Chatbot is ready! Type 'quit' to exit.")
while True:
    query = input("You: ")
    if query.lower() == 'quit':
        break

    response = generate_response(query)
    print(f"Bot: {response}")
    speak_response(response)
