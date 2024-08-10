import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import torch.nn as nn
from model import Seq2SeqLSTM
from nltk.tokenize import word_tokenize
from collections import defaultdict

class ConversationsDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tokens, tag = self.pairs[idx]
        src = torch.tensor([self.vocab.get(token, 0) for token in tokens], dtype=torch.long)
        trg = torch.tensor(self.vocab.get(tag, 0), dtype=torch.long)  # Ensure trg is a 1D tensor
        return src, trg, len(src)  # Return lengths for packing

def create_training_data(intents):
    pairs = []
    vocab = defaultdict(int)
    
    for intent in intents['intents']:
        tag = intent['tag']
        for pattern in intent['patterns']:
            tokens = word_tokenize(pattern.lower())
            pairs.append((tokens, tag))
            for token in tokens:
                vocab[token] += 1
            vocab[tag] += 1
    
    return pairs, vocab

def collate_fn(batch):
    src, trg, lengths = zip(*batch)
    src_lengths = torch.tensor(lengths, dtype=torch.long)
    src_padded = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)

    # Convert trg to a list of tensors and then concatenate
    trg = torch.stack(trg)  # Use stack instead of cat if trg tensors are single-dimensional
    return src_padded, trg, src_lengths

def train_model(model, dataloader, epochs=1500, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        model.train()  # Ensure model is in training mode
        for src, trg, lengths in dataloader:
            src, trg, lengths = src.to(device), trg.to(device), lengths.to(device)
            optimizer.zero_grad()
            output = model(src, lengths)
            
            # Ensure output and trg have the correct shape
            if output.shape[0] != trg.shape[0]:
                print(f"Shape mismatch: output {output.shape}, trg {trg.shape}")

            # Compute the loss
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()
            
            # Get predictions and calculate accuracy
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == trg).sum().item()
            total_predictions += trg.size(0)

        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Print epoch, loss, and accuracy
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}, Accuracy: {accuracy:.4f}')

        # Optionally print some example predictions
        model.eval()
        with torch.no_grad():
            example_input = "Hello"
            tokens = word_tokenize(example_input.lower())
            src = torch.tensor([vocab.get(token, 0) for token in tokens], dtype=torch.long).unsqueeze(0).to(device)
            output = model(src, lengths=[src.size(1)])
            _, predicted_idx = torch.max(output, 1)
            print(f"Example input: {example_input}")
            print(f"Model output index: {predicted_idx.item()}")

# Load intents and prepare data
with open('intents.json', 'r') as f:
    intents = json.load(f)

pairs, vocab = create_training_data(intents)

# Save vocab for later use
with open('vocab.json', 'w') as f:
    json.dump(vocab, f, indent=2)

dataset = ConversationsDataset(pairs, vocab)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

input_dim = len(vocab) + 1  # Plus one for padding token
hidden_dim = 128
output_dim = len(vocab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2SeqLSTM(input_dim, hidden_dim, output_dim).to(device)

train_model(model, dataloader, epochs=1000)

torch.save(model.state_dict(), 'chatbot_model.pth')