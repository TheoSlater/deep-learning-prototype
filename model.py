import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Seq2SeqLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        
        if lengths is not None:
            x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        x, (hn, cn) = self.lstm(x)
        
        if lengths is not None:
            x, _ = pad_packed_sequence(x, batch_first=True)
        
        x = self.fc(x[:, -1, :])  # Take the output from the last timestep
        return x
