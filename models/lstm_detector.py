import torch
import torch.nn as nn

class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(LSTMAnomalyDetector, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        self.reconstruction = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, input_size)
        )
        
        self.classification = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        lstm_out, _ = self.lstm(x)
        
        attention_weights = self.attention(lstm_out)
        self.last_attention_weights = attention_weights.detach().cpu().numpy()
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        reconstructed = self.reconstruction(context)
        classification = self.classification(context)
        
        return reconstructed, classification, attention_weights 