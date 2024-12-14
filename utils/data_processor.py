import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.sequence_length = 10  
        
    def prepare_data(self, df):
        # Extract features
        features = ['bandwidth', 'latency', 'packet_loss', 'packet_size', 'connection_count']
        X = df[features].values
        y = df['is_anomaly'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        X_seq = self._create_sequences(X_scaled)
        y_seq = y[:len(X_seq)]
        
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor(y_seq).reshape(-1, 1)
        
        return X_tensor, y_tensor

    def _create_sequences(self, data):
        sequences = []
        padding = np.zeros((self.sequence_length - 1, data.shape[1]))
        padded_data = np.vstack([padding, data])
        
        for i in range(len(data)):
            sequence = padded_data[i:i + self.sequence_length]
            sequences.append(sequence)
        return np.array(sequences)

    def process_new_data(self, df):
        features = ['bandwidth', 'latency', 'packet_loss', 'packet_size', 'connection_count']
        X = df[features].values
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)
        return torch.FloatTensor(X_seq) 