import torch
import torch.nn as nn
import torch.optim as optim
from data.synthetic_data import NetworkTrafficGenerator
from models.lstm_detector import LSTMAnomalyDetector
from models.traditional_detector import TraditionalAnomalyDetector
from utils.data_processor import DataProcessor
import numpy as np
import joblib
from sklearn.metrics import precision_recall_fscore_support

def get_batches(X, y, batch_size):
    """Helper function to create properly sized batches"""
    n_samples = len(X)
    indices = torch.randperm(n_samples)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield X[batch_indices], y[batch_indices]

def train_models(epochs=50, batch_size=32):  
    generator = NetworkTrafficGenerator()
    data = generator.generate_data(n_samples=50000, anomaly_ratio=0.05)
    
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    processor = DataProcessor()
    X_train, y_train = processor.prepare_data(train_data)
    X_val, y_val = processor.prepare_data(val_data)
    
    lstm_model = LSTMAnomalyDetector(input_size=5)  
    traditional_model = TraditionalAnomalyDetector()
    
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.BCELoss()
    
    best_loss = float('inf')
    best_f1 = 0
    patience = 7  
    patience_counter = 0
    
    for epoch in range(epochs):
        lstm_model.train()
        total_train_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        for batch_X, batch_y in get_batches(X_train, y_train, batch_size):
            decoded, classification, _ = lstm_model(batch_X)  
            
            reconstruction_loss = reconstruction_criterion(decoded, batch_X[:, -1, :])
            classification_loss = classification_criterion(classification, batch_y)
            loss = 0.5 * reconstruction_loss + 0.5 * classification_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            all_train_preds.extend(classification.detach().numpy() > 0.5)
            all_train_labels.extend(batch_y.numpy())
        
        lstm_model.eval()
        with torch.no_grad():
            decoded_val, classification_val, _ = lstm_model(X_val)  
            val_reconstruction_loss = reconstruction_criterion(decoded_val, X_val[:, -1, :])
            val_classification_loss = classification_criterion(classification_val, y_val)
            val_loss = 0.5 * val_reconstruction_loss + 0.5 * val_classification_loss
            
            val_preds = (classification_val.numpy() > 0.5).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val.numpy(), val_preds, average='binary'
            )
        
        avg_train_loss = total_train_loss / len(X_train)
        if (epoch + 1) % 5 == 0:  
            print(f'Epoch [{epoch+1}/{epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            torch.save(lstm_model.state_dict(), 'lstm_model.pth')
            joblib.dump(processor.scaler, 'scaler.joblib')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    traditional_model.fit(X_train.numpy())
    
    return lstm_model, traditional_model, processor

if __name__ == "__main__":
    lstm_model, traditional_model, processor = train_models()
    
    torch.save(lstm_model.state_dict(), 'lstm_model.pth')
    joblib.dump(processor.scaler, 'scaler.joblib') 