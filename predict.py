import torch
import pandas as pd
import joblib
from models.lstm_detector import LSTMAnomalyDetector
from models.traditional_detector import TraditionalAnomalyDetector
from utils.data_processor import DataProcessor
from data.synthetic_data import NetworkTrafficGenerator
from utils.visualizer import NetworkVisualizer
import matplotlib.pyplot as plt

def load_models(model_path='lstm_model.pth', scaler_path='scaler.joblib'):
    lstm_model = LSTMAnomalyDetector(input_size=5)
    lstm_model.load_state_dict(torch.load(model_path))
    lstm_model.eval()
    
    processor = DataProcessor()
    processor.scaler = joblib.load(scaler_path)
    
    return lstm_model, processor

def predict_anomalies(data, model, processor, threshold=0.7):
    X_tensor = processor.process_new_data(data)
    
    with torch.no_grad():
        _, predictions, attention_weights = model(X_tensor)
    
    predictions = (predictions.numpy() > threshold).astype(int)
    attention_weights = attention_weights.numpy()
    
    aligned_data = data.copy()
    aligned_data['is_anomaly'] = predictions
    
    confidence = predictions.max(axis=1) if len(predictions.shape) > 1 else predictions
    aligned_data['anomaly_confidence'] = confidence
    
    return aligned_data, attention_weights

if __name__ == "__main__":
    generator = NetworkTrafficGenerator()
    test_data = generator.generate_data(n_samples=1000)
    
    try:
        lstm_model, processor = load_models()
        
        results, attention_weights = predict_anomalies(test_data, lstm_model, processor)
        
        total_anomalies = results['is_anomaly'].sum()
        print(f"\nDetected {total_anomalies} anomalies in {len(results)} network traffic samples")
        
        print("\nExample of detected anomalies:")
        print(results[results['is_anomaly'] == 1].head())
        
        visualizer = NetworkVisualizer()
        
        attention_fig = visualizer.plot_attention_weights(
            results, attention_weights, timestamp_col='timestamp'
        )
        attention_fig.savefig('attention_weights.png')
        
        metrics_fig = visualizer.plot_metrics_with_anomalies(results)
        metrics_fig.savefig('anomaly_detection.png')
        
        plt.close('all')
        
    except FileNotFoundError:
        print("Error: Model files not found. Please train the models first using train.py") 