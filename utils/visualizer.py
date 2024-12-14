import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class NetworkVisualizer:
    def __init__(self):
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
            print("Note: Using default style. Install seaborn for better visualizations.")
    
    def plot_attention_weights(self, data, attention_weights, timestamp_col='timestamp', n_samples=100):
        """Visualize attention weights over time series data."""
        plt.figure(figsize=(15, 8))
        
        attention = attention_weights[-n_samples:]
        
        if len(attention.shape) == 3:
            if attention.shape[0] == 1:
                attention = attention.squeeze(0)  
            else:
                attention = attention.mean(axis=0)  
        elif len(attention.shape) > 3:
            raise ValueError(f"Unexpected attention weight shape: {attention.shape}")
        
        times = data[timestamp_col].iloc[-n_samples:]
        
        sns.heatmap(
            attention.T,
            cmap='YlOrRd',
            xticklabels=times.dt.strftime('%H:%M'),
            yticklabels=[f't-{i}' for i in range(attention.shape[0]-1, -1, -1)]
        )
        plt.title('Attention Weights Over Time')
        plt.xlabel('Time')
        plt.ylabel('Timesteps')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_metrics_with_anomalies(self, data, metrics=None, n_samples=200):
        """Plot network metrics with highlighted anomalies."""
        if metrics is None:
            metrics = ['bandwidth', 'latency', 'packet_loss', 'packet_size', 'connection_count']
        
        n_metrics = len(metrics)
        plt.figure(figsize=(15, 3*n_metrics))
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(n_metrics, 1, i)
            
            normal_mask = data['is_anomaly'] == 0
            plt.plot(data.index[-n_samples:], 
                    data[metric][-n_samples:], 
                    'b-', alpha=0.5, 
                    label='Normal')
            
            anomaly_mask = data['is_anomaly'] == 1
            plt.scatter(data.index[-n_samples:][anomaly_mask[-n_samples:]], 
                       data[metric][-n_samples:][anomaly_mask[-n_samples:]], 
                       c='r', label='Anomaly')
            
            if 'anomaly_confidence' in data.columns:
                confidence = data['anomaly_confidence'][-n_samples:]
                plt.fill_between(data.index[-n_samples:], 
                               data[metric][-n_samples:] * (1 - confidence),
                               data[metric][-n_samples:] * (1 + confidence),
                               color='r', alpha=0.1)
            
            plt.title(f'{metric.replace("_", " ").title()} Over Time')
            plt.legend()
        
        plt.tight_layout()
        return plt.gcf() 