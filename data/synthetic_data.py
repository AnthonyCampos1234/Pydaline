import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class NetworkTrafficGenerator:
    def __init__(self):
        self.noise_factor = 0.27290689727998174 
        self.traffic_patterns = {
            'normal': {
                'bandwidth_mean': 500,  
                'latency_mean': 20,     
                'packet_loss_mean': 0.1,  
                'packet_size_mean': 1500,  
                'connection_count_mean': 1000  
            },
            'anomaly': {
                'bandwidth_mean': 600,  
                'latency_mean': 25,     
                'packet_loss_mean': 0.3,  
                'packet_size_mean': 800,  
                'connection_count_mean': 2500  
            }
        }

    def generate_data(self, n_samples=10000, anomaly_ratio=0.1):
        n_anomalies = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomalies

        # Generate normal traffic
        normal_data = self._generate_pattern(n_normal, 'normal')
        
        # Generate anomalous traffic
        anomaly_data = self._generate_pattern(n_anomalies, 'anomaly')

        # Combine and shuffle data
        all_data = pd.concat([normal_data, anomaly_data]).sample(frac=1).reset_index(drop=True)
        return all_data

    def _generate_pattern(self, n_samples, pattern_type):
        pattern = self.traffic_patterns[pattern_type]
        
        time = np.linspace(0, 2*np.pi, n_samples)
        time_of_day = np.sin(time)
        day_of_week = np.sin(time/7)
        seasonal_trend = np.sin(time/365)
        random_events = np.random.exponential(1, n_samples) * 0.1
        
        noise = np.random.normal(0, self.noise_factor, n_samples)
        combined_variation = time_of_day + 0.5*day_of_week + 0.2*seasonal_trend + noise + random_events
        
        data = pd.DataFrame({
            'timestamp': [datetime.now() + timedelta(minutes=i) for i in range(n_samples)],
            'bandwidth': np.random.normal(
                pattern['bandwidth_mean'] * (1 + 0.2*time_of_day + 0.1*seasonal_trend + noise), 
                pattern['bandwidth_mean'] * 0.2, 
                n_samples
            ),
            'latency': np.random.normal(
                pattern['latency_mean'] * (1 + 0.1*day_of_week + 0.05*seasonal_trend + noise), 
                pattern['latency_mean'] * 0.15, 
                n_samples
            ),
            'packet_loss': np.random.normal(
                pattern['packet_loss_mean'] * (1 + noise), 
                pattern['packet_loss_mean'] * 0.1, 
                n_samples
            ),
            'packet_size': np.random.normal(
                pattern['packet_size_mean'] * (1 + 0.1*noise),
                pattern['packet_size_mean'] * 0.1,
                n_samples
            ),
            'connection_count': np.random.normal(
                pattern['connection_count_mean'] * (1 + 0.3*time_of_day + noise),
                pattern['connection_count_mean'] * 0.2,
                n_samples
            ),
            'is_anomaly': 1 if pattern_type == 'anomaly' else 0
        })
        
        return data

if __name__ == "__main__":
    generator = NetworkTrafficGenerator()
    data = generator.generate_data()
    print(data.head()) 