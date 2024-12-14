import time
from utils.network_monitor import NetworkMonitor
from utils.data_processor import DataProcessor
from models.lstm_detector import LSTMAnomalyDetector
import joblib
import torch
import json
from datetime import datetime
import pandas as pd

def main():
    monitor = NetworkMonitor()
    processor = DataProcessor()
    model = LSTMAnomalyDetector(input_size=5)
    
    model.load_state_dict(torch.load('lstm_model.pth'))
    model.eval()

    print("Starting network monitoring...")
    
    while True:
        try:
            metrics = monitor.measure_metrics()
            
            if metrics:
                df = pd.DataFrame([metrics])
                X_tensor = processor.process_new_data(df)
                
                with torch.no_grad():
                    _, prediction, attention = model(X_tensor)
                
                is_anomaly = prediction.item() > 0.7
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_entry = {
                    "timestamp": timestamp,
                    "metrics": metrics,
                    "is_anomaly": bool(is_anomaly),
                    "confidence": float(prediction.item())
                }
                
                with open('network_log.json', 'a') as f:
                    json.dump(log_entry, f)
                    f.write('\n')
                
                status = "ALERT: Anomaly Detected!" if is_anomaly else "Status: Normal"
                print(f"\r{timestamp} - {status}", end="")
                
                if datetime.now().minute == 0:
                    report = monitor.get_hourly_report()
                    report.to_csv(f'reports/hourly_report_{timestamp}.csv')
            
            time.sleep(60)  
            
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            monitor.save_history()
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

if __name__ == "__main__":
    main() 