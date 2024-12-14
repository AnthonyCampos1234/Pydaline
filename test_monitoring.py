from utils.network_monitor import NetworkMonitor
from datetime import datetime
import json

def test_monitoring():
    print("Starting Network Monitoring Test...")
    print("-" * 50)
    
    monitor = NetworkMonitor()
    
    try:
        print("Collecting metrics...")
        metrics = monitor.measure_metrics()
        
        if metrics:
            print("\nNetwork Metrics:")
            print(f"Timestamp: {metrics['timestamp']}")
            print(f"Bandwidth: {metrics['bandwidth']:.2f} MB/s")
            print(f"Latency: {metrics['latency']:.2f} ms")
            print(f"Packet Loss: {metrics['packet_loss']:.2f}%")
            print(f"Active Connections: {metrics['connection_count']}")
            
            print("\nWeather Data:")
            print(f"Temperature: {metrics.get('temperature', 'N/A')}°F")
            print(f"Humidity: {metrics.get('humidity', 'N/A')}%")
            print(f"Weather Condition: {metrics.get('weather_condition', 'N/A')}")
            print(f"Wind Speed: {metrics.get('wind_speed', 'N/A')} mph")
            
            print("\nRouter Events:")
            if 'router_issues' in metrics:
                print(f"Recent Issues: {metrics['router_issues']}")
                print(f"Number of Events: {metrics['recent_router_events']}")
            else:
                print("No router events detected")
            
            with open('test_output.json', 'w') as f:
                metrics['timestamp'] = metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                json.dump(metrics, f, indent=2)
                print("\nDetailed output saved to test_output.json")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_monitoring() 