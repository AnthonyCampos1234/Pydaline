from utils.network_monitor import NetworkMonitor
from datetime import datetime
import json

def test_monitoring():
    print("Starting Basic Network Monitoring Test...")
    print("-" * 50)
    
    try:
        print("Testing Weather API...")
        from utils.weather_monitor import WeatherMonitor
        weather = WeatherMonitor()
        weather_data = weather.get_current_weather()
        if weather_data:
            print("\nWeather Data Retrieved Successfully:")
            print(f"Temperature: {weather_data['temperature']}°F")
            print(f"Conditions: {weather_data['weather_condition']}")
        
        print("\nTesting Network Connectivity...")
        import subprocess
        result = subprocess.run(["ping", "-c", "1", "8.8.8.8"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("Network connectivity: OK")
            latency = float(result.stdout.split("time=")[1].split()[0])
            print(f"Latency: {latency} ms")
        
        print("\nTesting System Stats...")
        import psutil
        net_io = psutil.net_io_counters()
        print(f"Bytes sent: {net_io.bytes_sent / 1024 / 1024:.2f} MB")
        print(f"Bytes received: {net_io.bytes_recv / 1024 / 1024:.2f} MB")
        
        print("\nAll basic tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print("Try running with elevated privileges if needed.")
    
if __name__ == "__main__":
    test_monitoring() 