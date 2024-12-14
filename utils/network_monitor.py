import speedtest
import ping3
import psutil
import time
import pandas as pd
from datetime import datetime
from utils.weather_monitor import WeatherMonitor
from utils.router_parser import RouterLogParser
import platform
import subprocess

class NetworkMonitor:
    def __init__(self, target_host="8.8.8.8", weather_api_key=None):
        self.target_host = target_host
        self.speedtest = speedtest.Speedtest()
        self.history = []
        self.weather_monitor = WeatherMonitor(api_key=weather_api_key)
        self.router_parser = RouterLogParser()
        self._check_requirements()

    def _check_requirements(self):
        """Check if we have necessary permissions and access"""
        try:
            if platform.system() == "Darwin":  
                subprocess.run(["ping", "-c", "1", self.target_host], 
                             capture_output=True, check=True)
            else:  
                ping3.ping(self.target_host)
            
            psutil.net_io_counters()
            
        except subprocess.CalledProcessError:
            print("Error: Unable to ping. Try running with sudo or check permissions.")
            raise
        except Exception as e:
            print(f"Error during setup: {e}")
            print("Try running with elevated privileges if needed.")
            raise

    def measure_metrics(self):
        try:
            if platform.system() == "Darwin":  # macOS
                result = subprocess.run(["ping", "-c", "1", self.target_host], 
                                      capture_output=True, text=True)
                latency = float(result.stdout.split("time=")[1].split()[0])
            else:
                latency = ping3.ping(self.target_host) * 1000
            
            bytes_sent = psutil.net_io_counters().bytes_sent
            bytes_recv = psutil.net_io_counters().bytes_recv
            time.sleep(1)  
            new_bytes_sent = psutil.net_io_counters().bytes_sent
            new_bytes_recv = psutil.net_io_counters().bytes_recv
            
            bandwidth_up = (new_bytes_sent - bytes_sent) / 1024 / 1024  # MB/s
            bandwidth_down = (new_bytes_recv - bytes_recv) / 1024 / 1024  # MB/s

            try:
                connections = len(psutil.net_connections())
            except psutil.AccessDenied:
                print("Warning: Cannot access connection info. Using network interface count instead.")
                connections = len(psutil.net_if_stats())
            
            packet_loss = 0
            for _ in range(3):
                if ping3.ping(self.target_host) is None:
                    packet_loss += 1
            packet_loss = (packet_loss / 3) * 100

            metrics = {
                'timestamp': datetime.now(),
                'bandwidth': bandwidth_down,
                'latency': latency,
                'packet_loss': packet_loss,
                'packet_size': 1500,  
                'connection_count': connections
            }
            
            weather_data = self.weather_monitor.get_current_weather()
            if weather_data:
                metrics.update({
                    'temperature': weather_data['temperature'],
                    'humidity': weather_data['humidity'],
                    'weather_condition': weather_data['weather_condition'],
                    'wind_speed': weather_data['wind_speed']
                })
            
            router_events = self.router_parser.parse_recent_logs(hours=1)
            if router_events:
                metrics['recent_router_events'] = len(router_events)
                metrics['router_issues'] = [e['event_type'] for e in router_events]
            
            self.history.append(metrics)
            return metrics

        except Exception as e:
            print(f"Error measuring metrics: {e}")
            return None

    def get_hourly_report(self):
        df = pd.DataFrame(self.history)
        hourly_stats = df.set_index('timestamp').resample('H').agg({
            'bandwidth': ['mean', 'min', 'max'],
            'latency': ['mean', 'min', 'max'],
            'packet_loss': 'mean',
            'connection_count': 'mean'
        })
        return hourly_stats

    def save_history(self, filename='network_history.csv'):
        pd.DataFrame(self.history).to_csv(filename, index=False) 