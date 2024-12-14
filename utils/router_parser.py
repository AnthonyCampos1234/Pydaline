import re
from datetime import datetime
from typing import List, Dict, Optional
import glob
import os

class RouterLogParser:
    def __init__(self, log_dir: str = "/var/log/"):
        self.log_dir = log_dir
        self.patterns = {
            'connection_drop': r'Connection dropped|Link down|Disconnected',
            'auth_failure': r'Authentication failed|Auth timeout',
            'high_usage': r'bandwidth usage exceeded|high traffic detected',
            'interference': r'signal interference|weak signal|poor signal quality',
            'dhcp_issue': r'DHCP timeout|IP conflict|lease expired'
        }
    
    def parse_recent_logs(self, hours: int = 1) -> List[Dict]:
        events = []
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        try:
            log_files = glob.glob(os.path.join(self.log_dir, "*.log"))
            log_files.extend(glob.glob(os.path.join(self.log_dir, "syslog*")))
            
            for log_file in log_files:
                events.extend(self._parse_file(log_file, cutoff_time))
            
            return events
            
        except Exception as e:
            print(f"Error parsing router logs: {e}")
            return []
    
    def _parse_file(self, filepath: str, cutoff_time: float) -> List[Dict]:
        events = []
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    event = self._parse_line(line)
                    if event and event['timestamp'] >= cutoff_time:
                        events.append(event)
        except Exception as e:
            print(f"Error reading log file {filepath}: {e}")
        return events
    
    def _parse_line(self, line: str) -> Optional[Dict]:
        timestamp_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\b',
            r'\b\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b'
        ]
        
        try:
            for pattern in timestamp_patterns:
                match = re.search(pattern, line)
                if match:
                    timestamp_str = match.group(0)
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        timestamp = datetime.strptime(timestamp_str, '%b %d %H:%M:%S')
                        timestamp = timestamp.replace(year=datetime.now().year)
                    
                    for event_type, pattern in self.patterns.items():
                        if re.search(pattern, line, re.IGNORECASE):
                            return {
                                'timestamp': timestamp.timestamp(),
                                'event_type': event_type,
                                'message': line.strip(),
                                'source': 'router_log'
                            }
            return None
            
        except Exception as e:
            print(f"Error parsing log line: {e}")
            return None 