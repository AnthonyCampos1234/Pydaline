import requests
from datetime import datetime
import os
from typing import Dict, Optional
from dotenv import load_dotenv
import geocoder

load_dotenv()

class WeatherMonitor:
    def __init__(self, api_key: str = None, city: str = "Lewisburg,PA"):
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        if not self.api_key or self.api_key == 'your_api_key_here':
            raise ValueError("OpenWeather API key required. Set OPENWEATHER_API_KEY in .env file")
        
        self.city = city or os.getenv('CITY', 'Lewisburg,PA')
        self.base_url = "https://api.openweathermap.org/data/3.0/onecall"
        
        self.coordinates = self._get_coordinates()
        
    def _get_coordinates(self) -> tuple:
        """Get latitude and longitude for the city"""
        g = geocoder.osm(self.city)
        if g.ok:
            return (g.lat, g.lng)
        else:
            raise ValueError(f"Could not get coordinates for {self.city}")
    
    def get_current_weather(self) -> Optional[Dict]:
        """Get detailed weather data including forecasts"""
        try:
            params = {
                'lat': self.coordinates[0],
                'lon': self.coordinates[1],
                'appid': self.api_key,
                'units': 'imperial',  
                'exclude': 'minutely'  
            }
            
            response = requests.get(self.base_url, params=params)
            if response.status_code == 401:
                print("Weather API Error: Unauthorized. If you just registered, the API key may take 2-3 hours to activate.")
                return self._get_default_weather()
            
            response.raise_for_status()
            data = response.json()
            
            current = data['current']
            
            next_hour = data['hourly'][0]
            
            return {
                'timestamp': datetime.now(),
                'temperature': current['temp'],
                'humidity': current['humidity'],
                'weather_condition': current['weather'][0]['main'],
                'weather_description': current['weather'][0]['description'],
                'wind_speed': current['wind_speed'],
                'wind_gust': current.get('wind_gust', 0),
                'precipitation_prob': next_hour.get('pop', 0) * 100,  
                'precipitation': current.get('rain', {}).get('1h', 0),
                'clouds': current['clouds'],
                'uvi': current['uvi'],
                'visibility': current['visibility'],
                'alerts': [alert['event'] for alert in data.get('alerts', [])]
            }
            
        except Exception as e:
            print(f"Error fetching weather data: {e}")
            return self._get_default_weather()
    
    def _get_default_weather(self) -> Dict:
        """Return default weather data when API fails"""
        return {
            'timestamp': datetime.now(),
            'temperature': 70,
            'humidity': 50,
            'weather_condition': 'Unknown',
            'weather_description': 'No data available',
            'wind_speed': 0,
            'wind_gust': 0,
            'precipitation_prob': 0,
            'precipitation': 0,
            'clouds': 0,
            'uvi': 0,
            'visibility': 10000,
            'alerts': []
        } 