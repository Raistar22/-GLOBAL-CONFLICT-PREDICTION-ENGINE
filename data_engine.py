import requests
import pandas as pd
import io
import gzip
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataEngine:
    """
    Engine to fetch geopolitical event data from GDELT and flight data from OpenSky.
    """
    GDELT_V2_EVENTS_URL = "http://data.gdeltproject.org/gdeltv2/last15minupdates.txt"
    OPENSKY_URL = "https://opensky-network.org/api/states/all"

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def fetch_latest_gdelt_events(self):
        """
        Fetches the latest 15-minute event updates from GDELT 2.0.
        Returns a DataFrame of recent events.
        """
        try:
            response = requests.get(self.GDELT_V2_EVENTS_URL, timeout=10)
            if response.status_code != 200:
                logger.warning(f"GDELT API returned {response.status_code}. ACTIVATING FAIL-SAFE MODE (Engine will use simulated global intelligence data).")
                return self._get_mock_gdelt_data()
            
            # The updates file contains links to the actual CSVs
            # Format: size date_time zip_url
            lines = response.text.strip().split('\n')
            event_zip_url = None
            for line in lines:
                if '.export.CSV.zip' in line:
                    event_zip_url = line.split()[-1]
                    break
            
            if not event_zip_url:
                logger.error("No export CSV found in GDELT updates list.")
                return self._get_mock_gdelt_data()

            logger.info(f"Fetching GDELT events from: {event_zip_url}")
            event_response = requests.get(event_zip_url, timeout=10)
            
            # Unzip and read
            from zipfile import ZipFile
            from io import BytesIO
            
            with ZipFile(BytesIO(event_response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    # GDELT 2.0 Events columns (Simplified subset)
                    # Full spec is complex, we target key columns: 
                    # 1: Day, 7: Actor1Name, 12: Actor1CountryCode, 17: Actor2Name, 
                    # 22: Actor2CountryCode, 26: EventCode, 30: GoldsteinScale, 
                    # 31: NumMentions, 34: AvgTone, 40: ActionGeo_Lat, 41: ActionGeo_Long, 57: SourceURL
                    cols = [1, 7, 12, 17, 22, 26, 30, 31, 34, 40, 41, 57]
                    names = ['Date', 'Actor1', 'Actor1Country', 'Actor2', 'Actor2Country', 'EventCode', 
                             'Goldstein', 'Mentions', 'Tone', 'Lat', 'Long', 'Source']
                    
                    df = pd.read_csv(f, sep='\t', header=None, usecols=cols, names=names)
                    return df

        except Exception as e:
            logger.warning(f"GDELT intelligence feed unavailable: {e}. ACTIVATING FAIL-SAFE MODE.")
            return self._get_mock_gdelt_data()

    def _get_mock_gdelt_data(self):
        """Returns dummy GDELT data to keep the UI functional."""
        return pd.DataFrame({
            'Date': [20260302] * 7,
            'Actor1': ['UKR', 'ISR', 'CHN', 'USA', 'IRN', 'TWN', 'KOR'],
            'Actor1Country': ['UKR', 'ISR', 'CHN', 'USA', 'IRN', 'TWN', 'KOR'],
            'Actor2': ['RUS', 'IRN', 'PHL', 'RUS', 'ISR', 'CHN', 'PRK'],
            'Actor2Country': ['RUS', 'IRN', 'PHL', 'RUS', 'ISR', 'CHN', 'PRK'],
            'EventCode': ['190', '190', '040', '190', '190', '160', '190'],
            'Goldstein': [-10.0, -9.5, -4.0, -8.0, -9.0, -6.5, -9.2],
            'Mentions': [500, 450, 200, 300, 400, 150, 350],
            'Tone': [-12.5, -11.0, -5.2, -7.5, -10.5, -8.0, -11.5],
            'Lat': [50.45, 31.76, 14.59, 38.90, 35.68, 25.03, 37.56],
            'Long': [30.52, 35.21, 120.98, -77.03, 51.38, 121.56, 126.97],
            'Source': ['https://reuters.com', 'https://bbc.com', 'https://aljazeera.com', 'https://nytimes.com', 'https://theguardian.com', 'https://cnn.com', 'https://ap.org']
        })

    def fetch_flight_data(self):
        """
        Fetches current global flight states from OpenSky Network.
        """
        try:
            response = requests.get(self.OPENSKY_URL, timeout=10)
            if response.status_code == 200:
                data = response.json()
                states = data.get('states', [])
                # Columns: 0: icao24, 1: callsign, 2: origin_country, 5: longitude, 6: latitude, 7: baro_altitude
                df = pd.DataFrame(states, columns=['icao24', 'callsign', 'origin_country', 'time_position', 
                                                   'last_contact', 'longitude', 'latitude', 'baro_altitude', 
                                                   'on_ground', 'velocity', 'true_track', 'vertical_rate', 
                                                   'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source'])
                return df
            else:
                logger.warning(f"OpenSky API returned {response.status_code}. ACTIVATING FAIL-SAFE MODE.")
                return self._get_mock_opensky_data()
        except Exception as e:
            logger.warning(f"OpenSky flight feed unavailable: {e}. ACTIVATING FAIL-SAFE MODE.")
            return self._get_mock_opensky_data()

    def _get_mock_opensky_data(self):
        """Returns dummy flight data to keep the UI functional."""
        return pd.DataFrame([
            ['ICAO1', 'FLIGHT1', 'USA', 0, 0, -74.006, 40.7128, 30000, False, 500, 90, 0, None, 30000, None, False, 0],
            ['ICAO2', 'FLIGHT2', 'DEU', 0, 0, 13.405, 52.5200, 35000, False, 480, 270, 0, None, 35000, None, False, 0],
            ['ICAO3', 'FLIGHT3', 'UKR', 0, 0, 30.523, 50.450, 25000, False, 450, 180, 0, None, 25000, None, False, 0],
            ['ICAO4', 'FLIGHT4', 'ISR', 0, 0, 34.781, 32.085, 28000, False, 460, 0, 0, None, 28000, None, False, 0],
            ['ICAO5', 'FLIGHT5', 'TWN', 0, 0, 121.56, 25.03, 31000, False, 490, 45, 0, None, 31000, None, False, 0]
        ], columns=['icao24', 'callsign', 'origin_country', 'time_position', 'last_contact', 'longitude', 'latitude', 'baro_altitude', 'on_ground', 'velocity', 'true_track', 'vertical_rate', 'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source'])

if __name__ == "__main__":
    engine = DataEngine()
    print("Testing GDELT fetch...")
    gdelt_df = engine.fetch_latest_gdelt_events()
    if gdelt_df is not None:
        print(f"Fetched {len(gdelt_df)} events.")
        print(gdelt_df.head())
    
    print("\nTesting OpenSky fetch...")
    flights_df = engine.fetch_flight_data()
    if flights_df is not None:
        print(f"Fetched {len(flights_df)} flight states.")
        print(flights_df.head())
