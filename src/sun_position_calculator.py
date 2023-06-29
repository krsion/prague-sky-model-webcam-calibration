import pytz
from timezonefinder import TimezoneFinder
import json
from pysolar.solar import get_altitude, get_azimuth
import numpy as np
import datetime
import warnings


class SunPositionCalculator:
    """Given date, time, timezone, latitude and longitude calculates the sun azimuth angle and zenith angle.""
    """
    def __init__(self, filename:str):
        """Loads webcam information from a json file for sun position calculations."""
        warnings.filterwarnings("ignore", category=UserWarning, message="I don't know about leap seconds after 2021")
        self.webcam_positions = {}
        azimuths = {'Z': 270, 'JV': 135, 'V': 90, 'SV': 45, 'JZ': 225, 'Z-SZ': 292.5}
        with open(filename, 'r', encoding='utf-8') as f:
            webcams = json.load(f)
        tf = TimezoneFinder()
        for webcam in webcams:
            lat = float(webcam['lat'])
            lon = float(webcam['lon'])
            direction = webcam['smer']
            self.webcam_positions[webcam['file']] = {
                'lat': lat, 'lon': lon, 'alt': float(webcam['vyska']), 'tz': tf.timezone_at(lat=lat, lng=lon), 'direction': direction, 'azimuth': 0}
            if direction in azimuths:
                self.webcam_positions[webcam['file']]['azimuth'] = np.deg2rad(azimuths[direction])
        
        
    def _parse_filepath(self, filepath:str):
        """Separates the filepath into its components.

        Args:
            filepath (str): full fipepath to the image

        Returns:
            location: str, year:int, month:int, day:int, hour:int, minute:int
        """
        parts = filepath.split('/')
        location = parts[-3]
        year = int(parts[-2][:4])
        month = int(parts[-2][4:6])
        day = int(parts[-2][6:])
        hour = int(parts[-1][:2])
        minute = int(parts[-1][2:4])
        return location, year, month, day, hour, minute


    def sun_position(self, filename):
        """From date and time of capture of [filename] and webcams coordinates calculates the sun azimuth and zenith in radians.
        """
        location, year, month, day, hour, minute = self._parse_filepath(filename)
        lat = self.webcam_positions[location]['lat']
        lon = self.webcam_positions[location]['lon']
        tz = self.webcam_positions[location]['tz']
        date = datetime.datetime(year, month, day, hour, minute, tzinfo=pytz.timezone(tz))
        return {'sunZenith': np.deg2rad(90-get_altitude(lat, lon, date)), 'sunAzimuth': np.deg2rad(get_azimuth(lat, lon, date))}
