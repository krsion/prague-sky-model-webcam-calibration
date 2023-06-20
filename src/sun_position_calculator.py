import pytz
from timezonefinder import TimezoneFinder
import json
from pysolar.solar import get_altitude, get_azimuth
import numpy as np
import datetime


class SunPositionCalculator:
    def __init__(self):
        self.webcam_positions = self._webcams()

    def _webcams(self):
        azimuths = {'Z': 270, 'JV': 135, 'V': 90, 'SV': 45, 'JZ': 225, 'Z-SZ': 292.5}
        positions = {}
        with open('../data/webcams.json', 'r', encoding='utf-8') as f:
            webcams = json.load(f)
        tf = TimezoneFinder()
        for webcam in webcams:
            lat = float(webcam['lat'])
            lon = float(webcam['lon'])
            direction = webcam['smer']
            positions[webcam['file']] = {
                'lat': lat, 'lon': lon, 'alt': float(webcam['vyska']), 'tz': tf.timezone_at(lat=lat, lng=lon), 'direction': direction, 'azimuth': 0}
            if direction in azimuths:
                positions[webcam['file']]['azimuth'] = np.deg2rad(
                    azimuths[direction])
        return positions

    def _parse_filepath(self, filepath):
        parts = filepath.split('/')
        location = parts[-3]
        year = int(parts[-2][:4])
        month = int(parts[-2][4:6])
        day = int(parts[-2][6:])
        hour = int(parts[-1][:2])
        minute = int(parts[-1][2:4])
        return location, year, month, day, hour, minute

    def sun_position(self, filename):
        location, year, month, day, hour, minute = self._parse_filepath(filename)
        lat = self.webcam_positions[location]['lat']
        lon = self.webcam_positions[location]['lon']
        tz = self.webcam_positions[location]['tz']
        date = datetime.datetime(year, month, day, hour, minute, tzinfo=pytz.timezone(tz))
        return {'sunZenith': 90-get_altitude(lat, lon, date), 'sunAzimuth': get_azimuth(lat, lon, date)}
