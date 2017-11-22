#!/usr/bin/env python3
import argparse
from collections import (
    defaultdict,
    namedtuple,
)
import datetime
import json
import math
from pathlib import Path
import re
import sys
from xml.etree import ElementTree


from bs4 import BeautifulSoup
import psycopg2
import pytz
import requests
from tabulate import tabulate


from . import config


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers()

    altitude_arg = argparse.ArgumentParser(add_help=False)
    altitude_arg.add_argument(
        'altitude',
        type=int,
        help='Altitude to calculate',
    )

    perf_parser = subparsers.add_parser(
        'perf',
        parents=[altitude_arg],
        help='Calculate performance data for a given altitude and temperature',
    )
    perf_parser.add_argument(
        'performance_chart',
        type=Path,
        help='Path to performance chart file',
    )
    perf_parser.add_argument(
        'temperature',
        type=int,
        help='Temperature (F) to calculate',
    )
    perf_parser.add_argument(
        'field',
        choices=['takeoff_distance', '50_ft_obstacle', 'rate_of_climb'],
        help='Field to calculate',
    )
    perf_parser.add_argument(
        '--safety-margin', '-m',
        default=0,
        type=int,
        help='Additional margin to add to the estimate (default %(default)s)',
    )
    perf_parser.set_defaults(cmd=cmd_perf)

    winds_parser = subparsers.add_parser(
        'winds',
        parents=[altitude_arg],
        help='Get winds aloft at a given airport',
    )
    winds_parser.add_argument(
        'airport',
        help='Airport to get wind data for',
    )
    winds_parser.add_argument(
        'time',
        help='Time to fetch winds for (2017-11-20 1620PST',
    )
    winds_parser.set_defaults(cmd=cmd_winds)

    declination_parser = subparsers.add_parser(
        'declination',
        help='Get declination for a given point',
    )
    declination_parser.add_argument(
        'latitude',
        type=float,
        help='Latitude of position',
    )
    declination_parser.add_argument(
        'longitude',
        type=float,
        help='Longitude of position',
    )
    declination_parser.set_defaults(cmd=cmd_declination)

    route_parser = subparsers.add_parser(
        'route',
        help='Build a navlog for a given route',
    )
    route_parser.add_argument(
        'departure',
        help='Departure point of flight',
    )
    route_parser.add_argument(
        'destination',
        help='Destination of flight',
    )
    route_parser.add_argument(
        'performance_profile',
        type=Path,
        help='Path to performance profile file',
    )
    route_parser.add_argument(
        'cruise_speed',
        type=int,
        help='Cruise speed in kts',
    )
    route_parser.add_argument(
        'departure_time',
        help='Time of departure',
    )
    route_parser.add_argument(
        'route',
        nargs='*',
        help='Route of flight',
    )
    route_parser.set_defaults(cmd=cmd_route)

    airport_parser = subparsers.add_parser(
        'airport-info',
        help='Build a navlog for a given airport',
    )
    airport_parser.add_argument(
        'airport',
        help='Airport code to get information for',
    )
    airport_parser.set_defaults(cmd=cmd_airport_info)

    return parser.parse_args()


def bounding_fields(value, rows):
    below_val = -float('inf')
    above_val = float('inf')
    for row_val in rows:
        if row_val > below_val and row_val <= value:
            below_val = row_val

        if row_val < above_val and row_val >= value:
            above_val = row_val
    return below_val, above_val


def lerp(x_min, x_max, x_cur, y_min, y_max):
    if x_max == x_min:
        return y_max
    pct = (x_cur - x_min) / (x_max - x_min)
    return pct * (y_max - y_min) + y_min


def cmd_perf(args):
    performance_data = defaultdict(dict)
    with args.performance_chart.open('r') as f:
        header = []
        for line in f.readlines():
            line = line.strip()
            if not header:
                header = line.split(',')
                continue
            row = {}
            parts = line.split(',')
            for k, v in zip(header[2:], parts[2:]):
                if not v:
                    row[k] = None
                    continue
                row[k] = int(v)
            performance_data[int(parts[0])][int(parts[1])] = row

    low_alt, high_alt = bounding_fields(
        args.altitude,
        performance_data.keys(),
    )
    if low_alt == -float('inf') or high_alt == float('inf'):
        print('Bad altitude supplied')
        return False

    altitude_calcs = []
    for alt in [low_alt, high_alt]:
        low_temp, high_temp = bounding_fields(
            args.temperature,
            performance_data[alt].keys(),
        )

        temperature_calcs = []
        for temp in [low_temp, high_temp]:
            field = performance_data[alt][temp][args.field]
            if field is None:
                print(
                    '{} at {} and {} does not exist, cannot calculate'.format(
                        args.field,
                        alt,
                        temp,
                    ),
                )
                return False
            temperature_calcs.append(field)
        alt_field = lerp(
            low_temp,
            high_temp,
            args.temperature,
            temperature_calcs[0],
            temperature_calcs[1],
        )
        altitude_calcs.append(alt_field)

    estimate = lerp(
        low_alt,
        high_alt,
        args.altitude,
        altitude_calcs[0],
        altitude_calcs[1],
    )

    estimate *= 1 + args.safety_margin / 100

    print(
        'Estimated {} at {} and {}deg with {}% safety margin is {}'.format(
            args.field,
            args.altitude,
            args.temperature,
            args.safety_margin,
            math.ceil(estimate),
        ),
    )

    return True


COLUMNS = [3000, 6000, 9000, 12000, 18000, 24000, 30000, 34000, 39000]


def parse_datapoint(chunk):
    if re.match(r'\s+$', chunk):
        return None

    chunk = chunk.strip()

    if chunk[:4] == '9900':
        datum = {
            'direction': 'variable',
            'velocity': 'light',
        }
    else:
        datum = {
            'direction': int(chunk[:2]) * 10,
            'velocity': int(chunk[2:4]),
        }

        if datum['direction'] >= 51 and datum['direction'] <= 86:
            datum['direction'] -= 50
            datum['velocity'] += 100

    if len(chunk) > 4:
        if len(chunk) == 6:
            temp = int(chunk[4:6])
        else:
            temp = int(chunk[5:7])
            if chunk[4] == '-':
                temp = -temp
        datum['temperature'] = temp

    return datum


def parse_winds_line(line):
    winds = {3000: parse_datapoint(line[4:8])}
    rest = line[9:]
    i = 0
    while rest:
        end = 7
        if i >= 5:
            end = 6
        winds[COLUMNS[i + 1]] = parse_datapoint(rest[:end])
        rest = rest[end + 1:]
        i += 1
    return line[:3], winds


ValidDate = namedtuple('ValidDate', ['date', 'time'])
ValidRange = namedtuple('ValidRange', ['start_time', 'end_time'])


def parse_valid_times(line):
    parts = re.split(r'\s+', line)
    return (
        ValidDate(int(parts[1][:2]), int(parts[1][2:-1])),
        ValidRange(*[int(t) for t in parts[4][:-2].split('-')]),
    )


def get_all_winds(offset=None):
    params = None
    if offset:
        params = {'fint': str(offset).zfill(2)}
    URL = 'http://aviationweather.gov/products/nws/all'
    soup = BeautifulSoup(
        requests.get(URL, params=params).text,
        'html.parser',
    )
    winds_data = soup.find('pre').text
    lines = winds_data.split('\n')
    valid_times = parse_valid_times(lines[5])
    winds_lines = lines[8:-1]
    output = {}
    for line in winds_lines:
        airport, winds = parse_winds_line(line)
        output[airport] = winds
    return valid_times, output


def offset_from_time(time):
    utc_now = datetime.datetime.now(pytz.utc)
    if time < utc_now:
        raise ValueError('Cannot get winds aloft data in the past')

    hours_between = (time - utc_now).total_seconds() / 3600
    if hours_between > 24:
        raise ValueError(
            'Cannot get winds aloft data more than one day in the future',
        )
    elif hours_between > 12:
        return 24
    elif hours_between > 6:
        return 12
    else:
        return 6


def parse_time(time):
    return datetime.datetime.strptime(
        time,
        '%Y-%m-%d %H%M%Z',
    ).astimezone(pytz.timezone('UTC'))


def cmd_winds(args):
    winds_time = parse_time(args.time)

    try:
        offset = offset_from_time(winds_time)
    except ValueError as e:
        print(str(e))
        return False

    valid_range, winds = get_all_winds(offset)
    airport = args.airport.upper()
    if airport.startswith('K') and len(airport) == 4:
        airport = airport[1:]

    if airport not in winds:
        print('No winds found for {}'.format(args.airport.upper()))
        return False

    keys = bounding_fields(args.altitude, winds[airport].keys())
    rows = []
    for key in keys:
        data = winds[airport][key]
        if data is None:
            rows.append([key, '-', '-', '-'])
            continue

        rows.append([
            key,
            data.get('direction', '-'),
            data.get('velocity', '-'),
            data.get('temperature', '-'),
        ])

    print(
        tabulate(
            rows,
            headers=['altitude', 'direction', 'velocity', 'temperature'],
        ),
    )

    return True


def declination(latitude, longitude):
    URL = (
        'http://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination'
    )
    params = {
        'lat1': latitude,
        'lon1': longitude,
        'resultFormat': 'xml',
        'startMonth': datetime.datetime.now().month,
    }
    tree = ElementTree.fromstring(requests.get(URL, params=params).text)
    return float(tree.find('result/declination').text)


def cmd_declination(args):
    print(declination(args.latitude, args.longitude))

    return True


Stop = namedtuple(
    'Stop',
    ['location', 'altitude', 'speed', 'latitude', 'longitude'],
)


class Stop:

    def __init__(self, location, altitude, speed):
        self.location = location
        self.altitude = altitude
        self.speed = speed

    def __str__(self):
        ret = 'location {} altitude {} speed {}'.format(
            self.location,
            self.altitude,
            self.speed,
        )
        for prop in ['latitude', 'longitude']:
            if hasattr(self, prop):
                ret += ' {} {}'.format(prop, getattr(self, prop))
        return ret


def parse_stop(stop):
    if '/' not in stop:
        return Stop(location=stop)

    parts = stop.split('/', 1)

    altitude = None
    speed = None
    m = re.search(r'A(\d+)', stop)
    if m:
        altitude = int(m.groups()[0]) * 100
    m = re.search(r'N(\d+)', stop)
    if m:
        speed = int(m.groups()[0])
    return Stop(parts[0], altitude, speed)


RoutePart = namedtuple(
    'RoutePart',
    [
        'altitude',
        'attitude',
        'wind',
        'true_airspeed',
        'compass_heading',
        'distance',
        'groundspeed',
        'estimated_time_enroute',
        'fuel',
    ],
)


class Winds:

    def __init__(self, true_course, wind_correction_angle):
        self.true_course = true_course
        self.wind_correction_angle = wind_correction_angle
        self.declination = None
        self.compass_correction = 0

    @property
    def compass_heading(self):
        if self.declination is None:
            raise ValueError('Must have declination set')
        return self.true_course + self.wind_correction_angle + self.declination


def calculate_winds(bearing, speed, altitude, winds_aloft):
    low, hi = bounding_fields(altitude, winds_aloft.keys())
    if altitude - low < altitude - hi:
        choice = low
    else:
        choice = hi
    winds = winds_aloft[choice]
    wind_angle = math.radians(bearing - winds['direction'])
    speed_component = -math.sin(wind_angle) * winds['velocity']
    crosswind_correction = math.degrees(
        math.asin(
            winds['velocity'] * math.sin(wind_angle) / speed
        )
    )
    opposite_direction = bearing - 180
    wind_dir = winds['direction']
    if wind_dir < bearing and wind_dir > opposite_direction:
        crosswind_correction = -crosswind_correction
    groundspeed = speed - speed_component
    return groundspeed, winds, Winds(bearing, crosswind_correction)


def format_winds(winds):
    ret = 'dir {} vel {}'.format(winds['direction'], winds['velocity'])
    if 'temperature' in winds:
        ret += ' temp {}'.format(winds['temperature'])
    return ret


def temperature_from_metar(station):
    URL = 'http://tgftp.nws.noaa.gov/data/observations/metar/stations/{}.TXT'
    body = requests.get(URL.format(station.upper())).text
    m = re.search(r'\b(M?\d+/M?\d+)\b', body.split('\n')[1])
    if m is None:
        raise RuntimeError(
            'Unable to get METAR for {}'.format(station.upper()),
        )

    temperature_str, dewpoint_str = m.group(0).split('/')
    if temperature_str[0] == 'M':
        return -int(temperature_str[1:])
    return int(temperature_str)


def nearest_winds_aloft(airport_keys, latitude, longitude):
    sql = (
        "SELECT ST_Distance(ST_GeogFromText(%s), point) "
        'AS distance, icao_identifier FROM airports_airport '
        'WHERE icao_identifier IN ({}) ORDER BY distance ASC LIMIT 1'
        .format(', '.join(['%s'] * len(airport_keys)))
    )
    conn = psycopg2.connect(
        host=config.postgres['host'],
        port=config.postgres['port'],
        database=config.postgres['database'],
        user=config.postgres['username'],
        password=config.postgres['password'],
    )
    cur = conn.cursor()
    cur.execute(
        sql,
        ['POINT({} {})'.format(longitude, latitude)] + airport_keys,
    )
    # Strip the 'K' prefix
    key = cur.fetchone()[1][1:]
    conn.close()
    return key


def skyvector_link(route):
    lat_sum = 0
    lon_sum = 0
    route_str = ''
    for stop in route:
        lat_sum += stop.latitude
        lon_sum += stop.longitude
        route_str += '%20' + stop.location
        extra_info = ''
        if stop.altitude:
            extra_info += 'A{}'.format(int(stop.altitude / 100))
        if stop.speed:
            extra_info += 'N{}'.format(int(stop.speed))
        if extra_info:
            route_str += '/' + extra_info

    return 'https://skyvector.com/?ll={},{}&chart=301&fpl={}'.format(
        lat_sum / len(route),
        lon_sum / len(route),
        route_str,
    )


def cmd_route(args):
    departure_time = parse_time(args.departure_time)

    try:
        offset = offset_from_time(departure_time)
    except ValueError as e:
        print(str(e))
        return False

    _, winds = get_all_winds(offset)

    with args.performance_profile.open('r') as f:
        perf_data = json.load(f)

    stops = [Stop(args.departure.upper(), altitude=0, speed=0)]
    stops.extend([parse_stop(stop.upper()) for stop in args.route])
    stops.append(Stop(args.destination.upper(), altitude=0, speed=0))

    airports = get_airports()
    rows = []
    for stop in stops:
        if 'VP' in stop.location:
            pos = get_waypoint(stop.location)
            stop.latitude = pos['latitude']
            stop.longitude = pos['longitude']
            continue

        if stop.location not in airports:
            print('unknown stop {}'.format(stop.location))
            continue

        data = airports[stop.location]
        stop.latitude = data['latitude']
        stop.longitude = data['longitude']
        rows.append([
            stop.location,
            stop.altitude,
            stop.speed,
        ])

    total_distance = 0
    total_fuel = 0
    parts = []
    stop_rows = []
    for i in range(1, len(stops)):
        prev = stops[i - 1]
        cur = stops[i]
        distance = distance_between_points(
            prev.latitude,
            prev.longitude,
            cur.latitude,
            cur.longitude,
        )
        bearing = bearing_between_points(
            prev.latitude,
            prev.longitude,
            cur.latitude,
            cur.longitude,
        )
        if cur.altitude > prev.altitude:
            attitude = 'climb'
        elif cur.altitude < prev.altitude:
            attitude = 'descend'
        else:
            attitude = 'level'

        if cur.speed:
            speed = cur.speed
        else:
            if attitude == 'climb':
                if prev.altitude == 0:
                    speed = perf_data['vy_climb_speed']
                else:
                    speed = perf_data['cruise_climb_speed']
            else:
                speed = args.cruise_speed

        winds_aloft_airport = nearest_winds_aloft(
            ['K{}'.format(k) for k in winds],
            stop.latitude,
            stop.longitude,
        )

        groundspeed, selected_winds, part_winds = calculate_winds(
            bearing,
            speed,
            cur.altitude,
            winds[winds_aloft_airport],
        )

        if 'temperature' not in selected_winds:
            for location in [cur.location, prev.location]:
                try:
                    selected_winds['temperature'] = temperature_from_metar(
                        location,
                    )
                    break
                except Exception:
                    pass

        part_winds.declination = declination(
            abs(cur.latitude - prev.latitude) / 2,
            abs(cur.longitude - prev.longitude) / 2,
        )

        ete = distance / groundspeed
        if attitude == 'climb':
            fuel = ete * perf_data['climb_fuel_burn']
        else:
            fuel = ete * perf_data['cruise_fuel_burn']

        total_distance += distance
        total_fuel += fuel

        parts.append(
            RoutePart(
                cur.altitude,
                attitude,
                selected_winds,
                speed,
                part_winds.compass_heading,
                distance,
                groundspeed,
                ete,
                fuel,
            ),
        )
        stop_rows.append([
            prev.location,
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
        ])
        stop_rows.append([
            '├───────',
            cur.altitude,
            attitude,
            format_winds(selected_winds),
            '{}kts'.format(speed),
            '{}deg'.format(round(part_winds.compass_heading)),
            '{}nm'.format(math.ceil(distance)),
            '{}kts'.format(math.ceil(groundspeed)),
            '{}min'.format(math.ceil(ete * 60)),
            '{}gal'.format(math.ceil(fuel)),
        ])

    stop_rows.append([
        stops[-1].location,
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
    ])

    if len(stops) != len(parts) + 1:
        print(
            'Bad route calculation, expected one more stop than parts but got '
            '{} stops and {} parts'.format(len(stops), len(parts)),
        )
        return False

    print(
        tabulate(
            stop_rows,
            headers=[
                'location',
                'altitude',
                'attitude',
                'winds',
                'tas',
                'compass_heading',
                'distance',
                'groundspeed',
                'ete',
                'fuel',
            ],
        ),
    )

    print(
        'total distance: {}nm, total fuel: {}gal'.format(
            math.ceil(total_distance),
            math.ceil(total_fuel),
        ),
    )

    print(skyvector_link(stops))

    return True


def get_waypoint(waypoint):
    URL = 'https://nfdc.faa.gov/nfdcApps/services/ajv5/fix_search.jsp'
    soup = BeautifulSoup(
        requests.get(URL, params={'keyword': waypoint}).text,
        'html.parser',
    )
    lat = False
    lon = False
    point = {}
    for td in soup.find('table').find_all('td'):
        if 'Latitude:' in td.text:
            lat = True
            continue
        if lat:
            text = td.text.strip()
            parts = text.split(' ')
            lat_parts = parts[0].split('-')
            point['latitude'] = int(lat_parts[0]) + float(lat_parts[1]) / 60
            point['latitude'] += float(lat_parts[2]) / 3600
            if text[-1] == 'S':
                point['latitude'] = -point['latitude']
            lat = False
            continue

        if 'Longitude:' in td.text:
            lon = True
            continue
        if lon:
            text = td.text.strip()
            parts = text.split(' ')
            lon_parts = parts[0].split('-')
            point['longitude'] = int(lon_parts[0]) + float(lon_parts[1]) / 60
            point['longitude'] += float(lon_parts[2]) / 3600
            if text[-1] == 'W':
                point['longitude'] = -point['longitude']
            lon = False
            continue
    return point


def get_airports():
    path = Path('NfdcFacilities.xls')
    with path.open('r') as f:
        lines = f.readlines()

    headers = [h.replace('"', '') for h in lines[0].strip().split('\t')]
    airports = {}
    for line in lines[1:]:
        data = dict(zip(headers, line.strip().split('\t')))
        if data.get('IcaoIdentifier', None):
            ident = data['IcaoIdentifier']
        else:
            ident = data['LocationID'][1:]
        data['latitude'] = float(data['ARPLatitudeS'][:-1]) / 3600
        if data['ARPLatitudeS'][-1] == 'S':
            data['latitude'] = -data['latitude']
        data['longitude'] = float(data['ARPLongitudeS'][:-1]) / 3600
        if data['ARPLongitudeS'][-1] == 'W':
            data['longitude'] = -data['longitude']
        airports[ident] = data

    return airports


def distance_between_points(lat1, lon1, lat2, lon2):
    EARTH_RADIUS = 3440
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2 +
        math.cos(lat1_rad) * math.cos(lat2_rad) *
        math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS * c


def bearing_between_points(lat1, lon1, lat2, lon2):
    bearing = math.atan2(
        math.sin(lon2 - lon1) * math.cos(lat2),
        (
            math.cos(lat1) * math.sin(lat2) - math.sin(lat1)
            * math.cos(lat2) * math.cos(lon2 - lon1)
        ),
    )
    bearing = math.degrees(bearing)
    return (bearing + 360) % 360


def cmd_airport_info(args):
    airports = get_airports()
    print(airports[args.airport.upper()])

    return True


def main():
    args = parse_args()
    return args.cmd(args)


sys.exit(0 if main() else 1)
