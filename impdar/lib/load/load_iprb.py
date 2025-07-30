#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 David Lilien <dlilien90@gmail.com>
#
# Distributed under terms of the GNU GPL3.0 license.

"""
Load Impulse Radar iprb data
Written by Benjamin Hills <benjaminhhills@gmail.com>
July 30 2025
"""

import os
import struct
import datetime
import numpy as np
from scipy.interpolate import interp1d
from ..RadarData import RadarData
from ..gpslib import nmea_info, conversions_enabled


def load_iprb(iprb_fn):
    """Return a impdar.lib.RadarData.RadarData object loaded from an
    Impulse radar iprb file. "Impulse" hear not to be confused with
    the name for ImpDAR, just a coincidence that they use that name.

    Parameters
    ----------
    iprb_fn: str
        The file to load. Can include the '.iprb' or not.
    """
    iprb_data = RadarData(None)

    # We can take headers, base filenames, or data files
    if len(iprb_fn) <= 4:
        header_fn = iprb_fn + '.iprh'
        data_fn = iprb_fn + '.iprb'
        gps_fn = iprb_fn + '.cor'
    elif iprb_fn[-5:] == '.iprb':
        header_fn = iprb_fn[:-4] + 'iprh'
        data_fn = iprb_fn[:-4] + 'iprb'
        gps_fn = iprb_fn[:-4] + 'cor'
    else:
        header_fn = iprb_fn
        data_fn = iprb_fn[:-4] + 'iprb'
        gps_fn = iprb_fn[:-4] + 'cor'

    iprb_data.fn = data_fn

    # read from header file

    with open(header_fn, 'r') as f_header:
        header_list = [line.strip() for line in f_header if line.strip()]
    header = {}
    for line in header_list:
        hdline = line.split(":")
        header[hdline[0].strip()] = hdline[1].strip()

    iprb_data.chan = iprb_fn[-6]
    iprb_data.snum = int(header['SAMPLES'])
    sampling_freq = float(header['FREQUENCY'])
    iprb_data.dt = (1. / sampling_freq) * 1.0e-6
    iprb_data.travel_time = iprb_data.dt * np.arange(iprb_data.snum) * 1.0e6
    iprb_data.tnum = int(header['LAST TRACE'])
    iprb_data.trace_num = np.arange(iprb_data.tnum) + 1

    # FIX TRACE INT
    iprb_data.trace_int = float(header['TIME INTERVAL']) * np.ones(
        (iprb_data.tnum,))
    iprb_data.trig = np.ones((iprb_data.tnum,)) * 36
    iprb_data.trig_level = 0

    # read from gps file

    if os.path.exists(gps_fn):
        cor = np.genfromtxt(gps_fn, dtype=[('trace_num', int),
                                           ('date', 'S10'),
                                           ('time', 'S8'),
                                           ('lat', float),
                                           ('north', 'S1'),
                                           ('lon', float),
                                           ('east', 'S1'),
                                           ('elev', float),
                                           ('el_unit', 'S1'),
                                           ('pdop', float)])
        datetimes = np.array([d + b'T' + t for d, t in zip(
            cor['date'], cor['time'])], dtype=np.datetime64)
        decdays = datetimes - np.array(datetime.datetime(1, 1, 1, 0, 0, 0),
                                       dtype=np.datetime64)
        cor['lat'][cor['north'] != b'N'] = -1 * cor[
            'lat'][cor['north'] != b'N']
        cor['lon'][cor['east'] != b'E'] = -1 * cor['lon'][cor['east'] != b'E']

        iprb_data.decday = interp1d(cor['trace_num'],
                                     decdays,
                                     fill_value='extrapolate')(
            iprb_data.trace_num) / (24. * 60. * 60.)
        iprb_data.lat = interp1d(cor['trace_num'],
                                  cor['lat'],
                                  fill_value='extrapolate')(
            iprb_data.trace_num)
        iprb_data.long = interp1d(cor['trace_num'],
                                   cor['lon'],
                                   fill_value='extrapolate')(
            iprb_data.trace_num)
        iprb_data.elev = interp1d(cor['trace_num'],
                                   cor['elev'],
                                   fill_value='extrapolate')(
            iprb_data.trace_num)

        nminfo = nmea_info()
        nminfo.time = iprb_data.decday
        nminfo.lat = iprb_data.lat
        nminfo.lon = iprb_data.long
        nminfo.elev = iprb_data.elev
        if conversions_enabled:
            nminfo.get_utm()
            nminfo.get_dist()
            iprb_data.x_coord = nminfo.x
            iprb_data.y_coord = nminfo.y
            iprb_data.dist = nminfo.dist
        else:
            iprb_data.x_coord = iprb_data.long
            iprb_data.y_coord = iprb_data.lat
            iprb_data.dist = np.sqrt(
                iprb_data.x_coord ** 2.0 + iprb_data.y_coord ** 2.0) / 1000.0

    else:
        iprb_data.decday = np.arange(iprb_data.tnum)
        iprb_data.lat = np.arange(iprb_data.tnum)
        iprb_data.long = np.arange(iprb_data.tnum)
        iprb_data.dist = np.arange(iprb_data.tnum)
        iprb_data.elev = np.arange(iprb_data.tnum)
    iprb_data.pressure = np.zeros_like(iprb_data.dist)

    with open(data_fn, 'rb') as f_data:
        iprb_data.data = np.array(
            struct.unpack('<{:d}i'.format(
                iprb_data.tnum * iprb_data.snum), f_data.read()[:]),
            dtype=np.int32).reshape((iprb_data.snum, iprb_data.tnum),
                                    order='F')
    iprb_data.check_attrs()
    return iprb_data
