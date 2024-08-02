# %%
#!./.venv/bin python3
# -*- encoding: utf-8 -*-
"""
@filename :  tec_inversion_GEO.py
@desc     :  extract and inverse TEC data(1Hz) from RINEX files for each day
@time     :  2024/06/01
@author   :  _koii^_, IEF, CEA.
@Version  :  1.0
@Contact  :  koi_alley@outlook.com
"""

# here import libs
import gc
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from pathlib import Path

import georinex as grnx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import scipy.constants as const

# import tomli_w as tomlw
import tomllib as toml
import xarray as xr
from matplotlib import font_manager as fm


# %%
# -*- prepare directories and files -*-
#  *  存储路径与文件准备
def mk_archives(sourceDir: str, archives: list[str]) -> list[Path]:
    """create list(archive_paths) in source directory, no cover forced

    Args:
        sourceDir (str): source directory
        archives (list[str]): str_archive list

    Returns:
        list[Path]: path_archive list
    """
    dirs = {}
    for archive in archives:
        dirs[archive] = Path(sourceDir, archive).mkdir(
            mode=0o777, parents=True, exist_ok=True
        )
    return dirs


# %%
# -*- lla2ecef ecef2lla -*-
#  *  在地理坐标系lla(longitude,latitude,altitude)和地心地固空间坐标系ecef(x,y,z)之间相互转换
#  *  基于 WGS 84 地理坐标系统
def lla2ecef(lon: float, lat: float, alt: float) -> tuple[float, float, float]:
    lla = {'proj': 'latlong', 'ellps': 'WGS84', 'datum': 'WGS84'}
    xyz = {'proj': 'geocent', 'ellps': 'WGS84', 'datum': 'WGS84'}
    transformer = pyproj.Transformer.from_crs(lla, xyz, always_xy=True)
    return transformer.transform(lon, lat, alt, radians=False)


def ecef2lla(x: float, y: float, z: float) -> tuple[float, float, float]:
    lla = {'proj': 'latlong', 'ellps': 'WGS84', 'datum': 'WGS84'}
    xyz = {'proj': 'geocent', 'ellps': 'WGS84', 'datum': 'WGS84'}
    transformer = pyproj.Transformer.from_crs(xyz, lla, always_xy=True)
    return transformer.transform(x, y, z, radians=False)


# %%
# -*-  -*-
def calendarOf(year: int, doy: int) -> list:
    """

    Args:
        year (int): year
        doy (int): day of year

    Returns:
        list: calendar
    """
    base = datetime(int(year), 1, 1)
    the_day = base + timedelta(int(doy) - 1)
    calendar = [
        the_day.year,
        the_day.month,
        the_day.day,
        the_day.hour,
        the_day.minute,
        the_day.second,
        # 23,59,59
    ]
    return calendar


def modified_JulianDay(calendar: list) -> float:
    """

    Args:
        calendar (list): [year, month, day, hour, minute, second]

    Returns:
        float: modified_JulianDay
    """

    if len(calendar) < 6:
        for i in range(len(calendar), 6):
            calendar.append(0)
    year = calendar[0]
    month = calendar[1]
    day = calendar[2] + (calendar[3] * 3600 + calendar[4] * 60 + calendar[5]) / 86400
    yr = year + 4800
    mo = month
    if year < 0:
        print('Year is wrong')
        return False

    if mo <= 2:
        # Jan and Feb treated as 13th, 14th month of last year
        mo += 12
        yr -= 1

    shiftA = np.floor(30.6 * (mo + 1))
    century = np.floor(yr / 100)
    # remained historic issue (1582/2/24), lead to:
    # the day of 1582/10/05~14 was dropped
    if (
        (year < 1582)
        or (year == 1582 and month < 10)
        or (year == 1582 and month == 10 and day < 15)
    ):
        shiftB = -38
    else:
        shiftB = np.floor((century / 4) - century)
    shiftC = np.floor(365.25 * yr)
    julianDay = shiftA + shiftB + shiftC + day - 32167.5  # julian day
    mjd = julianDay - 2400000.5  # Modified Julian Day
    return mjd


def gpsWeekOf(year: int, doy: int) -> tuple[np.int64, np.int64, np.int64]:
    """

    Args:
        year (int): year
        doy (int): day of year

    Returns:
        tuple[np.int64, np.int64, np.int64]:
            week,
            seconds accumulated in the week,
            day of week starting from 0
    """
    calendar = calendarOf(year, doy)
    mjd = modified_JulianDay(calendar)
    # GPS week mode starts from the mjd of 44244
    eDay = mjd - 44244
    week = np.floor(eDay / 7).astype(np.int64)
    eDay = (eDay - week * 7).astype(np.int64)
    seconds = (eDay * 86400).astype(np.int64)
    return week, seconds, eDay


# %%
# -*- ippsProjection -*-
#  *  根据GNSS站点位置与卫星位置,计算出穿刺点位置和卫星在穿刺点处的天顶角
#  *  电离层薄壳假设
#  *  假定电离层高度为350km
def ippsProjection_CartesianSpace(
    xyzObs: tuple[float, float, float],
    xyzSat: tuple[float, float, float],
    HIONOS: float = 350 * const.kilo,
) -> tuple[float, list[float, float, float]]:
    """calculate the intersaction point(ipp) of receiver-satellite path with thin ionosphere,
        vector caculation proceeding in Cartesian space.

    Args:
        xyzObs (tuple[float, float, float]): in unit of m, EarchCentric EarthFixed geographic coords
        xyzSat (tuple[float, float, float]): in unit of m, EarchCentric EarthFixed geographic coords
        HIONOS (float, optional): Defaults to 350 km, the height of ionospheric thin layer

    Returns:
        tuple[float, list[float, float, float]]:
            elevation angle:degree, (lon:degree, lat:degree, altitude:m)
    """
    REARTH = 6371 * const.kilo

    xObs, yObs, zObs = xyzObs
    xSat, ySat, zSat = xyzSat
    vectorObs = np.array([xObs, yObs, zObs])
    vectorSat = np.array([xSat, ySat, zSat])

    rObs = np.sqrt(np.dot(vectorObs, vectorObs))
    eVectorObs = vectorObs / rObs  # cannot devid by zero

    dVector = vectorSat - vectorObs

    r_dVector = np.sqrt(np.dot(dVector, dVector))
    e_dVector = dVector / r_dVector

    zenObs = np.arccos(np.dot(eVectorObs, e_dVector))
    zenipp = np.arcsin(REARTH / (REARTH + HIONOS) * np.sin(zenObs))
    eleipp = 90.0 - np.degrees(zenipp)
    Delta = zenObs - zenipp

    r_obs2ipp = REARTH * np.sin(Delta) / np.sin(zenipp)

    xyzipp = np.array(vectorObs + r_obs2ipp * e_dVector)
    llaipp = ecef2lla(xyzipp[0], xyzipp[1], xyzipp[2])

    return eleipp, [llaipp]


# %%
# -*- ippsProjection -*-
def ippsProjection(
    xyzObs: tuple[float, float, float],
    xyzSat: tuple[float, float, float],
    HIONOS: float = 350 * const.kilo,
) -> tuple[float, list[float, float, float]]:
    """(RECOMMANDED)
        calculate the intersaction point(ipp) of receiver-satellite path with thin ionosphere,
        vector caculation proceeding in Geographic Coords (spherical space).

    Args:
        xyzObs (tuple[float, float, float]): in unit of m, EarchCentric EarthFixed geographic coords
        xyzSat (tuple[float, float, float]): in unit of m, EarchCentric EarthFixed geographic coords
        HIONOS (float, optional): Defaults to 350 km, the height of ionospheric thin layer

    Returns:
        tuple[float, list[float, float, float]]:
            elevation angle:degree, (lon:degree, lat:degree, altitude:m)
    """
    REARTH = 6371 * 1000

    xObs, yObs, zObs = xyzObs
    xSat, ySat, zSat = xyzSat

    dx, dy, dz = xSat - xObs, ySat - yObs, zSat - zObs
    dv = np.array([dx, dy, dz])
    dl = np.sqrt(np.dot(dv, dv))

    ix, iy, iz = dv / dl
    iv = np.array([ix, iy, iz])

    lonObs, latObs, _ = ecef2lla(xObs, yObs, zObs)
    rlonObs, rlatObs = np.radians(lonObs), np.radians(latObs)

    north = np.array(
        [
            -np.cos(rlonObs) * np.sin(rlatObs),
            -np.sin(rlonObs) * np.sin(rlatObs),
            np.cos(rlatObs),
        ]
    )
    east = np.array([-np.sin(rlonObs), np.cos(rlonObs), 0.0])
    vertical = np.array(
        [
            np.cos(rlonObs) * np.cos(rlatObs),
            np.sin(rlonObs) * np.cos(rlatObs),
            np.sin(rlatObs),
        ]
    )

    zen = np.arccos(np.dot(vertical, iv))
    ele = np.pi / 2.0 - zen
    azi = np.arctan2(np.dot(east, iv), np.dot(north, iv))

    if azi < 0:
        azi += 2 * np.pi

    a = zen - np.arcsin(REARTH / (REARTH + HIONOS) * np.cos(ele))

    latipp = np.arcsin(
        np.sin(rlatObs) * np.cos(a) + np.cos(rlatObs) * np.sin(a) * np.cos(azi)
    )
    lonipp = rlonObs + np.arcsin(np.sin(a) * np.sin(azi) / np.cos(latipp))

    latipp, lonipp, ele = np.degrees(latipp), np.degrees(lonipp), np.degrees(ele)

    return ele, lonipp, latipp, HIONOS


# %%
# -*- calculate slant Total electron content -*-
#  *  基于双频观测组合TEC解算原理
#  *  使用相位平滑伪距TEC (消除多径效应)
def cycleSlip_repair_TECR(
    dPhase: xr.Dataset,
    intervals: np.ndarray,
    stecRate: np.ndarray,
    COEF: float,
    GAMMA: float,
    FRQC1: float,
) -> np.ndarray:
    for index in range(17, len(dPhase)):
        if (
            intervals[index] > 20 * 60
            or np.abs(stecRate[index] * intervals[index]) > 25
        ):
            drate = 0
            estRate = 0  # if interval > 10min or |dTEC| > 25: estRate defaults to 0
        else:
            # for 30s data
            # rate_sort = np.sort(stecRate[index - 1 - 5 : index])
            # rateForward1 = rate_sort[1:5].mean()
            # rate_sort = np.sort(stecRate[index - 2 - 5 : (index - 1)])
            # rateForward2 = rate_sort[1:5].mean()
            # interval_sort = np.sort(intervals[index - 1 - 5 : index])
            # intervalForward1 = interval_sort[1:5].mean()

            rate_sort = np.sort(stecRate[index - 1 - 15 : index])
            rateForward1 = rate_sort[5:11].mean()
            rate_sort = np.sort(stecRate[index - 2 - 15 : (index - 1)])
            rateForward2 = rate_sort[5:11].mean()
            interval_sort = np.sort(intervals[index - 1 - 15 : index])
            intervalForward1 = interval_sort[5:11].mean()

            drate = (rateForward1 - rateForward2) / intervalForward1 * intervals[index]
            estRate = rateForward1 + drate

        if np.abs(stecRate[index] - estRate) > 0.15:
            cycle_slip = (
                COEF * (GAMMA - 1) * intervals[index] * estRate / FRQC1**2
                - dPhase[index]
                + dPhase[index - 1]
            )
            stecRate[index] = estRate
            dPhase[index:] = dPhase[index:] + cycle_slip
    return dPhase


def relative_slantTEC(
    FRQC1: float,
    FRQC2: float,
    pseudoRange1: xr.Dataset,
    pseudoRange2: xr.Dataset,
    carrierPhase1: xr.Dataset,
    carrierPhase2: xr.Dataset,
) -> np.ndarray:
    """the TEC(total electron content) integrated along slant receiver-to-satellite path
        through dual-frequency range and phase combined measurement

    Args:
        FRQC1 (float): frequency of signal1 emitted by satellite
        FRQC2 (float): frequency of signal2 emitted by satellite
        pseudoRange1 (xr.Dataset): pseudo range measurement of signal1
        pseudoRange2 (xr.Dataset): pseudo range measurement of signal2
        carrierPhase1 (xr.Dataset): phase measurement of signal1
        carrierPhase2 (xr.Dataset): phase measurement of signal2
        sample_period (float, optional): Defaults to 1.0, the same as sample rate of data.

    Returns:
        np.ndarray: slant TEC values
    """

    TECU = 1.00e16
    COEF = const.e**2 / (8 * const.pi**2 * const.epsilon_0 * const.electron_mass) * TECU
    COEF = 1 / COEF * (FRQC1 * FRQC2) ** 2 / (FRQC1**2 - FRQC2**2)
    GAMMA = (FRQC1 / FRQC2) ** 2

    waveLEN1 = const.speed_of_light / FRQC1
    waveLEN2 = const.speed_of_light / FRQC2

    dRange = pseudoRange1 - pseudoRange2  # unit: m
    dPhase = carrierPhase1 * waveLEN1 - carrierPhase2 * waveLEN2  # unit: m
    # stecRange = - COEF * dRange   # unit: TECU
    stecPhase = COEF * dPhase  # unit: TECU

    time = dPhase['time'].values
    intervals = (time[1:] - time[:-1]) / np.timedelta64(1, 's')
    stecRate = (stecPhase.values[1:] - stecPhase.values[:-1]) / intervals
    # stecRate = np.convolve(np.array(stecPhase), np.array([1, -1]), 'valid') / intervals
    intervals = np.hstack((intervals[0], intervals))
    stecRate = np.hstack((stecRate[0], stecRate))

    dPhase = cycleSlip_repair_TECR(
        dPhase, intervals, np.array(stecRate), COEF, GAMMA, FRQC1
    )
    combineIONOS = dRange + dPhase  # remove N_a and DPB
    dPhase_smth = np.mean(combineIONOS) - dPhase

    slantTEC = -COEF * dPhase_smth
    return slantTEC


# %%
# -*- calculate Differential Code Bias -*-
def sysmDCBs_byiri2020(
    stec: xr.Dataset,
    ipplon: float,
    ipplat: float,
    elevation: float,
    sat_sysm: str = 'BeiDou',
    to_plot: bool = False,
) -> np.float64:
    """Diffrential Code Bias correction utilizing IRI2020 ionosphere model

    Args:
        stec (xr.Dataset): sTEC data set with dimension(lon, lat, time)
        ipplon (float): longitude of IPP
        ipplat (float): latitude of IPP
        elevation (float): elevation angle in degrees from receiver to satellite
        sat_sysm (str, optional): Defaults to 'BeiDou'.
        to_plot (bool, optional): Defaults to False, not to plot DCB results

    Returns:
        np.float64: mean value of nighttime DCBs to represent constant DCBs of a day
    """
    from iricore import update as iri_update
    from iricore import vtec as iritec

    iri_update()
    sv = stec.sv.values
    description = (
        f':::   {sat_sysm} - {sv} : calculate the SPR DCBs of observing station... \n'
    )
    print(description)

    time = stec.time
    t_range = pd.date_range(time[0].values, periods=48, freq='30Min')

    glon, glat = ipplon, ipplat
    itecs, tnodes, stecs = np.array([]), np.array([]), np.array([])
    for t in t_range:
        dt = t.to_pydatetime()

        ipplon -= 360.0 if ipplon > 180 else ipplon
        IPP_tLocal = ipplon / 15 + dt.hour
        IPP_tLocal += 24.0 if IPP_tLocal < 0.0 else IPP_tLocal
        IPP_tLocal -= 24.0 if IPP_tLocal > 24.0 else IPP_tLocal

        if IPP_tLocal >= 23.0 or IPP_tLocal < 5.0:
            pass
        else:
            continue
        itec = iritec(dt, glat, glon)
        stec_select = stec.sel(time=t, method='nearest')
        tnodes = (np.hstack((tnodes, t)),)
        itecs, stecs = np.hstack((itecs, itec)), np.hstack((stecs, stec_select))

    DCBs = stecs - itecs / np.sin(np.radians(elevation))

    ## plot test
    if not to_plot:
        return np.mean(DCBs)
    else:
        stecs = stec.sel(time=t_range, method='nearest')
        ax = plt.figure(figsize=(6, 5)).add_subplot(111)
        ax.plot(t_range, itecs, color='b', marker=',', linestyle='-', label='iristec')
        ax.plot(
            t_range,
            (stecs - DCBs.mean()) * np.sin(np.radians(elevation)),
            color='c',
            marker=',',
            linestyle='-',
            label='vtec_corrected',
        )
        ax.axhline(DCBs.mean(), color='r', label='DCBs')

        plt.legend(labelcolor='linecolor')
        plt.show()
        return np.mean(DCBs)


# %%
def sysmDCBs_byGimtec(
    stec: xr.Dataset,
    ipplon: float,
    ipplat: float,
    elevation: float,
    gimnc: str,
    # satsys: str = 'BeiDou',
    to_plot: bool = False,
) -> np.float64:
    """(RECOMMANDED) Diffrential Code Bias correction referring to
        Global Ionosphere Map defaults from CODE,

    Args:
        stec (xr.Dataset): sTEC data set with dimension(lon, lat, time)
        ipplon (float): longitude of IPP
        ipplat (float): latitude of IPP
        elevation (float): elevation angle in degrees from receiver to satellite
        fimnc (str): the path of gim tec file in nc format
        to_plot (bool, optional): Defaults to False, not to plot DCB results

    Returns:
        np.float64: mean value of nighttime DCBs to represent constant DCBs of a day
    """
    # sv = stec.sv.values
    gimtec = xr.open_dataarray(gimnc)
    gtec = gimtec.interp(lat=ipplat, lon=ipplon, method='quadratic')
    t_range = pd.date_range(stec.time[0].values, periods=24 * 3, freq='20Min')
    gtime = np.array([])
    gtecs = np.array([])
    stecs = np.array([])
    for tnode in t_range:
        pydt = tnode.to_pydatetime()

        if ipplon > 180:
            ipplon -= 360.0

        tLocal = ipplon / 15.0 + pydt.hour

        if tLocal % 24 >= 23.0 or tLocal % 24 < 5.0:
            pass
        else:
            continue
        gtecNode = gtec.sel(time=tnode, method='nearest').values
        stecNode = stec.sel(time=tnode, method='nearest').values
        gtime = np.hstack((gtime, tnode))
        gtecs = np.hstack((gtecs, gtecNode))
        stecs = np.hstack((stecs, stecNode))

    DCBs = stecs - gtecs / np.sin(np.radians(elevation))

    ### plot test
    if not to_plot:
        return DCBs.mean()
    else:
        ax = plt.figure(figsize=(6, 5)).add_subplot(111)
        ax.plot(gtime, gtecs, color='b', marker=',', linestyle='-', label='gimtec')
        ax.plot(
            gtime,
            (stecs - DCBs.mean()) * np.sin(np.radians(elevation)),
            color='c',
            marker=',',
            linestyle='-',
            label='vtec_corrected',
        )
        ax.axhline(DCBs.mean(), color='r', label='DCBs')

        plt.legend(labelcolor='linecolor')
        plt.show()

        return DCBs.mean()


# DCBs
# test()


# %%
# -*- build up Dataset
def struct_TECs_of_stations(
    TECs: np.ndarray,
    time: np.ndarray,
    nameObs: str,
    sv: str,
    lonipp: float,
    latipp: float,
    elevation: float,
    # lonObs: float,
    # latObs: float,
    # altObs: float,
    llaSat: dict,
    DCBs: np.float64,
) -> xr.Dataset:
    """collect diverse datas and basic attributes into a Data set

    Args:
        TECs (np.ndarray): slant phase TEC (relative)
        time (np.ndarray): coordinate time
        nameObs (str): station name of receiver (observer)
        sv (str): satellite vehicle code
        lonipp (float): coordinate longitude of ipp
        latipp (float): coordinate latitude of ipp
        elevation (float): elevation angle in degree from receiver to satellite
        lonObs (float): coordinate longitude of receiver
        latObs (float): coordinate latitude of receiver
        altObs (float): altitude of receiver
        llaSat (dict): satellite geo positions of the day
        DCBs (np.float64): Differential Code Bias of satellite plus receiver

    Returns:
        xr.Dataset: complete sTEC data set of the sv for the station (receiver)
    """
    stec_smth = TECs.reshape(1, 1, -1)
    stec_dict = {
        'dims': {
            'observer': np.size([nameObs]),
            'sv': np.size([sv]),
            'time': np.size(time),
        },
        'coords': {
            'observer': {
                'dims': ('observer',),
                'attrs': {'name': 'GNSS station'},
                'data': [nameObs],
            },
            'sv': {
                'dims': ('sv',),
                'attrs': {
                    'name': 'satellite vehicle',
                    'code': list(llaSat),
                    'geoLon': np.array(list(llaSat.values()))[:, 0].astype(np.float32),
                    'geoLat': np.array(list(llaSat.values()))[:, 1].astype(np.float32),
                    'geoAlt': np.array(list(llaSat.values()))[:, 2].astype(np.float32),
                },
                'data': [sv],
            },
            'time': {
                'dims': ('time',),
                'attrs': {'name': 'time'},
                'data': time,
            },
        },
        'data_vars': {
            'sTEC': {
                'dims': ('observer', 'sv', 'time'),
                'attrs': {
                    'name': 'sTEC',
                    'long_name': 'bi_frequency (relative) sTEC',
                    'units': 'TECU',
                },
                'data': stec_smth,
            },
            'DCBs': {
                'dims': ('observer', 'sv'),
                'attrs': {
                    'name': 'DCBs',
                    'long_name': '(satellite plus receiver) Differential Code Bias',
                    'units': 'TECU',
                },
                'data': np.array([[DCBs]]).astype(np.float32),
            },
            'lonipp': {
                'dims': ('observer', 'sv'),
                'attrs': {
                    'name': 'longitude',
                    'units': 'degree_E',
                    'axis': 'X',
                },
                'data': np.array([[lonipp]]).astype(np.float32),
            },
            'latipp': {
                'dims': ('observer', 'sv'),
                'attrs': {
                    'name': 'latitude',
                    'units': 'degree_N',
                    'axis': 'Y',
                },
                'data': np.array([[latipp]]).astype(np.float32),
            },
            'elevation': {
                'dims': ('observer', 'sv'),
                'attrs': {
                    'name': 'elevation angle',
                    'units': 'degree',
                },
                'data': np.array([[elevation]]).astype(np.float32),
            },
            # 'lonObs': {
            #     'dims': ('observer'),
            #     'attrs': {
            #         'name': 'longitude',
            #         'units': 'degree_E',
            #         'axis': 'X',
            #     },
            #     'data': np.array([lonObs]).astype(np.float32),
            # },
            # 'latObs': {
            #     'dims': ('observer'),
            #     'attrs': {
            #         'name': 'latitude',
            #         'units': 'degree_N',
            #         'axis': 'Y',
            #     },
            #     'data': np.array([latObs]).astype(np.float32),
            # },
            # 'altObs': {
            #     'dims': ('observer'),
            #     'attrs': {
            #         'name': 'altitude',
            #         'units': 'm',
            #     },
            #     'data': np.array([altObs]).astype(np.float32),
            # },
        },
        # 'attrs':{
        #     'lonObs': lonObs,
        #     'latObs': latObs,
        #     'altObs': altObs,
        # }
    }

    stec_dset = xr.Dataset.from_dict(stec_dict)

    del stec_smth, stec_dict
    gc.collect()
    return stec_dset


# %%
# -*- extract satellite position information -*-
def readOrbit(
    xyzObs: tuple[float],
    xyzsSat: np.ndarray,
    timeSat: np.ndarray,  # [np.datetime64]
    obsTIME: np.ndarray,  # [np.datetime64]
    Hionos: float = 350.0,  # unit: km
) -> tuple[list, list, list]:
    infoipp = [
        ippsProjection(
            xyzObs,
            xyzSat,
            Hionos * const.kilo,
        )
        for xyzSat in xyzsSat
    ]
    infoipp = np.array(infoipp)
    elesipp = (
        xr.DataArray(infoipp[:, 0], dims=('time',), coords={'time': timeSat})
        .interp(time=obsTIME, method='cubic', kwargs={'fill_value': 'extrapolate'})
        .values
    )
    xsipp, ysipp, zsipp = lla2ecef(infoipp[:, 1], infoipp[:, 2], infoipp[:, 3])
    xsipp = xr.DataArray(xsipp, dims=('time',), coords={'time': timeSat}).interp(
        time=obsTIME, method='cubic', kwargs={'fill_value': 'extrapolate'}
    )
    ysipp = xr.DataArray(ysipp, dims=('time',), coords={'time': timeSat}).interp(
        time=obsTIME, method='cubic', kwargs={'fill_value': 'extrapolate'}
    )
    zsipp = xr.DataArray(zsipp, dims=('time',), coords={'time': timeSat}).interp(
        time=obsTIME, method='cubic', kwargs={'fill_value': 'extrapolate'}
    )
    lonsipp, latsipp, _ = ecef2lla(xsipp.values, ysipp.values, zsipp.values)
    # lonsipp = xr.DataArray(
    #     lonsipp,
    #     dims=('time',),
    #     coords={'time':obsTIME}
    #     )
    # latsipp = xr.DataArray(
    #     latsipp,
    #     dims=('time',),
    #     coords={'time':obsTIME}
    #     )
    return elesipp, lonsipp, latsipp


# %%
# -*- calculate tec for each satellite -*-
#  *  根据不同卫星信号的观测值计算TEC
def sTEC_of_sv(
    oVar: xr.Dataset,
    sp3: xr.Dataset,
    nameObs: str,
    sv: str,
    frqc: dict[str:float],
    GEO: dict[str:float],
    fGim: Path,
    Hionos: float = 350000,
    reference_model: str = 'GIM',
    DCBplot: bool = False,
) -> xr.Dataset:
    """slant TEC along path for each satellite-vehicle for each station

    Args:
        obs (xr.Dataset): observation variables
        sp3 (xr.Dataset): orbit variables
        sv (str): satellite-vehicle code
        frqc (dict[str:float]): satellite signal frequencies on various bands
        GEO (dict[str:float]): satellite informations
        Hionos (float): for ipp projection
        reference_model (str): for DCB  correction

    Returns:
        xr.Dataset: sTEC data set
    """

    try:
        if sv.startswith('C'):
            obs = (
                oVar[['C1', 'C7', 'L1', 'L7']].sel(sv=sv).dropna(dim='time', how='any')
            )
            C2I, C7I = obs['C1'], obs['C7']
            L2I, L7I = obs['L1'], obs['L7']
            frqc1, frqc2 = frqc['BeiDou']['E2I'], frqc['BeiDou']['E5bI']
            slantTEC = relative_slantTEC(frqc1, frqc2, C2I, C7I, L2I, L7I)
            del C2I, C7I, L2I, L7I
        elif sv.startswith('J'):
            obs = (
                oVar[['C1', 'C5', 'L1', 'L5']].sel(sv=sv).dropna(dim='time', how='any')
            )
            C1, C5 = obs['C1'], obs['C5']
            L1, L5 = obs['L1'], obs['L5']
            frqc1, frqc2 = frqc['QZSS']['L1'], frqc['QZSS']['L5']
            slantTEC = relative_slantTEC(frqc1, frqc2, C1, C5, L1, L5)
            del C1, C5, L1, L5

        if slantTEC is None:
            print(f':::  {nameObs} - {sv} : calculate sTEC failed > <. ')
            return None
        else:
            # print(f':::  {nameObs} - {sv} : calculate sTEC successfully ^ ^. ')
            pass

        xObs, yObs, zObs = oVar.position
        # lonObs, latObs, altObs = ecef2lla(xObs, yObs, zObs)
        llaSat = GEO['geographic']

        # xyzSat = GEO['cartesian']
        # for sv in GEO['GEO']:
        #     lonSat, latSat, altSat = llaSat[sv]
        #     xSat, ySat, zSat = lla2ecef(lonSat, latSat, altSat)
        #     xyzSat[sv] = [xSat, ySat, zSat]

        if sv in sp3.sv.values:
            # xyzsSat = (
            #     sp3['position'].sel(sv=sv).values * const.kilo
            # )  # unit: m
            # timeSat = sp3['position'].sel(sv=sv).time.values
            # elesipp, lonsipp, latsipp = readOrbit(
            #     oVar.position, xyzsSat, timeSat, obs.time, Hionos
            #     )

            xSat, ySat, zSat = (
                sp3['position'].sel(sv=sv).values.mean(axis=0) * const.kilo
            )
            llaSat[sv] = list(
                ecef2lla(xSat, ySat, zSat)
            )  # return lla (unit: degree, degree, m)
        else:
            xSat, ySat, zSat = lla2ecef(llaSat[sv][0], llaSat[sv][1], llaSat[sv][2])

        eleipp, lonipp, latipp, _ = ippsProjection(
            (xObs, yObs, zObs), (xSat, ySat, zSat), Hionos
        )
        # elesipp, lonsipp, latsipp = (
        #     np.array([eleipp]*len(obs.time.values)),
        #     np.array([lonipp]*len(obs.time.values)),
        #     np.array([latipp]*len(obs.time.values)),
        #     )

        # print(f':::  {nameObs} - {sv} : correcting DCB... ')
        if reference_model == 'GIM':
            DCBs = sysmDCBs_byGimtec(
                slantTEC,
                lonipp,
                latipp,
                eleipp,
                gimnc=fGim,
                to_plot=DCBplot,
            )
        elif reference_model == 'IRI':
            DCBs = sysmDCBs_byiri2020(
                slantTEC,
                lonipp,
                latipp,
                eleipp,
            )
            print(":::  WARNING: reference model must be one of 'GIM' | 'IRI'")
            return None

        dsetTEC = struct_TECs_of_stations(
            slantTEC.values,
            slantTEC.time.values,
            nameObs,
            sv,
            lonipp,
            latipp,
            eleipp,
            # lonObs,
            # latObs,
            # altObs,
            llaSat,
            DCBs,
        )
        del slantTEC
        gc.collect()
    except Exception as ecode:
        print(f':::  {nameObs} - {sv} : calculate sTEC exited with {ecode} > <. ')
        return None
    else:
        print(f':::  {nameObs} - {sv} : calculate sTEC successfully ^ ^. ')
        return dsetTEC


# %%
def TEC_of_receiver(
    idxObs,
    fRnx,
    satUse,
    stations,
    measures,
    sp3,
    attrs,
    frqcSat,
    GEO,
    fGim,
    Hionos,
    reference_model,
    DCBplot,
    # hdf5GEO,
    savedir,
    satsys,
    show_plots,
):
    # satUse = set()
    satMeas = []
    # rnxheader = grnx.obsheader2(fRnx)['PRN / # OF OBS']
    for satCHAR in satUse:
        satMeas = list(set(satMeas).union(set(measures[satCHAR])))

    # if len(satUse) == 0:
    #     return None
    # else:
    #     pass
    nameObs = fRnx.stem[0:4]
    if stations == 'ALL':
        pass
    elif (nameObs.lower()[:2] not in stations) and (nameObs.lower() not in stations):
        return None

    try:
        oVar = grnx.load(
            fRnx,
            use=satUse,
            meas=satMeas,
            fast=False,
        )
        if len(oVar.sv.values) == 0:
            return None
        else:
            pass
        rObs = np.sqrt(np.dot(oVar.position, oVar.position))
        if rObs < 500 * const.kilo:
            return None
        else:
            xObs, yObs, zObs = oVar.position
            lonObs, latObs, altObs = ecef2lla(xObs, yObs, zObs)
            attrs['samp_rate(s)'] = oVar.interval
            attrs['time_sysm'] = oVar.time_system
            attrs['rnx_model'] = oVar.rxmodel

    except Exception as ecode:
        print(
            f'\n{idxObs:3d}  {nameObs} - tec : loading RNX file exited with {ecode}, skipped... \n'
        )
        return None
    # tec_existed = False
    obsDset = []
    for sv in oVar.sv.values:
        geoTEC = sTEC_of_sv(
            oVar,
            sp3,
            nameObs,
            sv,
            frqcSat,
            GEO,
            fGim,
            Hionos * const.kilo,
            reference_model,
            DCBplot,
        )

        if geoTEC is None:
            continue
        else:
            pass

        obsDset.append(geoTEC)
        del geoTEC
        # if 'TECsingle' not in locals().keys():
        #     TECsingle = geoTEC
        # else:
        #     TECsingle = xr.concat([TECsingle, geoTEC], dim='sv')
    # tec_existed = True

    del oVar
    gc.collect()
    if len(obsDset) == 0:
        return None
    else:
        obsTEC = xr.concat(obsDset, dim='sv')
        # ds["temperature"] = (("x", "y", "time"), temp)
        obsTEC['lonObs'] = (('observer'), np.array([lonObs]).astype(np.float32))
        obsTEC['latObs'] = (('observer'), np.array([latObs]).astype(np.float32))
        obsTEC['altObs'] = (('observer'), np.array([altObs]).astype(np.float32))
        plotTEC_of_receiver(
            obsTEC.sel(observer=nameObs),
            nameObs,
            savedir=savedir,
            satsys=satsys,
            show_plots=show_plots,
        )
        print(f'\n{idxObs:3d}  {nameObs} - ALL : concat TEC successfully ^ ^. \n      ')
        return obsTEC

    # if tec_existed:
    # if 'TECall' not in locals().keys():
    #     TECall = TECsingle
    # else:
    #     TECall = xr.concat([TECall, TECsingle], dim='observer')
    # hdf5GEO[nameObs] = TECsingle.astype(np.float32).to_dataframe()

    # del TECsingle
    #     gc.collect()
    #
    # return nameObs, TECsingle


# %%
# -*- result check -*-
#  *  数据检验
#  *  进行绘图直观判断数据质量以及程序是否正常工作


# basic font and etc. configurations
def plot_configurations():
    config = {
        # 'font.family': 'serif',  # sans-serif/serif/cursive/fantasy/monospace
        'font.size': 5,  # medium/large/small
        'font.style': 'normal',  # normal/italic/oblique
        'font.weight': 'bold',
        'mathtext.default': 'regular',
        # 'font.serif': ['Times New Roman'],  # 'Simsun'宋体
        'axes.unicode_minus': False,  # 用来正常显示负号
    }

    fonts = fm.FontProperties(
        # family='Times New Roman',
        weight='bold',
        size=7,
    )
    return config, fonts


# %%
def plotTEC_of_receiver(
    TEC: xr.Dataset,
    nameObs: str,
    savedir: str,
    satsys: str = 'BeiDou',
    show_plots: bool = False,
) -> None:
    """plot tecs of diverse satellites for each station (receiver)

    Args:
        TEC (xr.Dataset): relative phase sTECs
        eleipp (dict): elevation angle in degrees from a receiver to satellites
        savedir (str): path for plot output
        satsys (str, optional): _Defaults to 'BeiDou'.
        show_plots (bool, optional): Defaults to False, not to show plots in interactive window .
    """
    config, fonts = plot_configurations()
    plt.rcParams.update(config)

    fig = plt.figure(figsize=(4, 1.6), dpi=300)
    ax = fig.add_subplot(111)

    colors = {
        'C01': 'crimson',
        'C02': 'turquoise',
        'C03': 'dodgerblue',
        'C04': 'gold',
        'C05': 'blueviolet',
        'J07': 'lawngreen',
    }

    for sv in TEC.sv.values:
        stec = TEC['sTEC'].sel(sv=sv).dropna(dim='time', how='all')
        DCBs = TEC['DCBs'].sel(sv=sv).values
        lon = TEC['lonipp'].sel(sv=sv).values
        lat = TEC['latipp'].sel(sv=sv).values
        ele = TEC['elevation'].sel(sv=sv).values
        stec = stec - DCBs

        ax.plot(
            stec['time'].values,
            stec.values * np.sin(np.radians(ele)),
            color=colors[sv],
            marker=',',
            ls='',
            lw=2,
            label=f'{sv}: {lon:6.2f}, {lat:6.2f}',
        )

    plt.legend(labelcolor='linecolor', markerscale=50, loc='upper right')
    plt.xlabel('date (UT)', fontproperties=fonts)
    plt.ylabel('TEC (TECU)', fontproperties=fonts)
    plt.title(f'{nameObs} {satsys} STEC', fontproperties=fonts)

    plt.savefig(f'{savedir}/{satsys}_sTEC_{nameObs}.png', bbox_inches='tight')

    if show_plots:
        plt.show()
    else:
        pass

    plt.clf()
    plt.close('all')
    gc.collect()
    return


# %%
# -*- MAIN PROCESSION: tec inversion -*-
#  *  观测日期\站点信息
#  *  定义数据存储路径
#  *  卫星信号频率\位置坐标
#  *  依次计算每个台站每颗卫星对应的TEC观测值
#  *  将TEC观测值结构化为Dataset
#  *  将不同台站不同卫星的数据顺次拼接, 得到每日观测值
#  *  将每日观测值存入nc文件中
def tec_inversion(
    show_plots: bool = False,
    reference_model: str = 'GIM',
    DCBplot: bool = False,
    poolCOUNT: int = 1,
    satsys: str = 'geoSat',
    sample_rate: str = '1s',
    satUse: set[str] = {'C', 'J'},
) -> tuple[xr.Dataset, str]:
    """MAIN PROCESSION: tec inversion for each day

    Args:
        show_plots (bool, optional): whether to show plots in the interactive window, defaults to False.
        reference_model (str, optional): correct DCB error refer to 'GIM' or 'IRI' model, defaults to 'GIM'.
        DCBplot(bool, optional): whether to show DCB result in plots.
        poolCOUNT(int): multiprocessing pools,
        satsys(str):
        sample_rate(str):
    """
    # initials ------------------------------------------------------------------------
    if poolCOUNT > cpu_count():
        print('WARNING: pool count is excessful! ')
    tomlConfig = './config.toml'
    with open(tomlConfig, mode='rb') as f_toml:
        selfConfig = toml.load(f_toml)

    stations = selfConfig['stations']
    year = selfConfig['year']
    doys = selfConfig['doys']

    # year = 2024
    # doys = [doy for doy in range(172, 182)]

    if doys == 'auto':
        year = int(datetime.today().strftime('%Y'))
        doys = [int(datetime.today().strftime('%j')) - 1]
    else:
        pass
    stryear = f'{year}'
    Hionos = selfConfig['Hionos']  # unit: km

    srcDisk = Path(selfConfig['source'])
    archives = selfConfig['archives']
    archGnss, archSp3, archGim, archNc = (
        archives['dataGnss'],
        archives['dataSp3'],
        archives['dataGim'],
        archives['dataNC'],
    )
    dirData, dirSp3, dirGim = srcDisk / archGnss, srcDisk / archSp3, srcDisk / archGim

    dirs = {}
    dirs['SP3'] = dirSp3 / stryear
    dirs['GIM'] = dirGim / archNc / stryear

    if dirData.is_dir():
        pass
    else:
        print('warning: datadir is invalid! ')

    archRnx, archTEC, archPlots = (
        archives['dataRnx'],
        archives['dataTEC'],
        archives['dataPlots'],
    )

    mk_archives(dirData / archTEC, [stryear])
    for doy in doys:
        strdoy = f'{doy:0>3.0f}'
        mk_archives(dirData / archTEC / stryear, [strdoy])
        mk_archives(dirData / archTEC / stryear / strdoy, [archPlots])
    del doy, strdoy

    tomlGEO = './geoSat.toml'
    with open(tomlGEO, mode='rb') as f_toml:
        GEO = toml.load(f_toml)

    measures = GEO['measurements']
    frqcSat = GEO['sig_frqc']
    attrs = {
        'signals': list(frqcSat),
        'sat_sysm': satsys,
        'Hionos(km)': Hionos,
    }

    # run cycles ----------------------------------------------------------------------
    for doy in doys:
        strdoy = f'{doy:0>3.0f}'
        dirs['RNX'] = dirData / archRnx / stryear / strdoy
        dirs['TEC'] = dirData / archTEC / stryear / strdoy
        dirs['plots'] = dirData / archTEC / stryear / strdoy / archPlots
        # Path(dirs['plots']).mkdir(mode=0o777, parents=True, exist_ok=True)

        if dirs['RNX'].is_dir():
            pass
        else:
            continue
        ncGEO = Path(dirs['TEC'], f'{satsys}_TEC_{strdoy}_{sample_rate}_{stryear}.nc')
        if ncGEO.is_file():
            print(
                f'---* *--- {strdoy}: nc file has already existed, skipped...---* *--- \n'
            )
            continue
        else:
            pass
        print(f'\n---* *--- mission: Calculating tecs of DOY {strdoy}---* *--- \n')

        filesRnx = sorted(dirs['RNX'].glob('*_*S_[M,C,J]O.*x'))
        # filesRnx = sorted(dirs['RNX'].glob(f'*{strdoy}0.*d'))
        if year <= 2018:
            gpsWeek, _, weekDay = gpsWeekOf(year, doy)
            filesSp3 = dirs['SP3'].glob(f'*{gpsWeek}{weekDay}.sp3')
        else:
            filesSp3 = dirs['SP3'].glob(f'*{stryear}{strdoy}*.SP3*')

        filesGim = dirs['GIM'].glob(f'CO*{stryear}{strdoy}*.nc')
        fSp3, fGim = sorted(filesSp3)[0], sorted(filesGim)[0]
        sp3 = grnx.load(fSp3)

        obsTECs = []
        for idx, _ in enumerate(filesRnx):
            if idx % poolCOUNT == 0:
                pass
            else:
                continue
            pool = Pool(poolCOUNT)
            arguments = (
                (
                    idxObs,
                    fRnx,
                    satUse,
                    stations,
                    measures,
                    sp3,
                    attrs,
                    frqcSat,
                    GEO,
                    fGim,
                    Hionos,
                    reference_model,
                    DCBplot,
                    #  hdf5GEO,
                    dirs['plots'],
                    satsys,
                    show_plots,
                )
                for idxObs, fRnx in enumerate(
                    filesRnx[idx : idx + poolCOUNT], start=idx
                )
            )
            results = [
                pool.apply_async(TEC_of_receiver, args=argument)
                for argument in arguments
            ]
            pool.close()
            pool.join()
            obsTECs.extend([res.get() for res in results if res.get() is not None])
            # nameVars = ['sv', 'time', 'sTEC', 'DCBs', 'lonipp', 'latipp', 'elevation']
            # hdf5GEO.append(nameObs, TECsingle.to_dataframe().astype(np.float32))
            # hdf5GEO.append('/'.join([nameObs, 'attrs']), pd.Series(TECsingle.attrs))

        # hdf5GEO.append('observer', pd.Series(obsNlist).astype(str))
        # # hdf5GEO.append('vars', pd.Series(nameVars))
        # hdf5GEO.close()

        obsTECs = xr.concat(obsTECs, dim='observer')
        obsTECs.attrs = attrs
        obsTECs.to_netcdf(
            ncGEO,
            encoding={
                'time': {'zlib': True, 'complevel': 6},
                'sTEC': {'zlib': True, 'complevel': 6},
                'DCBs': {'zlib': True, 'complevel': 6},
                'lonipp': {'zlib': True, 'complevel': 6},
                'latipp': {'zlib': True, 'complevel': 6},
                'elevation': {'zlib': True, 'complevel': 6},
            },
        )
        # del TECall
    del doy, strdoy
    print('\n---* *--- mission accomplished ---* *--- \n')
    return obsTECs, ncGEO
    # test()


# %%
if __name__ == '__main__':
    obsTECs, ncGEO = tec_inversion(
        show_plots=False, reference_model='GIM', DCBplot=False, poolCOUNT=6
    )

# %%
