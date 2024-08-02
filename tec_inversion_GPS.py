# %%
#!./.venv/bin python3
# -*- encoding: utf-8 -*-
"""
@filename :  tec_inversion_GPS.py
@desc     :  extract and inverse TEC data(1Hz) from RINEX files for each day
@time     :  2024/06/06
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
            estRate = 0  # if interval > 5min or |dTEC| > 25: estRate defaults to 0
        else:
            # if index >= 32:
            rate_sort = np.sort(stecRate[index - 1 - 15 : index])
            rateForward1 = rate_sort[5:11].mean()
            rate_sort = np.sort(stecRate[index - 2 - 15 : (index - 1)])
            rateForward2 = rate_sort[5:11].mean()
            interval_sort = np.sort(intervals[index - 1 - 15 : index])
            intervalForward1 = interval_sort[5:11].mean()
            # else:
            # rate_sort = np.sort(stecRate[1:index])
            # rateForward1 = rate_sort[5:11].mean()
            # rate_sort = np.sort(stecRate[0 : index - 1])
            # rateForward2 = rate_sort[5:11].mean()
            drate = (rateForward1 - rateForward2) / intervalForward1 * intervals[index]
            estRate = rateForward1 + drate

        if np.abs(stecRate[index] - estRate) > 0.05:
            cycle_slip = (
                COEF * (GAMMA - 1) * intervals[index] * estRate / FRQC1**2
                - dPhase[index]
                + dPhase[index - 1]
            )
            stecRate[index] = stecRate[index - 5 : index].mean() + drate
            dPhase[index:] = dPhase[index:] + cycle_slip
    return dPhase


def relative_slantTEC(
    FRQC1: float,
    FRQC2: float,
    pseudoRange1: xr.Dataset,
    pseudoRange2: xr.Dataset,
    carrierPhase1: xr.Dataset,
    carrierPhase2: xr.Dataset,
    sample_period: float = 1.0,
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
def sysmDCBs_byTriseries(
    stec: xr.Dataset,
    ipplon: float,
    ipplat: float,
    elevation: float,
    sat_sysm: str = 'GPS',
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
    ipplons: list[float],
    ipplats: list[float],
    elevations: list[float],
    gimnc: str,
    satsys: str = 'GPS',
    to_plot: bool = False,
) -> np.float64:
    """(RECOMMANDED) Diffrential Code Bias correction referring to
        Global Ionosphere Map defaults from CODE,

    Args:
        stec (xr.Dataset): sTEC data set with dimension(lon, lat, time)
        ipplons (list[float]): longitude of IPPs
        ipplats (list[float]): latitude of IPPs
        elevations (list[float]): elevation angle in degrees from receiver to satellite
        fimnc (str): the path of gim tec file in nc format
        sat_sysm (str, optional): Defaults to 'BeiDou'.
        to_plot (bool, optional): Defaults to False, not to plot DCB results

    Returns:
        np.float64: mean value of nighttime DCBs to represent constant DCBs of a day
    """
    # elevations = np.array(elevations)
    # ipplons = np.array(ipplons)
    # ipplats = np.array(ipplats)
    elevMAX = np.max(elevations)
    tidx_lower = np.where(elevations == elevMAX)
    ilon_lower = ipplons[tidx_lower]
    ilat_lower = ipplats[tidx_lower]

    stec_lower = stec.values[tidx_lower]
    time_lower = stec.time.values[tidx_lower]

    gimtec = xr.open_dataarray(gimnc)
    gtec_lower = gimtec.interp(
        lat=ilat_lower,
        lon=ilon_lower,
        time=time_lower,
        method='cubic',
        kwargs={'fill_value': 'extrapolate'},
    ).values

    DCBs = stec_lower - gtec_lower / np.sin(np.radians(elevMAX))
    DCBs = DCBs.mean()

    ### plot test
    if not to_plot:
        return DCBs
    else:
        gtecs = []
        gtime = []
        t_range = stec.time.values
        for tnode, glon, glat in zip(t_range[::30], ipplons[::30], ipplats[::30]):
            gtec = gimtec.interp(
                lat=glat, lon=glon, time=tnode, method='quadratic'
            ).values
            gtime.append(tnode)
            gtecs.append(gtec)

        # print(stec.values.shape, elevations.shape )
        vtecs = (stec.values - DCBs) * np.sin(np.radians(elevations))

        ax = plt.figure(figsize=(6, 5)).add_subplot(111)
        ax.plot(gtime, gtecs, color='b', marker=',', linestyle='-', label='gimtec')
        ax.plot(
            t_range,
            vtecs,
            color='c',
            marker=',',
            linestyle='-',
            label='vtec_corrected',
        )
        ax.axhline(DCBs, color='r', label='DCBs')

        plt.legend(labelcolor='linecolor')
        plt.show()

        return DCBs


# DCBs
# test()


# %%
# -*- build up Dataset
def struct_TECs_of_stations(
    TECs: np.ndarray,
    time: np.ndarray,
    lonsipp: list[float],
    latsipp: list[float],
    nameObs: str,
    sv: str,
    DCBs: np.float64,
    # lonObs: float,
    # latObs: float,
    # altObs: float,
) -> xr.Dataset:
    """collect diverse datas and basic attributes into a Data set

    Args:
        TECs (np.ndarray): slant phase TEC (relative)
        time (np.ndarray): coordinate time
        lonipp (list[float]): coordinate longitude
        latipp (list[float]): coordinate latitude
        sv (str): satellite vehicle code
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
                    'long_name': 'dual_frequency combined sTEC',
                    'units': 'TECU',
                },
                'data': stec_smth,
            },
            # 'elevation': {
            #     'dims': ('observer', 'sv', 'time'),
            #     'attrs': {
            #         'name': 'elevation angle',
            #         'units': 'degree',
            #     },
            #     'data': [[elevations]],
            # },
            'lonipp': {
                'dims': ('observer', 'sv', 'time'),
                'attrs': {
                    'name': 'longitude',
                    'units': 'degree_E',
                    'axis': 'X',
                },
                'data': np.array([[lonsipp]]).astype(np.float32),
            },
            'latipp': {
                'dims': ('observer', 'sv', 'time'),
                'attrs': {
                    'name': 'latitude',
                    'units': 'degree_N',
                    'axis': 'Y',
                },
                'data': np.array([[latsipp]]).astype(np.float32),
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
    }

    stec_dset = xr.Dataset.from_dict(stec_dict)

    del stec_smth, stec_dict
    gc.collect()
    return stec_dset


# %%
# # -*- extract satellite position information -*-
def readOrbit(
    xyzObs: tuple[float],
    xyzsSat: np.ndarray,
    timeSat: np.ndarray,  # [np.datetime64]
    obsTIME: np.ndarray,  # [np.datetime64]
    Hionos: float = 350.0,  # unit: km
) -> tuple[xr.DataArray, list, list]:
    infoipp = [
        ippsProjection(
            xyzObs,
            xyzSat,
            Hionos * const.kilo,
        )
        for xyzSat in xyzsSat
    ]
    infoipp = np.array(infoipp)

    elesipp = xr.DataArray(
        infoipp[:, 0], dims=('time',), coords={'time': timeSat}
    ).interp(time=obsTIME, method='cubic', kwargs={'fill_value': 'extrapolate'})
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
    return elesipp, lonsipp, latsipp


# %%
# -*- calculate tec for each satellite -*-
#  *  根据不同卫星信号的观测值计算TEC
def sTEC_of_sv(
    oVar: xr.Dataset,
    sp3: xr.Dataset,
    nameObs: str,
    sv: str,
    interval: float,
    measures: list[str],
    frqc: dict[str:float],
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
    # print(f'{sv:>12s} : calculate combination TEC starts:\n')
    try:
        obs = oVar.sel(sv=sv).dropna(dim='time', how='any')
        if obs.time.values.shape[0] < int(1.0 * 60 * 60 / interval):  # shorter than 2hr
            return None
        else:
            pass
        La, Lb = obs[measures[0]], obs[measures[1]]
        Ca, Cb = obs[measures[2]], obs[measures[3]]
        slantTEC = relative_slantTEC(
            frqc[measures[0]], frqc[measures[1]], Ca, Cb, La, Lb
        )
        del Ca, Cb, La, Lb
        if slantTEC is None:
            print(f':::  {nameObs} - {sv} : calculate sTEC failed > <. ')
            return None
        else:
            pass

        xyzsSat = sp3['position'].sel(sv=sv).values * const.kilo  # unit: m

        timeSat = sp3['position'].sel(sv=sv).time.values
        elesipp, lonsipp, latsipp = readOrbit(
            oVar.position, xyzsSat, timeSat, obs.time, Hionos
        )
        if np.isin(np.nan, lonsipp):
            print('error in projection')
            return None

        # if np.max(elesipp.values) <= 20.:
        #     return None
        # else:
        #     pass
        # xObs, yObs, zObs = oVar.position
        # lonObs, latObs, altObs = ecef2lla(xObs, yObs, zObs)

        # print(f':::  {nameObs} - {sv} : correcting DCB... ')
        if reference_model == 'GIM':
            DCBs = sysmDCBs_byGimtec(
                slantTEC,
                lonsipp,
                latsipp,
                elesipp.values,
                gimnc=fGim,
                to_plot=DCBplot,
            )
        elif reference_model == 'Tri':
            DCBs = sysmDCBs_byTriseries(slantTEC, lonsipp, latsipp, elesipp.values)
        else:
            print(":::  WARNING: reference model must be one of 'GIM' | 'Tri'")
            return None

        dsetTEC = struct_TECs_of_stations(
            slantTEC.values,
            slantTEC.time.values,
            lonsipp,
            latsipp,
            nameObs,
            sv,
            DCBs,
            # lonObs,
            # latObs,
            # altObs,
        )
        idxtime = elesipp.where(elesipp > 10.0).time.values
        dsetTEC = dsetTEC.sel(time=idxtime)
        del slantTEC
        gc.collect()

        if len(dsetTEC['lonipp'].loc[nameObs, sv].dropna(dim='time', how='all')) == 0:
            return None
        else:
            # print(f':::  {nameObs} - {sv} : calculate sTEC successfully ^ ^. ')
            return dsetTEC
        print(8)
    except Exception as ecode:
        print(f':::  {nameObs} - {sv} : calculate sTEC exited with {ecode} > <. ')
        return None


# %%
def readRnxFile(
    fRnx,
    satCHAR,
    stations,
    # hdf5GPS,
    # hdf5S4,
    sp3,
    idxObs,
    frqcSat,
    fGim,
    Hionos,
    reference_model,
    DCBplot,
    attrs,
    savedir,
    satsys,
    show_plots,
):
    rnxheader = grnx.obsheader2(fRnx)
    # if satCHAR not in rnxheader['PRN / # OF OBS']:
    #     return None
    # else:
    #     pass

    if 'L1' in rnxheader['# / TYPES OF OBSERV']:
        pass
    else:
        return None

    if 'L5' in rnxheader['# / TYPES OF OBSERV']:
        measures = ['L1', 'L5', 'C1', 'C5']
    elif 'L2' in rnxheader['# / TYPES OF OBSERV']:
        measures = ['L1', 'L2', 'C1', 'C2']
    else:
        return None

    nameObs = fRnx.stem[0:4]

    if stations == 'ALL':
        pass
    elif (nameObs.lower()[:2] not in stations) and (nameObs.lower() not in stations):
        return None

    try:
        oVar = grnx.load(fRnx, use=satCHAR, meas=measures, fast=False)
        if len(oVar.sv.values) == 0:
            return None
        else:
            pass
        if np.sqrt(np.dot(oVar.position, oVar.position)) < 500 * const.kilo:
            return None
        else:
            xObs, yObs, zObs = oVar.position
            lonObs, latObs, altObs = ecef2lla(xObs, yObs, zObs)
            attrs['samp_rate(s)'] = oVar.interval
            attrs['time_sysm'] = oVar.time_system
            attrs['rnx_model'] = oVar.rxmodel
    except Exception as ecode:
        gc.collect()
        print(f'     {nameObs} - loading RNX file exited with {ecode}, skipped... ')
        return None
    # tec_existed = False
    dsetTEC = []
    for sv in oVar.sv.values:
        gpsTEC = sTEC_of_sv(
            oVar,
            sp3,
            nameObs,
            sv,
            oVar.interval,
            measures,
            frqcSat,
            fGim,
            Hionos,
            reference_model,
            DCBplot,
        )
        if gpsTEC is None:
            continue
        else:
            dsetTEC.append(gpsTEC)

        # S4indices = S4_of_sv(
        #     nameObs, oVar, sv, sp3, Hionos
        #     )
        # if S4indices is None:
        #     continue
        # else:
        #     dsetS4.append(S4indices)

        # try:
        del gpsTEC
    del oVar
    gc.collect()

    if len(dsetTEC) == 0:
        return None
    else:
        TECsingle = xr.concat(dsetTEC, dim='sv')
        TECsingle['lonObs'] = (('observer'), np.array([lonObs]).astype(np.float32))
        TECsingle['latObs'] = (('observer'), np.array([latObs]).astype(np.float32))
        TECsingle['altObs'] = (('observer'), np.array([altObs]).astype(np.float32))
        print(f'{idxObs:3d}  {nameObs} - ALL : concat successfully ^ ^. \n')
        plotTEC_of_receiver(
            TECsingle.sel(observer=nameObs),
            nameObs,
            savedir=savedir,
            satsys=satsys,
            show_plots=show_plots,
        )

        return TECsingle.sel(observer=nameObs).observer, TECsingle.sel(observer=nameObs)


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
        savedir (str): path for plot output
        satsys (str, optional): _Defaults to 'BeiDou'.
        show_plots (bool, optional): Defaults to False, not to show plots in interactive window .
    """
    config, fonts = plot_configurations()
    plt.rcParams.update(config)

    fig = plt.figure(figsize=(4, 1.6), dpi=300)
    ax = fig.add_subplot(111)

    for sv in TEC.sv.values:
        stec = TEC['sTEC'].sel(sv=sv).dropna(dim='time', how='all')
        lonipp = TEC['lonipp'].sel(sv=sv).dropna(dim='time', how='all')
        latipp = TEC['latipp'].sel(sv=sv).dropna(dim='time', how='all')
        DCBs = TEC['DCBs'].sel(sv=sv).values
        stec = stec - DCBs

        # time = stec['time']

        ax.scatter(
            lonipp,
            latipp,
            c=stec,
            cmap='jet',
            marker=',',
            s=0.2,
            vmax=300.0,
            vmin=0.0,
        )

    # plt.legend(labelcolor='linecolor', markerscale=50, loc='upper right')
    plt.xlabel('lon', fontproperties=fonts)
    plt.ylabel('lat', fontproperties=fonts)
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
    satsys: str = 'GPS',
    sample_rate: str = '01s',
    satCHAR: str = 'G',
) -> tuple[xr.Dataset, str]:
    """MAIN PROCESSION: tec inversion for each day

    Args:
        show_plots (bool, optional): whether to show plots in the interactive window, defaults to False.
        reference_model (str, optional): correct DCB error refer to 'GIM' or 'IRI' model, defaults to 'GIM'.
        DCBplot(bool, optional): whether to show DCB result in plots.
        poolCOUNT(int): multiprocessing pools
    """
    # initials ------------------------------------------------------------------------

    if poolCOUNT > cpu_count():
        print('WARNING: pool count is excessful! ')

    # tomlConfig = './config.toml'
    tomlConfig = '/home/echoo/codePy/IEFion/config.toml'
    with open(tomlConfig, mode='rb') as f_toml:
        selfConfig = toml.load(f_toml)

    stations = selfConfig['stations']
    # year = selfConfig['year']
    # doys = selfConfig['doys']
    year = 2024
    doys = [doy for doy in range(1, 183)]

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

    # tomlGPS = './GPS.toml'
    tomlGPS = '/home/echoo/codePy/IEFion/GPS.toml'
    with open(tomlGPS, mode='rb') as f_toml:
        GPS = toml.load(f_toml)

    frqcSat = GPS['sig_frqc']
    frqces = np.array(list(frqcSat.values())) / const.mega
    attrs = {
        'signals': list(frqcSat),
        'frqces(MHz)': [frqces[0], frqces[2]],
        'sat_sysm': satsys,
        'Hionos(km)': Hionos,
    }

    # run cycles ----------------------------------------------------------------------
    for doy in doys:
        strdoy = f'{doy:0>3.0f}'
        dirs['RNX'] = dirData / archRnx / stryear / strdoy
        dirs['TEC'] = dirData / archTEC / stryear / strdoy
        dirs['plots'] = dirData / archTEC / stryear / strdoy / archPlots

        ncGPS = Path(dirs['TEC'], f'{satsys}_TEC_{strdoy}_{sample_rate}_{stryear}.nc')
        if ncGPS.is_file():
            continue

        # hdf5GPS = pd.HDFStore(pathGPS, mode="w", complevel=4, complib='blosc:zstd')

        # ncS4 = Path(dirs['TEC'], f'{satsys}_S4_{strdoy}_{sample_rate}_{stryear}.nc')
        # hdf5S4 = pd.HDFStore(pathS4, mode="w", complevel=4, complib='blosc:zstd')
        # print(f'Calculating tecs of DOY {strdoy} by Hatch filter')

        # filesRnx = sorted(dirs['RNX'].glob('*01D_*S_[M,C,J]O.*x'))
        filesRnx = sorted(dirs['RNX'].glob(f'*{strdoy}0.*d'))
        if year <= 2018:
            gpsWeek, _, weekDay = gpsWeekOf(year, doy)
            filesSp3 = dirs['SP3'].glob(f'*{gpsWeek}{weekDay}.sp3')
        else:
            filesSp3 = dirs['SP3'].glob(f'*{stryear}{strdoy}*.SP3*')

        filesGim = dirs['GIM'].glob(f'CO*{stryear}{strdoy}FIN.nc')
        fSp3, fGim = sorted(filesSp3)[0], sorted(filesGim)[0]
        sp3 = grnx.load(fSp3)

        obsNames = []
        for idx, _ in enumerate(filesRnx):
            if idx % poolCOUNT == 0:
                pass
            else:
                continue
            pool = Pool(poolCOUNT)
            arguments = (
                (
                    fRnx,
                    satCHAR,
                    stations,
                    sp3,
                    idxObs,
                    frqcSat,
                    fGim,
                    Hionos,
                    reference_model,
                    DCBplot,
                    attrs,
                    dirs['plots'],
                    satsys,
                    show_plots,
                )
                for idxObs, fRnx in enumerate(
                    filesRnx[idx : idx + poolCOUNT], start=idx
                )
            )
            results = [
                pool.apply_async(readRnxFile, args=argument) for argument in arguments
            ]
            pool.close()
            pool.join()
            for res in results:
                if res.get() is not None:
                    nameObs, dsetTECs = res.get()
                    obsNames.append(nameObs)
                    dsetTECs.attrs = attrs
                    comp = dict(zlib=True, complevel=6)
                    encoding = {var: comp for var in dsetTECs.data_vars}
                    dsetTECs.to_netcdf(
                        ncGPS, mode='a', group=str(nameObs.values), encoding=encoding
                    )
        dsetNames = xr.concat(obsNames, dim='observer')
        dsetNames.to_netcdf(
            ncGPS,
            mode='a',
            group='observer',
        )
    print('mission accomplished * *')
    return ncGPS
    # test()


# %%
if __name__ == '__main__':
    ncGPS = tec_inversion(
        show_plots=False,
        reference_model='GIM',
        DCBplot=False,
        poolCOUNT=16,
        sample_rate='30s',
    )

# %%
