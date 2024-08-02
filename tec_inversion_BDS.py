# %%
#!./.venv/bin python3
# -*- encoding: utf-8 -*-
"""
@filename :  tec_inversion_BDS.py
@desc     :  extract and inverse TEC data(1Hz) from RINEX files for each day
@time     :  2024/06/01
@author   :  _koii^_ (Liu Hong), Institute of Earthquake Forecasting (IEF), CEA.
@Version  :  1.0
@Contact  :  koi_alley@outlook.com
"""

# here import libs
import gc
from datetime import datetime
from pathlib import Path

import georinex as grnx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import scipy.constants as const
import tomli_w as tomlw
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

    return ele, [lonipp, latipp, HIONOS]


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

        if np.abs(stecRate[index] - estRate) > 0.15:
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
# -*- calculate tec for each satellite -*-
#  *  根据不同卫星信号的观测值计算TEC
def sTEC_of_sv(
    obs: xr.Dataset, sv: str, frqc: dict[str:float], sat_sysm: str = 'BeiDou'
) -> xr.Dataset:
    """slant TEC along path for each satellite-vehicle for each station

    Args:
        obs (xr.Dataset): observation variables
        sv (str): satellite-vehicle code
        frqc (dict[str:float]): satellite signal frequencies on various bands
        sat_sysm (str, optional): Defaults to 'BeiDou'.

    Returns:
        xr.Dataset: sTEC data set
    """
    print(f'\n      {sat_sysm} - {sv:3} : calculate combination TEC starts:\n')
    try:
        C2I = obs['C1']
        C7I = obs['C7']
        L2I = obs['L1']
        L7I = obs['L7']
        stec = relative_slantTEC(frqc['E2I'], frqc['E5bI'], C2I, C7I, L2I, L7I)
        del C2I, C7I, L2I, L7I

    except Exception as ecode:
        print(
            f'{sat_sysm:>12s} - {sv:3} : frequency combination TEC exited with {ecode}. \n'
        )
    else:
        print(f'{sat_sysm:>12s} - {sv:3} : frequency combination TEC successfully. \n')
        return stec


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
    satsys: str = 'BeiDou',
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
        sat_sysm (str, optional): Defaults to 'BeiDou'.
        to_plot (bool, optional): Defaults to False, not to plot DCB results

    Returns:
        np.float64: mean value of nighttime DCBs to represent constant DCBs of a day
    """
    gimtec = xr.open_dataarray(gimnc)
    gtec = gimtec.interp(lat=ipplat, lon=ipplon, method='cubic')

    sv = stec.sv.values
    print(f'      {satsys} - {sv} : calculate the SPR DCBs of observing station... \n')

    time = stec.time
    t_range = pd.date_range(time[0].values, periods=24 * 6, freq='10Min')

    gtecs = np.array([])
    tnodes = np.array([])
    stecs = np.array([])
    for tnode in t_range:
        pydt = tnode.to_pydatetime()

        if ipplon > 180:
            ipplon -= 360.0

        IPP_tLocal = ipplon / 15.0 + pydt.hour
        if IPP_tLocal < 0:
            IPP_tLocal += 24.0
        elif IPP_tLocal > 24:
            IPP_tLocal -= 24.0
        else:
            pass

        if IPP_tLocal >= 23.0 or IPP_tLocal < 5.0:
            pass
        else:
            continue

        gtecNode = gtec.interp(time=tnode, method='quadratic').values
        stecNode = stec.sel(time=tnode, method='nearest').values

        gtecs = np.hstack((gtecs, gtecNode))
        tnodes = np.hstack((tnodes, tnode))
        stecs = np.hstack((stecs, stecNode))

    DCBs = stecs - gtecs / np.sin(np.radians(elevation))

    ### plot test
    if not to_plot:
        return DCBs.mean()
    else:
        t_range = pd.date_range(time[0].values, periods=48, freq='30Min')
        stecs = stec.interp(time=t_range, method='quadratic')
        gtecs = gtec.interp(time=t_range, method='quadratic')

        ax = plt.figure(figsize=(6, 5)).add_subplot(111)
        ax.plot(t_range, gtecs, color='b', marker=',', linestyle='-', label='gimtec')
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

        return DCBs.mean()


# DCBs
# test()


# %%
# -*- build up Dataset
def struct_TECs_of_stations(
    TECs: np.ndarray,
    time: np.ndarray,
    lon: np.float64,
    lat: np.float64,
    elevation: float,
    nameObs: str,
    sv: str,
    DCBs: np.float64,
    llaSat: dict,
    attrs: dict,
) -> xr.Dataset:
    """collect diverse datas and basic attributes into a Data set

    Args:
        TECs (np.ndarray): slant phase TEC (relative)
        time (np.ndarray): coordinate time
        lon (np.float64): coordinate longitude
        lat (np.float64): coordinate latitude
        elevation (float): elevation angle in degree from receiver to satellite
        nameObs (str): station name of receiver (observer)
        sv (str): satellite vehicle code
        DCBs (np.float64): Differential Code Bias of satellite plus receiver
        llaSat (dict): satellite geo positions of the day
        attrs (dict): other attributes

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
                    'geoLon': np.array(list(llaSat.values()))[:, 0],
                    'geoLat': np.array(list(llaSat.values()))[:, 1],
                    'geoHgt': np.array(list(llaSat.values()))[:, 2],
                },
                'data': [sv],
            },
            'time': {
                'dims': ('time',),
                'attrs': {'name': 'time'},
                'data': time,
            },
            'lon': {
                'dims': ('observer', 'sv'),
                'attrs': {
                    'name': 'longitude',
                    'units': 'degree_E',
                    'axis': 'X',
                },
                'data': [[lon]],
            },
            'lat': {
                'dims': ('observer', 'sv'),
                'attrs': {
                    'name': 'latitude',
                    'units': 'degree_N',
                    'axis': 'Y',
                },
                'data': [[lat]],
            },
            'elevation': {
                'dims': ('observer', 'sv'),
                'attrs': {
                    'name': 'elevation angle',
                    'units': 'degree',
                },
                'data': [[elevation]],
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
                'data': [[DCBs]],
            },
        },
        'attrs': attrs,
    }

    stec_dset = xr.Dataset.from_dict(stec_dict)

    del stec_smth
    return stec_dset


# %%
# -*- extract satellite position information -*-
def readOrbit(
    BDS: dict,
    fSp3: 'str',
    stryear: str,
    doy: str,
    tomlBDS: str,
    update_toml: bool = False,
) -> tuple[dict, dict]:
    """update the orbit information and extract position coords

    Args:
        BDS (dict): satellite information dict
        fSp3 (str): path of the 'extended Standard Product 3 orbit format' file
        stryear (str): str year
        doy (str): the day sequence of the year
        tomlBDS (str): path of the satellite configuration toml-file
        update_toml (bool, optional): Defaults to False.

    Returns:
        tuple[dict, dict]: xyz(m) postion dict and
           geo_position (lon, lat, altitude: degree, degree, m) dict
    """
    llaSat = BDS['geographic']
    xyzSat = BDS['cartesian']

    for sv in BDS['GEO']:
        lonSat, latSat, altSat = llaSat[sv]
        xSat, ySat, zSat = lla2ecef(lonSat, latSat, altSat)
        xyzSat[sv] = [xSat, ySat, zSat]

    # The Extended Standard Product 3 Orbit Format
    try:
        sp3 = grnx.load(fSp3)
        for sv in BDS['GEO']:
            if sv not in sp3.sv.values:
                continue
            else:
                pass

            xSat = (
                sp3['position'].sel(sv=sv, ECEF='x').values.mean() * const.kilo
            )  # unit: m
            ySat = (
                sp3['position'].sel(sv=sv, ECEF='y').values.mean() * const.kilo
            )  # unit: m
            zSat = (
                sp3['position'].sel(sv=sv, ECEF='z').values.mean() * const.kilo
            )  # unit: m

            xyzSat[sv] = [xSat, ySat, zSat]
            llaSat[sv] = list(
                ecef2lla(xSat, ySat, zSat)
            )  # return lla (unit: degree, degree, m)

            BDS['year'] = stryear
            BDS['doy'] = doy

            # update toml file
            if update_toml:
                with open(tomlBDS, 'wb') as ftoml:
                    tomlw.dump(BDS, ftoml)
    except Exception as e:
        print(e)

    return xyzSat, llaSat


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
    eleipp: dict,
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
    }
    for observer in TEC.observer:
        nameObs = TEC.observer.values[0]

        for sv in TEC.sel(observer=observer).sv.values:
            stec = (
                TEC['sTEC'].sel(observer=observer, sv=sv).dropna(dim='time', how='all')
            )

            DCBs = TEC['DCBs'].sel(observer=observer, sv=sv).values
            lon = TEC['lon'].sel(observer=observer, sv=sv).values
            lat = TEC['lat'].sel(observer=observer, sv=sv).values
            stec = stec - DCBs
            elevation = eleipp[sv]

            time = stec['time']

            ax.plot(
                time,
                stec * np.sin(np.radians(elevation)),
                color=colors[sv],
                marker=',',
                ls='',
                lw=1.5,
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
    satsys: str = 'BeiDou',
    sample_rate: str = '30s',
) -> tuple[xr.Dataset, str]:
    """MAIN PROCESSION: tec inversion for each day

    Args:
        show_plots (bool, optional): whether to show plots in the interactive window, defaults to False.
        reference_model (str, optional): correct DCB error refer to 'GIM' or 'IRI' model, defaults to 'GIM'.
        DCBplot(bool, optional): whether to show DCB result in plots.
    """
    # initials ------------------------------------------------------------------------
    tomlConfig = './config.toml'
    with open(tomlConfig, mode='rb') as f_toml:
        selfConfig = toml.load(f_toml)

    stations = selfConfig['stations']
    # year = selfConfig['year']
    # doys = selfConfig['doys']

    year = 2023
    doys = [doy for doy in range(1, 366)]

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

    tomlBDS = './BeiDou.toml'
    with open(tomlBDS, mode='rb') as f_toml:
        BDS = toml.load(f_toml)

    frqcSat = BDS['sig_frqc']
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

        ncBDS = Path(dirs['TEC'], f'{satsys}_TEC_{strdoy}_{sample_rate}_{stryear}.nc')

        print(f'Calculating tecs of DOY {strdoy} by Hatch filter')

        # filesRnx = sorted(dirs['RNX'].glob('*01D_01S_[M,C,J]O.*x'))
        filesRnx = sorted(dirs['RNX'].glob(f'*{strdoy}O.*d'))
        filesSp3 = dirs['SP3'].glob(f'*FIN*{stryear}{strdoy}*.SP3.gz')
        filesGim = dirs['GIM'].glob(f'CO*{stryear}{strdoy}FIN.nc')
        fSp3, fGim = sorted(filesSp3)[0], sorted(filesGim)[0]

        xyzSat, llaSat = readOrbit(
            BDS, fSp3, stryear, strdoy, tomlBDS, update_toml=False
        )

        for idxObs, fRnx in enumerate(filesRnx, start=1):
            nameObs = fRnx.stem[0:4]
            if stations == 'ALL':
                pass
            elif (nameObs.lower()[:2] not in stations) and (
                nameObs.lower() not in stations
            ):
                continue

            try:
                oVar = grnx.load(
                    fRnx, fast=False, use=['C'], meas=['C1', 'C7', 'L1', 'L7']
                )
                if len(oVar.sv.values) == 0:
                    continue
                else:
                    pass
                rObs = np.sqrt(np.dot(oVar.position, oVar.position))
                if rObs < 500 * const.kilo:
                    continue
                else:
                    attrs['samp_rate(s)'] = oVar.interval
                    attrs['time_sysm'] = oVar.time_system
                    attrs['rnx_model'] = oVar.rxmodel
            except Exception as ecode:
                print(
                    f'     {nameObs} - loading RNX file exited with {ecode}, skipped... '
                )
                continue

            svs = oVar.sv.values
            eleipp, llaipp = (
                {
                    sv: ippsProjection(
                        oVar.position,
                        xyzSat[sv],
                        Hionos * const.kilo,
                    )[0]
                    for sv in svs
                },
                {
                    sv: ippsProjection(
                        oVar.position,
                        xyzSat[sv],
                        Hionos * const.kilo,
                    )[1]
                    for sv in svs
                },
            )

            tec_existed = False
            for sv in svs:
                lon, lat, elevation = (
                    llaipp[sv][0],
                    llaipp[sv][1],
                    eleipp[sv],
                )  # units: degree, degree, degree

                print(f'\n{idxObs:3d}  {nameObs} - {sv} : {llaSat[sv]}')
                try:
                    data = oVar.sel(sv=sv).dropna(dim='time', how='any')

                    slantTEC = sTEC_of_sv(data, sv, frqcSat)
                    if slantTEC is None:
                        continue
                    else:
                        pass
                    if reference_model == 'GIM':
                        DCBs = sysmDCBs_byGimtec(
                            slantTEC,
                            lon,
                            lat,
                            elevation,
                            gimnc=fGim,
                            to_plot=DCBplot,
                        )
                    elif reference_model == 'IRI':
                        DCBs = sysmDCBs_byiri2020(slantTEC, lon, lat, elevation)
                    else:
                        print("WARNING: reference model must be one of 'GIM' | 'IRI'")
                        return

                except Exception as ecode:
                    print(f'      {satsys} - {sv} : calculate exited with {ecode}. \n')
                else:
                    print(f'      {satsys} - {sv} : calculate TEC successfully. \n')

                try:
                    BeiDouTEC = struct_TECs_of_stations(
                        slantTEC.values,
                        slantTEC.time.values,
                        lon,
                        lat,
                        elevation,
                        nameObs,
                        sv,
                        DCBs,
                        llaSat,
                        attrs,
                    )
                    if 'TECsingle' not in locals().keys():
                        TECsingle = BeiDouTEC
                    else:
                        TECsingle = xr.concat([TECsingle, BeiDouTEC], dim='sv')

                    del BeiDouTEC, slantTEC

                except Exception as ecode:
                    print(f'      {satsys} - {sv} : concat TEC exited with {ecode}. \n')
                else:
                    # print(f'      {satsys} - {sv} : concat TEC successfully. \n')
                    tec_existed = True

            if tec_existed:
                if 'TECall' not in locals().keys():
                    TECall = TECsingle
                else:
                    TECall = xr.concat([TECall, TECsingle], dim='observer')

                plotTEC_of_receiver(
                    TECsingle,
                    eleipp,
                    savedir=dirs['plots'],
                    satsys=satsys,
                    show_plots=show_plots,
                )
                del TECsingle
                gc.collect()
                print(
                    f'\n{idxObs:3d}  {nameObs} - ALL : concat TEC successfully. \n      '
                )

        TECall.to_netcdf(ncBDS)
        # del TECall
        # del doy, strdoy
        print('mission accomplished * *')
    return TECall, ncBDS
    # test()


# %%
if __name__ == '__main__':
    TECall, ncBDS = tec_inversion(
        show_plots=False, reference_model='GIM', DCBplot=False
    )

# %%
