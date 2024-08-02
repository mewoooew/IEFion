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
from datetime import datetime
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
# -*- calculate S4 for each satellite -*-
#  *  根据不同卫星信号的观测值计算S4
def S4_of_sv(
    obs: xr.Dataset, obsNAME: str, sv: str, sat_sysm: str = 'GPS'
) -> xr.Dataset:
    """slant TEC along path for each satellite-vehicle for each station

    Args:
        obs (xr.Dataset): observation variables
        obsNAME (str): GNSS receiver
        sv (str): satellite-vehicle code
        sat_sysm (str, optional): Defaults to 'GPS'.

    Returns:
        xr.Dataset: sTEC data set
    """
    print(f'\n      {sat_sysm} - {sv:3} : calculate S4 indices starts:\n')
    try:
        
        tStart = np.datetime64(obs['S1'].time.values[0], 'm')
        tEnd = np.datetime64(obs['S1'].time.values[-1], 'm')
        tnodes = pd.date_range(tStart, tEnd, freq='1min')
        S4_indices = np.array([])
        S4_time = np.array([])
        for tnode0 in tnodes:
            tnode1 = tnode0 + np.timedelta64(1, 'm')
            CNR = obs['S1'].loc[tnode0:tnode1].values  #unit: dBHz
            if len(CNR) < 10:
                continue
            else:
                pass
            CNR = 10**(0.1*CNR)
            S4 = np.sqrt((np.mean(CNR**2)-np.mean(CNR)**2)/np.mean(CNR)**2)
            S4_indices = np.hstack((S4_indices, S4))
            S4_time = np.hstack((S4_time, tnode0))
        
    except Exception as ecode:
        print(
            f'{sat_sysm:>12s} - {sv:3} : calculate S4 indices exited with {ecode}. \n'
        )
    else:
        print(f'{sat_sysm:>12s} - {sv:3} : calculate S4 indices successfully. \n')
        return S4_indices, S4_time

# %%
# -*- build up Dataset
def struct_S4_of_stations(
    S4indices: np.ndarray,
    time: np.ndarray,
    lons: list[float],
    lats: list[float],
    elevations: list[float],
    nameObs: str,
    sv: str,
    attrs: dict,
) -> xr.Dataset:
    """collect diverse datas and basic attributes into a Data set

    Args:
        TECs (np.ndarray): slant phase TEC (relative)
        time (np.ndarray): coordinate time
        lons (list[float]): coordinate longitude
        lats (list[float]): coordinate latitude
        elevations (list[float]): elevation angle in degree from receiver to satellite
        nameObs (str): station name of receiver (observer)
        sv (str): satellite vehicle code
        DCBs (np.float64): Differential Code Bias of satellite plus receiver
        attrs (dict): other attributes

    Returns:
        xr.Dataset: complete sTEC data set of the sv for the station (receiver)
    """
    S4indices = S4indices.reshape(1, 1, -1)
    S4_dict = {
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
            'lon': {
                'dims': ('observer', 'sv', 'time'),
                'attrs': {
                    'name': 'longitude',
                    'units': 'degree_E',
                    'axis': 'X',
                },
                'data': [[lons]],
            },
            'lat': {
                'dims': ('observer', 'sv', 'time'),
                'attrs': {
                    'name': 'latitude',
                    'units': 'degree_N',
                    'axis': 'Y',
                },
                'data': [[lats]],
            },
        },
        'data_vars': {
            'sTEC': {
                'dims': ('observer', 'sv', 'time'),
                'attrs': {
                    'name': 'S4',
                    'long_name': 'ionospheric scitillation indices',
                    'units': 'TECU',
                },
                'data': S4indices,
            },
            'elevation': {
                'dims': ('observer', 'sv', 'time'),
                'attrs': {
                    'name': 'elevation angle',
                    'long_name': 'satellite elevation angle to IPP',
                    'units': 'degree',
                },
                'data': [[elevations]],
            },
        },
        'attrs': attrs,
    }

    s4_dset = xr.Dataset.from_dict(S4_dict)

    del S4_dict
    return s4_dset


# %%
# # -*- extract satellite position information -*-
def readOrbit(
    xyzObs: tuple[float],
    xyzsSat: np.ndarray,
    timeSat: np.ndarray,    # [np.datetime64]
    obsTIME: np.ndarray,    # [np.datetime64]
    Hionos: float = 350.0,    # unit: km
) -> tuple[list, list, list]:
    
    infoipp = [ippsProjection(
        xyzObs,
        xyzSat,
        Hionos * const.kilo,
        )
        for xyzSat in xyzsSat
        ]
    infoipp = np.array(infoipp)
    elesipp = xr.DataArray(
        infoipp[:,0], 
        dims=('time',), 
        coords={'time':timeSat}
        ).interp(time=obsTIME, method='cubic', kwargs={'fill_value': 'extrapolate'}).values
    xsipp, ysipp, zsipp = lla2ecef(
        infoipp[:,1], infoipp[:,2], infoipp[:,3])
    xsipp = xr.DataArray(
        xsipp, 
        dims=('time',), 
        coords={'time':timeSat}
        ).interp(time=obsTIME, method='cubic', kwargs={'fill_value': 'extrapolate'})
    ysipp = xr.DataArray(
        ysipp, 
        dims=('time',), 
        coords={'time':timeSat}
        ).interp(time=obsTIME, method='cubic', kwargs={'fill_value': 'extrapolate'})
    zsipp = xr.DataArray(
        zsipp, 
        dims=('time',), 
        coords={'time':timeSat}
        ).interp(time=obsTIME, method='cubic', kwargs={'fill_value': 'extrapolate'})
    lonsipp, latsipp, _ = ecef2lla(
        xsipp.values,ysipp.values,zsipp.values
        )
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
        size=7)
    return config, fonts


# %%
def plotS4_of_receiver(
    S4: xr.Dataset,
    savedir: str,
    satsys: str = 'BeiDou',
    show_plots: bool = False,
) -> None:
    """plot tecs of diverse satellites for each station (receiver)

    Args:
        S4 (xr.Dataset): S4
        savedir (str): path for plot output
        satsys (str, optional): _Defaults to 'BeiDou'.
        show_plots (bool, optional): Defaults to False, not to show plots in interactive window .
    """
    config, fonts = plot_configurations()
    plt.rcParams.update(config)

    fig = plt.figure(figsize=(4, 1.6), dpi=300)
    ax = fig.add_subplot(111)
    
    for observer in S4.observer:
        nameObs = S4.observer.values[0]

        for sv in S4.sel(observer=observer).sv.values:
            s4indices = (
                S4['sTEC'].sel(observer=observer, sv=sv).dropna(dim='time', how='all')
            )

            lons = (
                S4['lon'].sel(observer=observer, sv=sv).dropna(dim='time', how='all')
            )
            lats = (
                S4['lat'].sel(observer=observer, sv=sv).dropna(dim='time', how='all')
            )
            

            # time = stec['time']

            ax.scatter(
                lons,
                lats,
                c=s4indices.values,
                cmap='ncar_gist', 
                marker=',',
                s=0.2,
                vmax=1.,
                vmin=0.,
            )

    # plt.legend(labelcolor='linecolor', markerscale=50, loc='upper right')
    plt.xlabel('lon', fontproperties=fonts)
    plt.ylabel('lat', fontproperties=fonts)
    plt.title(f'{nameObs} {satsys} S4', fontproperties=fonts)

    plt.savefig(f'{savedir}/{satsys}_s4_{nameObs}.png', bbox_inches='tight')

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
def calculate_S4_indices(show_plots: bool = False, 
                  ) -> tuple[xr.Dataset, str]:
    """MAIN PROCESSION: tec inversion for each day

    Args:
        show_plots (bool, optional): whether to show plots in the interactive window, defaults to False.
        reference_model (str, optional): correct DCB error refer to 'GIM' or 'IRI' model, defaults to 'GIM'.
        DCBplot(bool, optional): whether to show DCB result in plots.
    """
    # initials ------------------------------------------------------------------------
    satsys, sample_rate = 'MGNSS', '1min'

    tomlConfig = './config.toml'
    with open(tomlConfig, mode='rb') as f_toml:
        selfConfig = toml.load(f_toml)

    stations = selfConfig['stations']
    year = selfConfig['year']
    doys = selfConfig['doys']
    if doys=='auto':
        year = int(datetime.today().strftime('%Y'))
        doys = [int(datetime.today().strftime('%j'))-1]
    else:
        pass
    stryear = f'{year}'
    Hionos = selfConfig['Hionos']  # unit: km

    srcDisk = Path(selfConfig['source'])
    archives = selfConfig['archives']
    archGnss, archSp3 = (
        archives['dataGnss'],
        archives['dataSp3']
    )
    dirData, dirSp3 = srcDisk / archGnss, srcDisk / archSp3

    dirs = {}
    dirs['SP3'] = dirSp3 / stryear

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
        strdoy = f'{doy:>3.0f}'
        mk_archives(dirData / archTEC / stryear, [strdoy])
        mk_archives(dirData / archTEC / stryear / strdoy, [archPlots])
    del doy, strdoy

    
    attrs = {
        'sat_sysm': satsys,
        'Hionos(km)': Hionos,
    }

    # run cycles ----------------------------------------------------------------------
    for doy in doys:
        strdoy = f'{doy:0>3.0f}'
        dirs['RNX'] = dirData / archRnx / stryear / strdoy
        dirs['TEC'] = dirData / archTEC / stryear / strdoy
        dirs['plots'] = dirData / archTEC / stryear / strdoy / archPlots

        ncS4_GPS = Path(dirs['TEC'], f'{satsys}_TEC_{strdoy}_{sample_rate}_{stryear}.nc')

        print(f'Calculating tecs of DOY {strdoy} by Hatch filter')

        filesRnx = sorted(dirs['RNX'].glob('*01D_01S_*O.*x'))
        filesSp3 = dirs['SP3'].glob(f'*{stryear}{strdoy}*.SP3.gz')
        fSp3 = sorted(filesSp3)[0]
        sp3 = grnx.load(fSp3)
        
        for idxObs, fRnx in enumerate(filesRnx, start=1):
            nameObs = fRnx.stem[0:4]
            if stations == 'ALL':
                pass
            elif (nameObs.lower()[:2] not in stations) and (
                nameObs.lower() not in stations
            ):
                continue

            try:
                oVar = grnx.load(fRnx, fast=False, use={'G', 'C'}, meas=['S1'])
                rObs = np.sqrt(np.dot(oVar.position, oVar.position))
                if len(oVar.sv.values) == 0:
                    continue
                else:
                    pass
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
            
            tec_existed = False
            for sv in svs:
                
                try:
                    data = oVar.sel(sv=sv).dropna(dim='time', how='any')
                    if data.time.values.shape[0] < 30:
                        continue
                    else:
                        pass
                    
                    xyzsSat = (
                        sp3['position'].sel(sv=sv).values * const.kilo
                    )  # unit: m
                    
                    timeSat = sp3['position'].sel(sv=sv).time.values
                    
                    
                    sigIntens, S4time = S4_of_sv(data, nameObs, sv)
                    elesipp, lonsipp, latsipp = readOrbit(
                        oVar.position, xyzsSat, timeSat, S4time, Hionos
                        )
                    S4indices = struct_S4_of_stations(
                        sigIntens,
                        S4time,
                        lonsipp,
                        latsipp,
                        elesipp,
                        nameObs,
                        sv,
                        attrs,
                    )
                    if 'S4single' not in locals().keys():
                        S4single = S4indices
                    else:
                        S4single = xr.concat([S4single, S4indices], dim='sv')
                    del sigIntens

                except Exception as ecode:
                    print(f'      {satsys} - {sv} : concat TEC exited with {ecode}. \n')
                else:
                    # print(f'      {satsys} - {sv} : concat TEC successfully. \n')
                    tec_existed = True

            if tec_existed:
                if 'S4all' not in locals().keys():
                    S4all = S4single
                else:
                    S4all = xr.concat([S4all, S4single], dim='observer')
                plotS4_of_receiver(
                    S4single,
                    savedir=dirs['plots'],
                    satsys=satsys,
                    show_plots=show_plots,
                )
                del S4single
                gc.collect()
                print(
                    f'\n{idxObs:3d}  {nameObs} - ALL : concat TEC successfully. \n      '
                )
        S4all.to_netcdf(ncS4_GPS)
        # del TECall
    del doy, strdoy
    print('mission accomplished * *')
    return S4all, ncS4_GPS
    # test()


# %%
if __name__ == '__main__':
    S4all, ncS4_GPS = calculate_S4_indices(show_plots=False) 

# %%
