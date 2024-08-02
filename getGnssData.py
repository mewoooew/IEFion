# %%
#!./.venv/bin python3
# -*- encoding: utf-8 -*-
"""
@filename :  getGnssData.py
@desc     :  update SP3(orbit) files and IONEX(gim tec) files of the day
@time     :  2024/06/01
@author   :  _koii^_ (Liu Hong), Institute of Earthquake Forecasting (IEF), CEA.
@Version  :  1.0
@Contact  :  koi_alley@outlook.com
"""

# here import libs
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import sh
import tomllib as toml
import xarray as xr


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
# -*-  -*-
def get_rnx_form_geonet(dest: Path, year: int, doy: int) -> None:
    """get GNSS RINEX data from New Zealand GeoNet shared data center

    Args:
        dest (Path): destination directory
        year (int): year
        doy (int): day of year
    """
    stryear, strdoy = f'{year}', f'{doy:0>3.0f}'
    url = 'https://data.geonet.org.nz/gnss/rinex/'  # rinex1Hz
    url = '/'.join([url, stryear, strdoy])

    fSite = dest / 'geonet_info.txt'
    if fSite.is_file():
        pass
    else:
        sh.curl('-o', fSite, url)

    with open(fSite, 'r') as f_read:
        for line in f_read:
            regex = re.compile(r'<a href')
            ext = re.compile(r'.gz')

            if re.search(regex, line) and re.search(ext, line):
                rnxName = line.split('>')[1]
                rnxName = rnxName.split('<')[0]
                f_output = '/'.join([dest, rnxName])
                f_remote = '/'.join([url, rnxName])

                if f_output.is_file():
                    print(rnxName, 'existed, skipping...')
                    continue

                # with open(f_output, "wb") as f:
                #     for chunk in r.iter_content(chunk_size=1000):
                #         f.write(chunk)
                # f.close()

                sh.wget('-c', '-b', f_remote, '-P', dest)
                print(rnxName, ' get')
    return


# %%
# -*-  -*-
def get_rnx_form_auscors(dest: str, year: int, doy: int) -> None:
    """get GNSS RINEX data from Austrilian Continuous Observation Reference Network

    Args:
        dest (str): destination directory
        year (int): year
        doy (int): day of year
    """
    stryear, strdoy = f'{year}', f'{doy:0>3.0f}'

    fSite = '/run/media/echoo/koidata/AUSCORS/auscors.txt'
    f = open(fSite, 'r')
    namelist = f.readlines()

    url = 'https://ga-gnss-data-rinex-v1.s3.amazonaws.com/public/daily'
    url = '/'.join([url, stryear, strdoy])
    user, passwd = 'anonymous', 'morecho99@gmail.com'

    for idxl, line in enumerate(namelist):
        rnxName = line.split('\t')[0]
        f_output, f_remote = '/'.join([dest, rnxName]), '/'.join([url, rnxName])

        if Path(f_output).is_file():
            continue

        print(idxl, f_output)
        sh.wget(
            f'--user={user}', f'--password={passwd}', '-c', '-b', f_remote, '-P', dest
        )
    return


# %%
# -*-  -*-
def get_eph_form_CDDIS(dest: Path, GPSweek: np.int64, dayseq: int) -> Path:
    """get satellite ephemeris files from Crustal Dynamics Data Information System

    Args:
        dest (Path): destination directory
        GPSweek (np.int64): GPS week mode
        dayseq (int): Defaults to 7, day of GPS week starting from 0

    Returns:
        Path: _description_
    """
    filename = ''.join(['igs', str(GPSweek), str(dayseq), '.sp3.Z'])
    destf = dest / filename

    if Path.is_file(destf):
        print(f'ephfile: {filename} existed...')
        return destf

    url = 'https://cddis.nasa.gov/archive/gnss/products'
    path = '/'.join([url], str(GPSweek), filename)
    r = requests.get(path)

    with open(destf, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1000):
            f.write(chunk)
    f.close()

    return destf


# %%
# -*-  -*-
def read_ionFile(
    IonFile: str,
    iontype: str = 'code',
    ncpath: str = '/run/media/echoo/koidata/IGSgim/nc',
    rnxtype: str = 'new',
    tag: str = '',
) -> str:
    """extract GIM TEC data from IONEX files
        this piece of codes from https://github.com/GNSSpy-Project/gnsspy/

    Args:
        IonFile (str): IONEX files, in which TEC in unit of 0.1TECU
        iontype (str, optional): Defaults to 'code',
            the specific provider from one of IGS institutes
        ncpath (str, optional): Defaults to '/run/media/echoo/koidata/IGSgim/nc',
            directory for output nc file
        rnxtype (str, optional): Defaults to 'new', RINEX version.

    Returns:
        str: path of output nc file
    """

    fionex = open(IonFile, errors='ignore')
    obsLines = fionex.readlines()

    line = 0
    while True:
        if 'END OF HEADER' in obsLines[line]:
            line += 1
            break
        else:
            line += 1
    del obsLines[0:line]
    # --------------------------------------------------------------------------------------
    nTIME = 25 if iontype == 'code' else 13

    gtecList = np.zeros([nTIME, 71, 72])
    for etime in range(nTIME):
        line = 0
        del obsLines[0:2]  # delete epoch line
        for phi in range(71):
            gtec = obsLines[1] + obsLines[2] + obsLines[3] + obsLines[4] + obsLines[5]
            gtec = gtec.split()
            gtec = [int(tec) for tec in gtec]
            for lamda in range(len(gtec) - 1):
                gtecList[etime, phi, lamda] = gtec[lamda]
            del obsLines[0:6]
        del obsLines[0]
        # ---------------------------------------------------------------------------------------
    fionex.close()  # close the file

    ionNAME, ionSUFFIX = Path(IonFile).stem, Path(IonFile).suffix
    # freq = ionNAME[-6:-4].lower()
    ltec = gtecList * 0.1  # ion data unit 0.1TECU => 1TECU
    institute = ionNAME[:4].upper()

    doy = int(ionNAME[15:18]) if rnxtype == 'new' else int(ionNAME[4:7])
    year = int(ionNAME[11:15]) if rnxtype == 'new' else int(ionSUFFIX[-3:-1]) + 2000

    base = datetime(year, 1, 1)
    date = base + timedelta(doy - 1)

    nEPOCH, nLAT, nLON = ltec.shape[0], ltec.shape[1], ltec.shape[2]

    LATS = np.linspace(87.5, -87.5, nLAT)
    LONS = np.linspace(-180, 180, nLON)

    tstart = np.datetime64(date.strftime('%Y-%m-%dT00:00:00'))
    tend = tstart + np.timedelta64(1, 'D')
    TIME = pd.date_range(tstart, tend, periods=nEPOCH)
    gimtec = xr.DataArray(
        ltec,
        coords=[TIME, LATS, LONS],
        dims=(
            'time',
            'lat',
            'lon',
        ),
        attrs={'name': 'gimtec', 'institute': institute, 'unit': 'TECU'},
    )

    if ncpath:
        gimnc = f'{ncpath}/{institute}{year}{doy:>03.0f}{tag}.nc'
        if Path(gimnc).is_file():
            print('ncfile already existed...')
        else:
            gimtec.to_netcdf(gimnc)
    else:
        pass
    return gimnc


# %%
# -*-  -*-
def get_ionex_from_igsWHU(
    dest: int, year: int, doy: int, ionexMode: str = 'RAP'
) -> str:
    """get IONEX files from IGS center of Wuhan University

    Args:
        dest (int): directory for output ionex file
        year (int): year
        doy (int): day of year
        ionexMode (str, optional): RAP:rapid | FIN:final.

    Returns:
        str: path of output unzipped ionex file
    """
    stryear, strdoy = f'{year}', f'{doy:0>3.0f}'

    if (year >= 2022 and doy >= 331) or (year >= 2023):
        ionexName = f'COD0OPS{ionexMode}_{stryear}{strdoy}0000_01D_01H_GIM.INX.gz'  #
    else:
        ionexName = f'codg{strdoy}0.{stryear[-2:]}i.Z'  #

    dest = '/'.join([dest, stryear])
    Path(dest).mkdir(mode=0o777, parents='True', exist_ok=True)
    destf = '/'.join([dest, ionexName])
    if Path(destf).is_file():
        print(f'ionex file: {ionexName} existed...')

        if (year >= 2022 and doy >= 331) or (year >= 2023):
            return destf.replace('.gz', '')
        else:
            return destf.replace('.Z', '')

    remoteServ = 'ftp://'
    remoteHost = 'igs.gnsswhu.cn'
    remoteDir = f'/pub/gps/products/ionex/{stryear}/{strdoy}'
    remoteFile = '/'.join([remoteDir, ionexName])
    user, passwd = 'anonymous', ''

    # shell command:
    # lftp ftp://anonymous:@igs.gnsswhu.cn -e
    # \ "lcd dest; mget /pub/gps/products/ionex/2023/002/COD0OPSFIN_20230020000_01D_01H_GIM.INX.gz; quit"
    try:
        sh.lftp(
            f'{remoteServ}{user}:{passwd}@{remoteHost}',
            '-e',
            f'lcd {dest}; mget {remoteFile}; quit',
        )
        if Path(destf).is_file():
            sh.gzip('-d', destf)
    except sh.ErrorReturnCode as e:
        print(e)

    if (year >= 2022 and doy >= 331) or (year >= 2023):
        return destf.replace('.gz', '')
    else:
        return destf.replace('.Z', '')


# %%
# -*-  -*-
def get_fOrbit_from_igsWHU(
    dest: str,
    year: int,
    doy: int,
    instiIGS: str = 'WHU',
    sp3Mode: str = 'ULT',  # RAP:rapid | FIN:final | ULT
) -> Path:
    """get SP3 Orbit files from IGS center of Wuhan University

    Args:
        dest (int): directory for output ionex file
        year (int): year
        doy (int): day of year
        instiIGS (str, optional): Defaults to 'WHU', provider from IGS cenetrs
        sp3Mode (str, optional): Defaults to 'ULT', RAP:rapid | FIN:final | ULT

    Returns:
        Path: path of output zipped sp3 file
    """
    gpsWeek, _, weekday = gpsWeekOf(year, doy)
    stryear, strdoy = f'{year}', f'{doy:0>3.0f}'

    if instiIGS == 'WHU':
        if gpsWeek < 1966:  # 2017/252
            print('warning: too early year to search...')
        elif gpsWeek < 2034:  # 2019/001
            sp3Name = f'wum{gpsWeek}{weekday}.sp3.Z'
        else:
            sp3Name = f'WUM0MGX{sp3Mode}_{stryear}{strdoy}0000_02D_*_ORB.SP3.gz'
    elif instiIGS == 'GFZ':
        if gpsWeek < 2038:
            sp3Name = f'gbm{gpsWeek}{weekday}.sp3.Z'
        else:
            sp3Name = f'GFZ0MGX{sp3Mode}_{stryear}{strdoy}0000_01D_*_ORB.SP3.gz'

    dest = Path(dest, stryear)
    dest.mkdir(mode=0o777, parents='True', exist_ok=True)
    destf = dest / sp3Name
    if Path.is_file(destf):
        print(f'ionex file: {sp3Name} existed...')
        return destf

    remoteServ, remoteHost = 'ftp://', 'igs.gnsswhu.cn'
    remoteDir = f'/pub/gps/products/mgex/{gpsWeek}/'
    remoteFile = f'{remoteDir}{sp3Name}'
    user, passwd = 'anonymous', ''

    try:
        # shell command:
        # lftp ftp://anonymous:@igs.gnsswhu.cn -e
        # \ "lcd dest; mget /pub/gps/products/mgex/2315/WUM0MGXULT_20230020000_01D_05M_ORB.SP3.gz; quit"

        sh.lftp(
            f'{remoteServ}{user}:{passwd}@{remoteHost}',
            '-e',
            f'lcd {dest}; mget {remoteFile}; quit',
        )
        if Path.is_file(destf):
            sh.gzip('-d', destf)
    except sh.ErrorReturnCode as e:
        print(e)

    return destf


# %%
# -*-  -*-
def update_sp3_and_ionex_files() -> None:
    # tomlConfig = './config.toml'
    tomlConfig = '/home/echoo/codePy/IEFion/config.toml'
    with open(tomlConfig, mode='rb') as f_toml:
        selfConfig = toml.load(f_toml)

    year = selfConfig['year']
    doys = selfConfig['doys']

    # year = 2024
    # doys = [doy for doy in range(182, 183)]

    if doys == 'auto':
        year = int(datetime.today().strftime('%Y'))
        doys = [int(datetime.today().strftime('%j')) - 1]
    else:
        pass
    archives = selfConfig['archives']
    dirSP3 = '/'.join([selfConfig['source'], archives['dataSp3']])
    dirGIM = '/'.join([selfConfig['source'], archives['dataGim']])
    dirGIM_nc = '/'.join(
        [selfConfig['source'], archives['dataGim'], archives['dataNC'], f'{year}']
    )

    for doy in doys:
        # print(doy)
        destf = get_fOrbit_from_igsWHU(dirSP3, year, doy, instiIGS='WHU', sp3Mode='NRT')
        print(destf)

    for doy in doys:
        # print(doy)
        destf = get_ionex_from_igsWHU(dirGIM, year, doy, ionexMode='RAP')
        print(destf)
        gimnc = read_ionFile(
            destf, iontype='code', ncpath=dirGIM_nc, rnxtype='new', tag='RAP'
        )
        print(gimnc)
    return


# %%
# -*-  -*-
if __name__ == '__main__':
    update_sp3_and_ionex_files()


# %%
