# %%
#!./.venv/bin python3
# -*- encoding: utf-8 -*-
"""
@filename :  convert2Rinex_mprocess.py
@desc     :  multiprocess Trimble T02 data(1Hz) into RINEX format rapidly for each day
@time     :  2024/06/01
@author   :  _koii^_ (Liu Hong), Institute of Earthquake Forecasting (IEF), CEA.
@Version  :  1.0
@Contact  :  koi_alley@outlook.com
"""

# here import libs
import re
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path

import sh
import tomllib as toml


# %%
# -*- check exectable binary files if pre-installed -*-
#  *  预安装程序检查
def is_Commands_sh(Command: str) -> None:
    """check if fbinary installed

    Args:
        Command (str, optional): one of 'runpkr' | 'teqc' | 'gfzrnx' | 'lftp').
    """
    try:
        if Command == 'runpkr00':
            sh.runpkr00()
        if Command == 'teqc':
            sh.teqc('+help')
        if Command == 'gfzrnx':
            sh.gfzrnx('--version')
        if Command == 'lftp':
            sh.lftp('--help')
    except sh.ErrorReturnCode as e:
        print(f'Command {e.full_cmd} exited with {e.exit_code}')

    return


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


def new_txts(srcdir: str, fnames: list[str], extension: str) -> list[Path]:
    foutpus = {}
    for fname in fnames:
        fpath = Path(f'{srcdir}/{fname}.{extension}')
        if not Path.is_file(fpath):
            fpath.touch()
        foutpus[fname] = fpath
    return foutpus


def move_files_in(dirDispos: Path, dirCache: Path, item: str = 'file') -> None:
    """p.s. files moved in dirCache need to be deleted manually later

    Args:
        dirDispos (Path):
        dirCache (Path):
        item (str, optional): Defaults to 'file'.
    """
    for file in dirDispos.glob('*'):
        if item is None:
            continue
        elif item == 'file':
            if file.is_file():
                sh.mv('-f', file, dirCache)
        elif item == 'dir':
            if file.is_dir():
                sh.mv('-f', file, dirCache)
        else:
            print('WARNING: item must be one of file|dir|None !')
    return


def rm_files_in(dirDispos: Path, item: str = 'file') -> None:
    """p.s. files moved in dirCache need to be deleted manually later

    Args:
        dirDispos (Path):
        item (str, optional): Defaults to 'file'.
    """
    for file in dirDispos.glob('*'):
        if item is None:
            continue
        elif item == 'file':
            if file.is_file():
                sh.rm('-f', file)
        elif item == 'dir':
            if file.is_dir():
                sh.rm('-rf', file)
        else:
            print('WARNING: item must be one of file|dir|None !')
    return


def rnx2crx_files_in(dirRnx):
    """p.s. compact rnx files to crx

    Args:
        dirRnx (Path):
    """
    for file in dirRnx.glob('*.rnx'):
        if file.is_file():
            sh.rnx2crx(file)
            sh.rm('-f', file)
        else:
            print('WARNING: item must be RNX FILE !')
    return


# %%
# -*- runpkr00: TGD -> T02 -*-
#  *  将天宝接收器T02数据文件转换为二进制数据文件TGD
def run_runpkr00(
    dirT02: Path, dirTGD: Path, obsHOUR: str, INFO: str, stations: str
) -> str:
    hr_dirT02 = Path(dirT02, obsHOUR)
    hr_dirTGD = Path(dirTGD, obsHOUR)
    for count, fT02 in enumerate(hr_dirT02.glob('*.t02'), start=1):
        obsName = fT02.stem[:4]
        if stations == 'ALL':
            pass
        elif (obsName[:2] not in stations) and (obsName not in stations):
            continue

        fTGD = dirTGD / obsHOUR / f'{fT02.stem}.tgd'
        if fTGD.is_file():
            continue

        try:
            sh.runpkr00('-d', '-g', fT02, hr_dirTGD)
        except sh.ErrorReturnCode as e:
            print(
                f'::: {obsHOUR}th hour - {obsName}: converting skipped with {e.exit_code}. \n'
            )

    return f'{obsHOUR}th hour - {INFO} data processed \n'


def convert_TrimbleT02toTGD_runpkr00(
    hours: dict[str(int) : str],
    dirT02: Path,
    dirTGD: Path,
    stations: str,
    poolCOUNT: int = 4,
) -> None:
    """_summary_

    Args:
        hours (dict[str(int) : str]): {'00':'a', '01':'b', '02':'c', ...} from T02 name rule
        dirT02 (Path): /source_directory/T02_archive/year/doy
        dirTGD (Path): /source_directory/TGD_archive/year/doy
        stations (str): 'ALL' or obsNAME str list
        poolCOUNT (int, optional): Defaults to 4. Multiprocessing pool count, must be smaller than cpucore count,
                                   known as cpu_count()
    """

    INFO = 'runpkr00 - trimble T02 converting to TGD binary'
    print(INFO)

    keyHOURS = sorted(hours.keys())
    mk_archives(dirTGD, keyHOURS)

    for idx, _ in enumerate(keyHOURS):
        if idx % poolCOUNT == 0:
            pass
        else:
            continue

        pool = Pool(poolCOUNT)
        arguments = [
            (dirT02, dirTGD, obsHOUR, INFO, stations)
            for obsHOUR in keyHOURS[idx : idx + poolCOUNT]
        ]
        results = [
            pool.apply_async(run_runpkr00, args=argument) for argument in arguments
        ]
        [print('::: ', res.get()) for res in results]

        pool.close()
        pool.join()
    return


# # test 12m 41.1s
# convert_TrimbleT02toTGD_runpkr00(
#     hours={f'{key:02}': chr(key + ord('a')) for key in range(0, 24)},
#     dirT02=Path('/run/media/echoo/koidata/CMONOC/raw1s/2023/352'),
#     dirTGD=Path('/run/media/echoo/koidata/CMONOC/TGD/2023/352'),
#     stations='ALL',
#     poolCOUNT=4,
# )


# %%
# -*- teqc: extract geo position -*-
#  *  对TGD文件使用质量核检功能qc计算出站点静地坐标
#  *  存入文本文件以供后期GFZRNX转换RNX文件时填充站点位置信息
def cal_xyzObs_teqc(
    dirTGD: Path,
    f_qcheck: Path,
    f_update: Path,
) -> None:
    """teqc: extract observation position info from TGDfiles and write in f_update,
             f_update will be used by gfzrnx when converting RINEXfiles

    Args:
        dirTGD (Path): /source_directory/TGD_archive/year/doy
        f_qcheck (Path): /source_directory/T02_archive/year/doy/f_qcheck.txt
        f_update (Path): /source_directory/T02_archive/year/doy/f_update.txt
    """
    INFO = 'teqc - calculate obs_ecef position'

    print(INFO)

    with open(f_update, 'w') as fUpdate:
        fUpdate.write(
            'update_insert:\n'
            '#-------------\n'
            '  O - ALL:\n'
            '    "OBSERVER / AGENCY": { '
            '0 : "GNSS Observer", '
            '1 : "Trimble" }\n'
            '\n'
            '  N - ALL:\n'
            '    "LEAP SECONDS": { 0 : "4" }\n'
        )

    for count, fTGD in enumerate(dirTGD.rglob('**/*d00.tgd'), start=1):
        try:
            sh.teqc(
                '+qc',
                '+dm',
                '3',
                '+C1-5',
                '+G',
                '+R',
                '+J',
                '-no_orbit',
                'S+J+I',
                '-rep',
                #                     '+plot', '-ion', '-iod', '-mp', '-sn',
                fTGD,
                _out=f_qcheck,
            )
        except sh.ErrorReturnCode as e:
            print(f'{count:>7d} {INFO} - {fTGD.stem}: exited with {e.exit_code}. \n')

        with open(f_qcheck, 'r') as f_qcinfo, open(f_update, 'a') as fUpdate:
            for line in f_qcinfo:
                regex = re.compile(r'4-character ID')

                if re.search(regex, line):
                    obsName = line.split(':')[1].split('(')[0]
                    # with open(f_update, 'a') as f_RNXupdate:
                    fUpdate.write(f'\n  O - {obsName}:\n')

                regex = re.compile(r'antenna WGS 84 [(]xyz[)]')
                if re.search(regex, line):
                    obsPosi = line.split(':')[1].split('(')[0].split(' ')
                    # with open(f_update, 'a') as fUpdate:
                    fUpdate.write(
                        f'    "APPROX POSITION XYZ": '
                        f'{{ 0: {obsPosi[1]}, '
                        f'1: {obsPosi[2]}, '
                        f'2: {obsPosi[3]} }}\n'
                    )

                    print(
                        f"{count:>7d} - {INFO}: \n"
                        f"{'':>8s} {obsName}- WGS84 xyz: {obsPosi[1:4]} .\n"
                    )
                    break
    return


# %%
# -*- teqc: TGD -> RINEX2 -*-
def run_teqcConvert(
    obsYEAR: str,
    obsDOY: str,
    obsHOUR: str,
    dirTGD: Path,
    dirRNX: Path,
    dirNAV: Path,
    INFO: str,
) -> str:
    hr_dirTGD = Path(dirTGD, obsHOUR)
    hr_dirRNX = Path(dirRNX, obsHOUR)
    hr_dirNAV = Path(dirNAV, obsHOUR)
    resINFO = []
    resINFO.append(f'\n  {obsHOUR} - STA - {INFO}')
    for count, fTGD in enumerate(hr_dirTGD.glob('*.tgd'), start=1):
        fname = fTGD.stem
        fNAV2 = Path(hr_dirNAV, f'{fname}.{obsYEAR}n')  # GPS nav file extention: .yrn
        fRNX2 = Path(hr_dirRNX, f'{fname}.{obsYEAR}o')
        if Path.is_file(fRNX2):
            continue

        try:
            sh.teqc(
                '-week',
                f'20{obsYEAR}/{obsDOY}',
                '-tr',
                'd',
                '+C1-5',
                '+G',
                '-R',
                '-S',
                '-E',
                '+J7',
                '-I',
                '+P',
                '+C2',
                '+L2',
                '+L1_2',
                '+L2_2',
                '-CA_L1',
                '-L2C_L2',
                '-SA_G1',
                '-SA_G2',
                '+L5',
                '-L6',
                '+L7',
                '-L8',
                '+relax',
                # '-set_mask', '30', \
                # '+nav',
                # f'{fNAV2},-,-,-,-,-',
                fTGD,
                _out=fRNX2,
            )

        except sh.ErrorReturnCode as e:
            resINFO.append(
                f'{count:>10d} - {fTGD.stem}: converting exited with {e.exit_code}. '
            )

        try:
            sh.teqc('+v', f'{fRNX2}')
        except sh.ErrorReturnCode as e:
            fRNX2.unlink()
            resINFO.append(
                f'{count:>10d} - {fRNX2.stem}: RNX format is wrong with {e.exit_code}, DELETED.'
            )
    resINFO.append(f'\n  {obsHOUR} - {(count+3)//4:3d} - {INFO}')
    return resINFO


def convert_TGDtoRNX2_teqc(
    obsYEAR: str,
    obsDOY: str,
    HOURS: dict[str:str],
    dirTGD: Path,
    dirRNX: Path,
    dirNAV: Path,
    poolCOUNT: int = 4,
) -> None:
    """teqc: convert TGDfiles to RINEX2files

    Args:
        obsYEAR (str): the last two character of stryear  i.e. '23'('2023'), '24'('2024')
        obsDOY (str): the sequence day in the year of observation date
        HOURS (dict[str:str]): {'00':'a', '01':'b', '02':'c', ...} from T02 name rule
        dirTGD (Path): /source_directory/TGD_archive/year/doy
        dirRNX (Path): /source_directory/RNX_archive/year/doy
        dirNAV (Path): /source_directory/NAV_archive/year/doy
        poolCOUNT (int, optional): Defaults to 4. Multiprocessing pool count, must be smaller than cpucore count,
                                   known as cpu_count()
    """
    if not Path.is_dir(dirTGD):
        print(f'\n  {dirTGD} not existing, skipping...')
        return
    INFO = 'teqc: convert TGD to RINEX2 observation file '
    print(INFO)

    keyHOURS = sorted(HOURS.keys())
    mk_archives(dirRNX, keyHOURS)
    mk_archives(dirNAV, keyHOURS)

    for idx, _ in enumerate(keyHOURS):
        if idx % poolCOUNT == 0:
            pass
        else:
            continue
        pool = Pool(poolCOUNT)
        arguments = [
            (obsYEAR, obsDOY, obsHOUR, dirTGD, dirRNX, dirNAV, INFO)
            for obsHOUR in keyHOURS[idx : idx + poolCOUNT]
        ]
        results = [
            pool.apply_async(run_teqcConvert, args=argument) for argument in arguments
        ]
        # reslines = [res.get()for res in results]
        for res in results:
            reslines = res.get()
            [print(line) for line in reslines]

        pool.close()
        pool.join()

    return


# # test 13m 28.8s
# convert_TGDtoRNX2_teqc(
#     obsYEAR='23',
#     obsDOY='352',
#     HOURS={f'{key:02}': chr(key + ord('a')) for key in range(0, 24)},
#     dirTGD=Path('/run/media/echoo/koidata/CMONOC/TGD/2023/352'),
#     dirRNX=Path('/run/media/echoo/koidata/CMONOC/RNX/2023/352'),
#     dirNAV=Path('/run/media/echoo/koidata/CMONOC/NAV/2023/352'),
#     poolCOUNT=4,
# )


# %%
# -*- gfzrnx: unify RINEX files -*-
#  *  实际使用过程中, georinex对RINEX3的读取并不友好, 对由teqc转换后的RINEX2读取也不行
#  *  这里实际是将teqc的RINEX2文件进行纠正, kv选项保持原版本, 输出仍为RINEX2格式
#  *  但是能够被georinex.load()方法读取
#  *  命名风格为RINEX3是为了与teqc转换处的RINEX2文件(o文件)进行区分
#  *  数据存储方式实际为RINEX格式
def run_gfzrnx(
    dirRNX: Path,
    fRNX2: Path,
    fUpdate: Path,
    obsNAME: str,
    fTYPE: str,
    satellite: str,
    count: int,
) -> str:
    if satellite == 'BeiDou':
        satsysm = 'C'
        obsTYPES = 'L1,L7,C1,C7,S1'
    elif satellite == 'GPS':
        satsysm = 'G'
        obsTYPES = 'L1,L2,L5,C1,C2,C5,S1'
    elif satellite == 'mixed':
        satsysm = 'CGJ'
        obsTYPES = 'L1,L2,L5,L7,C1,C2,C5,C7,S1'
    elif satellite == 'geo':
        satsysm = 'CJ'
        obsTYPES = 'L1,L5,L7,C1,C5,C7,S1'
    else:
        print(
            'WARNING: just support BeiDou | GPS satsysms! \n'
            "One can modify source code referring to 'gfzrnx -h' \n"
            'for collecting data from other satsysms >_< \n'
        )

    try:
        sh.gfzrnx(
            '-finp',
            fRNX2,
            '-f',
            '-chk',
            # '-vo', '3',
            '-obs_types',
            obsTYPES,
            '-satsys',
            f'{satsysm}',
            '-crux',
            fUpdate,
            '-kv',
            '-fout',
            f'{dirRNX}/::RX3::CHN',
        )
    except sh.ErrorReturnCode as e:
        return (
            f'{count:4d} - {fTYPE} - {obsNAME} - {satellite} : '
            f'converting exited with {e.exit_code}. \n'
        )
    else:
        return (
            f'{count:4d} - {fTYPE} - {obsNAME} - {satellite} : '
            f'converted successfully. \n'
        )


def convert_RNX2toRNX3_gfzrnx(
    obsYEAR: str,
    obsHOURS: dict[str:str],
    dirRNX: Path,
    fUpdate: Path,
    fTYPE: str = 'obs',
    satellite: str = 'BeiDou',
    poolCOUNT: int = 4,
):
    """gfzrnx: concat RINEXfiles of diverse hours into a file of the day for each station

    Args:
        obsYEAR (str): the last two character of stryear  i.e. '23'('2023'), '24'('2024')
        obsHOURS (dict[str:str]): {'00':'a', '01':'b', '02':'c', ...} from T02 name rule
        dirRNX (Path): /source_directory/TGD_archive/year/doy
        fUpdate (Path): /source_directory/T02_archive/year/doy/f_update.txt
        fTYPE (str, optional): RINEX file type. Defaults to 'obs' | 'nav'.
        satellite (str, optional): Defaults to 'BeiDou' | 'GPS'.
        poolCOUNT (int, optional): Defaults to 4. Multiprocessing pool count, must be smaller than cpucore count,
                                   known as cpu_count()
    """
    INFO = f'\n  gfzrnx - {fTYPE} : convert RINEX2 to RINEX3 \n'
    print(INFO)

    fsuffix = 'o' if fTYPE == 'obs' else 'n'
    obsnameDict = {
        RNX2_file.stem[0:4]: RNX2_file.stem[0:4]
        for RNX2_file in dirRNX.glob(f'**/*.{obsYEAR}{fsuffix}')
    }
    obsNAMES = sorted(obsnameDict.values())

    for idx, _ in enumerate(obsNAMES):
        if idx % poolCOUNT == 0:
            pass
        else:
            continue

        pool = Pool(poolCOUNT)
        results = []
        for count, obsNAME in enumerate(obsNAMES[idx : idx + poolCOUNT], start=idx + 1):
            fRNX2 = [
                f'{dirRNX}/{obsHOUR}/{obsNAME}*.{obsYEAR}{fsuffix}'
                for obsHOUR in obsHOURS.keys()
            ]
            argument = (dirRNX, fRNX2, fUpdate, obsNAME, fTYPE, satellite, count)
            results.append(pool.apply_async(run_gfzrnx, args=argument))

        pool.close()
        pool.join()
        [print(':::', res.get()) for res in results]


# # test 60min
# convert_RNX2toRNX3_gfzrnx(
#     obsYEAR='23',
#     obsHOURS={f'{key:02}': chr(key + ord('a')) for key in range(0, 24)},
#     dirRNX=Path('/run/media/echoo/koidata/CMONOC/RNX/2023/352'),
#     fUpdate=Path('/run/media/echoo/koidata/CMONOC/raw1s/2023/352/RNX_update.txt'),
#     fTYPE='obs',
#     satellite='BeiDou',
#     poolCOUNT=4,
# )


# %%
# -*- main procession
# * T02 -> TGD
# * TGD -> RINEX2
# * RINEX2 -> RINEX3 (实为纠正后的RINEX2)
def convertT02_to_RINEX(
    poolCOUNT: int = 6,
    satsysm: str = 'BeiDou',
    extract_nav: bool = False,
    toTGD: bool = True,
    toRinex2: bool = True,
    toRinex3: bool = True,
    toRemoveTGD: bool = True,
    toRemoveRinex2: bool = True,
    toCrx: bool = True,
) -> None:
    """MAIN PROCESSION: Convert receiver T02 format file to RINEX format
    Args:
        poolCOUNT(int): one of 1|2|3|4|6|8|12|24 and smaller <= CPU_COUNT
        satsysm(str): GNSS satellite system, 'BeiDou' | 'GPS' | 'mixed' | 'geo'
        extract(bool): if extract Navigation information
        toTGD(bool):
        toRinex2(bool):
        toRinex3(bool):
        toRemoveTGD(bool):
        toRemoveRinex2(bool):
        toCrx(bool):
    """

    today = datetime.today().strftime('%Y_%m_%d')

    tomlConfig = './config.toml'
    with open(tomlConfig, mode='rb') as f_toml:
        selfConfig = toml.load(f_toml)

    stations = selfConfig['stations']
    year = selfConfig['year']
    doys = selfConfig['doys']
    if doys == 'auto':
        year = int(datetime.today().strftime('%Y'))
        doys = [int(datetime.today().strftime('%j')) - 1]
    else:
        pass
    stryear = f'{year}'
    print(
        f"today is {today}: {datetime.today().strftime('%j')}, \n"
        f"and the observing doy is {doys}"
    )
    archives = selfConfig['archives']
    archGnss = archives['dataGnss']
    archT02, archTGD = archives['dataT02'], archives['dataTGD']
    archRnx, archNav = archives['dataRnx'], archives['dataNav']
    # archRmv = archives['dataRmv']

    srcDir = Path(selfConfig['source'], archGnss)
    dirT02, dirTGD = srcDir / archT02 / stryear, srcDir / archTGD / stryear
    dirRnx, dirNav = srcDir / archRnx / stryear, srcDir / archNav / stryear

    # dirRmv = srcDir / archRmv / today

    CPU_COUNT = cpu_count()

    if poolCOUNT > CPU_COUNT:
        print('WARNING: cpuCore conuts are excesssful! ')

    dirs = {}
    obsHOURS = {f'{key:02}': chr(key + ord('a')) for key in range(0, 24)}

    # ::: ----------------------------------------------------------------------------------

    for doy in doys:
        strdoy = f'{doy:0>3.0f}'
        stryr = stryear[-2:]

        # ---------------------------------------------------------------------------------
        dirs['T02'], dirs['TGD'] = dirT02 / strdoy, dirTGD / strdoy
        dirs['RNX'], dirs['NAV'] = dirRnx / strdoy, dirNav / strdoy

        if dirs['T02'].is_dir():
            print(sh.du('-sh', dirs['T02']))
            dirs['TGD'].mkdir(mode=0o777, parents=True, exist_ok=True)
            dirs['RNX'].mkdir(mode=0o777, parents=True, exist_ok=True)
            dirs['NAV'].mkdir(mode=0o777, parents=True, exist_ok=True)
        else:
            print(f"ERROR: dirTO2 is invalid! \n, {dirs['T02']}")
            continue

        fconfig = new_txts(dirs['T02'], ['qCheckinfo', 'rnxUpdate'], 'txt')
        # -------------------------- costs  : 40 min --------------------------------------
        if toTGD:
            convert_TrimbleT02toTGD_runpkr00(
                obsHOURS, dirs['T02'], dirs['TGD'], stations, poolCOUNT
            )
            cal_xyzObs_teqc(dirs['TGD'], fconfig['qCheckinfo'], fconfig['rnxUpdate'])
        else:
            pass

        # -------------------------- costs  : 60 min --------------------------------------
        if toRinex2:
            convert_TGDtoRNX2_teqc(
                stryr,
                strdoy,
                obsHOURS,
                dirs['TGD'],
                dirs['RNX'],
                dirs['NAV'],
                poolCOUNT,
            )
        else:
            pass

        if toRemoveTGD:
            sh.rm('-rf', dirs['TGD'])

        # -------------------------- costs  : 140 min | 218 min ---------------------------
        if toRinex3:
            convert_RNX2toRNX3_gfzrnx(
                stryr,
                obsHOURS,
                dirs['RNX'],
                fconfig['rnxUpdate'],
                'obs',
                satsysm,
                poolCOUNT - 2,
            )
        else:
            pass

        # -------------------------- costs  : 7 min ---------------------------------------
        if extract_nav:
            convert_RNX2toRNX3_gfzrnx(
                stryr,
                strdoy,
                obsHOURS,
                dirs['NAV'],
                fconfig['rnxUpdate'],
                'nav',
                satsysm,
                poolCOUNT - 3,
            )
            rm_files_in(dirs['toRemove'], item='dir')
        else:
            pass

        if toRemoveRinex2:
            rm_files_in(dirs['RNX'], item='dir')

        if toCrx:
            rnx2crx_files_in(dirs['RNX'])

    return


# %%
if __name__ == '__main__':
    convertT02_to_RINEX(poolCOUNT=2, satsysm='BeiDou')
    # poolCOUNT = 2
    # runpkr00: 43min
    # teqc | gfzrnx: 203min

# %%
