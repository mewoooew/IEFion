# IEFion GNSS Data Converter

convert GNSS ( **BDS** | **GPS** ) data form receiver raw format (trimble **T02** ) to **RINEX**, and extract high-precision **TEC** data from RINEX file. 


## ✨ Features

- 🦊 **Cycle slip repair** by means of TECR[^LIU] and MW-combination[^MELBOURNE][^WÜBBENA]. 
- 🦝 **DCB correction** referring to CODE **GIM TEC** (optional: **IRI2020** model). 
- 🦥 the TEC data is obtained by **phase smoothed preseudo-range** method with dual-frequency measurements[^GUO]. 
- 🐍 With **python** multiprocession, fast batch I/O operation and matrix computation. 
- 🐧 Based on **Linux**, compiling the binaries with high performanc.

## 🥄 Before starting

1. **Linux platform** (following procedures conducted on Archlinux as an example )
   * preinstall required ***tools*** from remote official package library
     ```bash
     sudo pacman -Syu gzip
     sudo pacman -Syu lftp
     sudo pacman -Syu libxcrypt-compat
     ```
2. **Pre-install binaries**
   
   * [**runpkr00**](https://kb.unavco.org/article/trimble-runpkr00-latest-versions-744.html)[^Trimble]
     ```bash
     # -*- install -*-
     # for archlinux flavor, convert runpkr00 (rpm package) using alien tool
     yay -Syu alien_package_converter   # install alien
     cd path_to/ief_gnss/bin
     sudo alien -t runpkr00-5.40-1trmb.i586.rpm   # convert package
     sudo tar -zxvf *-5.40.tgz   # unzip the binary
     sudo cp -rf ./usr /

     # if RedHat flavors, can install the package directly
     rpm -ivh runpkr00-5.40-1trmb.i586.rpm
     
     # -*- check if successful -*-
     runpkr00 
     # if True: 
     # runpkr00 - Utility to unpack Trimble R00\T00\T01\T02 files, Version 5.40 (Linux) ( t01lib 8.63 )
     # Copyright (c) Trimble Navigation Limited 1992-2012.  All rights reserved.
     ```
   * [**teqc**](https://www.unavco.org/software/data-processing/teqc/teqc.html) [^UNAVCO]
     
     ```bash
     # -*- grant exectable rights -*-
     # Linux x86_64 (64-bit, dynamically-linked 
     # from RedHat kernel 2.4.21-4.ELsmp) 
     chmod +x teqc
     sudo mv ./teqc /usr/local/bin/
     # -*- check if successful -*-
     teqc +version
     # if True: 
     # executable:  teqc
     # version:     teqc  2019Feb25
     # build:       Linux 2.4.21-27.ELsmp|Opteron|gcc|Linux 64|=+
     ```
   * [**gfzrnx**](https://gnss.gfz-potsdam.de/services/gfzrnx/download) [^GFZ] 
     ```bash
     # -*- install -*-
     sudo mv path_to/ief_gnss/gfzrnx_2.1.9_lx64 /usr/local/bin/   # move to $PATH
     cd /usr/local/bin
     sudo chmod a+x gfzrnx_2.1.9_lx64   # Change Modification to grant exectable rights 
     sudo ln -s ./gfzrnx_2.1.9_lx64 gfzrnx   # create Soft LiNk gfzrnx
     
     # -*- check if successful -*-
     gfzrnx -h
     # if True: 
     # © Helmholtz-Centre Potsdam - GFZ German Research Centre for Geosciences
     # Section 1.1 Space Geodetic Techniques
     # see https://gnss.gfz-potsdam.de/services/gfzrnx
     # for license details and manual
     # Thomas Nischan, nisn@gfz-potsdam.de
     # -----------------------------------------------------------------

     # VERSION: gfzrnx-2.1.9
     ```
   
3. **Create python(^3.11) virtual environment** 
   * recommand creating and maintaining virtual env **(.venv)** using [**poetry**](https://python-poetry.org/docs/) tool referred to [Code and Me](https://blog.kyomind.tw/python-poetry/)
     ```bash
     # -*- install -*-
     # NOTICE: python^3.11 must be pre-installed
     # refer to 'With the official installer' method on poetry official website
     curl -sSL https://install.python-poetry.org | python3 - 
     # better to be added in .bashrc or .zshrc file
     export PATH=$PATH:$HOME/.local/bin   # explicit environment variable
     
     # -*- reboot shell -*-
     
     # -*- create virtual env -*-
     cd path_to/ief_gsss   # in which pyproject.toml exists 
     # create virtual env in the working directory
     poetry config virtualenvs.in-project true  
     poetry env use python3
     poetry lock   # parse dependencies in pyproject.toml
     poetry install   # install dependencies in pyproject.toml
     
     # -*- enter the virtual env -*-
     poetry shell  
     ```
   * dependencies | requirements:
     ```toml
     [tool.poetry.dependencies]
     python = "^3.11"
     pathlib = "^1.0.1"
     datetime = "^5.5"
     georinex = "^1.16.2"
     scipy = "^1.13.1"
     pandas = "^2.2.2"
     numpy = "^1.26.4"
     xarray = "^2024.5.0"
     matplotlib = "^3.9.0"
     pyproj = "^3.6.1"
     iricore = "^1.8.0"
     sh = "^2.0.6"
     tomli-w = "^1.0.0"
     astropy = "^6.1.0"
     requests = "^2.32.2"
     ```
4. **Prepare data directories**
   * Create a **directory tree** as below
     ```xml
     # OBSV:obsName, ISO:country, R:fromReceiver, 
     # YEAR:stryear, DOY:dayOfYear, D:Day, S:Second, 
     # C:BeiDou, Ofile
     
     dirTree # *: settings for outputs
     # except for archive dataT02 shohuld be set up manually by users, 
     # other archives can be created by algorithms automatically :)
      - sourceDir
        |-- dataGnss
        |   |-- dataRnx
        |   |   |-- stryear
        |   |   |   |-- doy
        |   |   |   |   |-- OBSV00ISO_R_YEARDOY0000_01D_01S_CO.RNX
        |   |   |   |   |-- AHBB00CHN_R_20233430000_01D_01S_CO.RNX 
        |   |-- dataT02 -- stryear -- doy   # must be valid
        |   |-- dataTGD -- stryear -- doy
        |   |-- dataNav -- stryear -- doy
        |   |-- dataTEC*
        |   |   |-- stryear
        |   |   |   |-- doy
        |   |   |   |   |-- dataPlots # .png outputs 
        |   |   |   |   |-- BeiDou_TEC_DOY_01s_YEAR.nc
        |   |   |   |   |-- BeiDou_TEC_343_01s_2023.nc (e.g.)
        |   |-- dataRmv -- today # data to remove later
        |-- dataSp3
        |   |-- stryear
        |   |   |-- doy
        |   |   |   |-- IGSIMGXFIN_YEARDOY0000_01D_05M_ORB.SP3
        |   |   |   |-- WUM0MGXFIN_20233430000_01D_05M_ORB.SP3 (e.g.)
        |-- dataGim
        |   |-- nc # processed to be ncFomat in advance
        |   |   |-- stryear
        |   |   |   |-- COD0YEARDOY.nc
        |   |   |   |-- COD02023343.nc (e.g.)
     ```

## 🪐 Basic Usage

1. Set attributes of observation date and data directory in path_to/ief_gnss/**config.toml** file
   * [REQUIRED] **`year`** must is a 4-digit integer and **`doy`** (ordinal day of year) is a number list wrapped with `[]` bracket.  
   * [REQUIRED] dataT02 archive exists and contains source T02 data, while other archives could be created by algorithms when running if not existed. 
   * [OPTIONAL] **`stations`** (observers) set defaults to `ALL`, and modified by editting **`stations`** list, which can be matched vaguely with 1-3 character or completely with 4 character of station name by algorithms. 
   * [OPTIONAL] ionosphere height set defaults to 350 km, and can be modified with attribute **`Hionos`**.
2. Run **convert2Rinex_mprocess.py** 
   * to get RINEX data format.
3. Run **getGnssData.py**
   * to get GIM TEC data and .sp3 Orbit data prepared for DCBs correction algorithms and satellite positioning algorithms, respectively in tec_inversion_BDS.py
4. Run **tec_inversion_BDS.py**
   * to get the BDS TEC with cycle slip repair and DCBs correction;  
5. Output data **TEC** will be injected into **`netCDF`** (.nc) file in the **`dataTEC`** archive ^~^.
   > three .py scripts can function independently with each other, provided that the input data for pointed script has already existed .

   > for matlab coders who are accustomed to .mat format data > <, try to run **convert_nc2mat.m**
   
## 🛸 Performance Test 

1. **Cycle slip** repair
   |                                   |                                   |
   | --------------------------------- | --------------------------------- |
   | ![](./plots/BeiDou_sTEC_AHBB.png) | ![](./plots/BeiDou_sTEC_BJFS.png) |
   | ![](./plots/BeiDou_sTEC_DLHA.png) | ![](./plots/BeiDou_sTEC_FJXP.png) |
   |                                   |                                   |
2. **DCBs** correction
   |                        |                        |                        |
   | ---------------------- | ---------------------- | ---------------------- |
   | ![](./plots/DCBs1.png) | ![](./plots/DCBs2.png) | ![](./plots/DCBs3.png) |
   | ![](./plots/DCBs4.png) | ![](./plots/DCBs5.png) | ![](./plots/DCBs6.png) |
   |                        |                        |                        |
3. **Time** costs
   * Test data: CMONOC `260` stations, `5` BDS GEO satellites, `1s` sample rate, `poolCOUNT` set to `2`: 
   * Test environment 1:
      * OS: Arch linux x86_64
      * CPU: 11th Gen Intel i7-11800H (16) @ 4.600GHz 
      * Memory: 16GB
      * **Cost**: ~**2.5 hour** to multiprocess the whole workflow. 
   * Test environment 2:
      * OS: Arch Linux on Windows 10 x86_64 (WSL2)
      * CPU: Intel i7-6700 (8) @ 3.407GHz 
      * Memory: 8GB
      * **Cost**: ~**5 hour** to multiprocess the whole workflow. 
<div align=center><img src="./plots/workflow_mermaids.png" style="zoom:50%"></div>

## 🐬 References

1. [^LIU]: LIU Z. A new automated cycle slip detection and repair method for a single dual-frequency GPS receiver[J]. Journal of Geodesy, 2011, 85(3): 171-183. doi:10.1007/s00190-010-0426-y.
2. [^MELBOURNE]:MELBOURNE W G. The case for ranging in GPS-based geodetic systems[C]. US Department of Commerce Rockville, Maryland, 1985: 373-386.
3. [^WÜBBENA]:WÜBBENA G. Software developments for geodetic positioning with GPS using TI-4100 code and carrier measurements[C]//Vol. 19. US Department of Commerce Rockville, Maryland, 1985: 403-412.
4. [^GUO]:GUO J, OU J, YUAN Y, et al. Optimal carrier-smoothed-code algorithm for dual-frequency GPS data[J]. Progress in Natural Science, 2008, 18(5): 591-594. doi:10.1016/j.pnsc.2007.12.010. 
5. [^Trimble]: runpkr00 - Utility to unpack Trimble R00\T00\T01\T02 files, Version 5.40 (Linux) ( t01lib 8.63 ). Copyright (c) Trimble Navigation Limited 1992-2012. All rights reserved.
6. [^UNAVCO]:TEQC: The Multi-Purpose Toolkit for GPS/GLONASS Data, L. H. Estey and C. M. Meertens, GPS Solutions (pub. by John Wiley & Sons), Vol. 3, No. 1, pp. 42-49, https://doi.org/10.1007/PL00012778, 1999.
7. [^GFZ]: Nischan, Thomas (2016): GFZRNX - RINEX GNSS Data Conversion and Manipulation Toolbox. GFZ Data Services. [http://dx.doi.org/10.5880/GFZ.1.1.2016.002](https://doi.org/10.5880/GFZ.1.1.2016.002).

