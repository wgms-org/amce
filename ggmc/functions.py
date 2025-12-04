"""functions"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy  # usage: scipy.stats.funct()
from pathlib import Path
import math
from .helpers import *
from .kriging import *
from typing import List,Tuple

# imported from step 3 (with duplicates from above removed)
import math
import time
import fnmatch

# 4_Kriging_regional_mass_balance
from .propagation import wrapper_latlon_double_sum_covar, sig_dh_spatialcorr, sig_rho_dv_spatialcorr, ba_anom_spatialcorr

"""
Workflow constants
"""

DENSITY_FACTOR = (0.85, 0.06)
# Density of ice (mean, sigma) relative to water (1000 kg m-3); see Dussaillant et al 2024 and Huss 2013


"""
Workflow functions
"""

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Section: 1_glacier_change_data Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def data_prep_spt_anom(fog_version_string: str,
                       input_data_path_string: str,
                       output_data_path_string: str) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """
    This script reads glacier-wide mass-balance data from the WGMS FoG database
    and provides functions for related analysis and plots.

    Parameters
    ----------
    fog_version_string: str
        A string input of the form 'YYYY-01' indicating the FOG version (e.g., '2025-01')
    input_data_path_string: str
        A relative path string to the glacier mass balance series ('fog_bw-bs-ba_2025-01.csv').
    output_data_path_string: str
        A string that indicates the relative path to the directory where output files should be delivered.

    Returns
    -------
    Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]
        Four pd.Dataframe objects (within a tuple) representing the mass balance for the following periods:
            'ANNUAL_BALANCE'
            'SUMMER_BALANCE'
            'WINTER_BALANCE'
            'ANNUAL_BALANCE_UNC'

    """

    # Retrieve the current working directory where the workflow is being run
    path = os.getcwd()

    # input the fog version
    fog_version = fog_version_string

    # test that the input file path is relative
    if Path(input_data_path_string).is_absolute() == True:
        raise ValueError('input data must be provided as a relative path from the working directory of your workflow script.')
    in_data_file = Path(input_data_path_string)
    out_dir = Path(path,output_data_path_string,f'fog-{fog_version}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # read glacier data from csv files into dataframe
    input_df = pd.read_csv(in_data_file, delimiter=',', header=0)
    print('Input file ({}), data fields: \n {}'.format(in_data_file, list(input_df.columns)))

    # number crunching: create unique list of glacier ids and years with data
    wgms_id_lst = input_df['WGMS_ID'].unique().tolist()
    yr_lst = list(range(min(input_df['YEAR']), max(input_df['YEAR']) + 1, 1))
    reg_lst = input_df['GLACIER_REGION_CODE'].unique().tolist()
    reg_lst.remove('SAN')
    reg_lst= reg_lst + ['SA1','SA2'] # Separate Andes in two regions:

    # number crunching: create & export data frame with mass-balance data from input file
    ba_file = Path(out_dir,f'fog_{fog_version}_ba.csv')
    bs_file = Path(out_dir,f'fog_{fog_version}_bs.csv')
    bw_file = Path(out_dir,f'fog_{fog_version}_bw.csv')
    ba_unc_file = Path(out_dir,f'fog_{fog_version}_ba_unc.csv')

    # create mass-balance data csv's if they have not been produced before
    ba_df = create_mb_dataframe(input_df, wgms_id_lst, yr_lst, 'ANNUAL_BALANCE')
    ba_df.to_csv(ba_file, sep=',', encoding='utf-8', index=True, index_label='YEAR')
    bs_df = create_mb_dataframe(input_df, wgms_id_lst, yr_lst, 'SUMMER_BALANCE')
    bs_df.to_csv(bs_file, sep=',', encoding='utf-8', index=True, index_label='YEAR')
    bw_df = create_mb_dataframe(input_df, wgms_id_lst, yr_lst, 'WINTER_BALANCE')
    bw_df.to_csv(bw_file, sep=',', encoding='utf-8', index=True, index_label='YEAR')
    ba_unc_df = create_mb_dataframe(input_df, wgms_id_lst, yr_lst, 'ANNUAL_BALANCE_UNC')
    ba_unc_df.to_csv(ba_unc_file, sep=',', encoding='utf-8', index=True, index_label='YEAR')

    return ba_df, bs_df, bw_df, ba_unc_df


def data_prep_elevation_change(fog_version_string: str,
                               elevation_change_input_string: str,
                               glacier_series_input_string: str,
                               provider_list_to_drop: List[str],
                               output_data_path_string: str,
                               f_dens_input: float = DENSITY_FACTOR[0],
                               sig_dens_input: float = DENSITY_FACTOR[1]) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    This script reads glacier-wide elevation change data from the WGMS FoG database
    and transform it into specific glacier mass balance change
    Mass balance uncertainties include dh/dt uncertainty + density conversion uncertainty

    Parameters
    ----------
    fog_version_string: str
    A string input of the form 'YYYY-01' indicating the FOG version (e.g., '2025-01')
    elevation_change_input_string: str
    A relative path string to the elevation change data ('FOG_ELEVATION_CHANGE_DATA_2025-01.csv').
    glacier_series_input_string: str
    A relative path string to the glacier series data ('FOG_GLACIER_SERIES_2025-01.csv').
    provider_list_to_drop: List[str]
    A list of string names of authors within the dataset whose records should be removed.
    output_data_path_string: str
    A string that indicates the relative path to the directory where output files should be delivered.
    f_dens_input: float = DENSITY_FACTOR[0]
    Float values for the density conversion factor; see workflow constants section
    sig_dens_input: float = DENSITY_FACTOR[1]
    Float values for the density conversion factor sigma; see workflow constants section

    Returns
    -------
    Tuple[pd.DataFrame,pd.DataFrame]
        Two pd.Dataframe objects (within a tuple); the first is a dataset of WGMS coordinates and
        the second is the outputted mass balance dataset


    """

    # Define input
    path = os.getcwd()

    # input the fog version
    fog_version = fog_version_string

    # format the output directory as necessary
    out_dir = Path(path,output_data_path_string,f'fog-{fog_version}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # read the input files from the provided relative paths
    if Path(elevation_change_input_string).is_absolute() == True:
        raise ValueError('elevation_change_input_string must be provided as a relative path from the working directory of your workflow script.')
    if Path(glacier_series_input_string).is_absolute() == True:
        raise ValueError('glacier_series_input_string must be provided as a relative path from the working directory of your workflow script.')
    c3s_geo_data = Path(elevation_change_input_string)
    c3s_geo_series = Path(glacier_series_input_string)
    c3s_gla_series = Path(glacier_series_input_string)

    # Write the coordinate data
    id_coord_df= pd.read_csv(c3s_geo_series, delimiter=',', header=0, usecols=['WGMS_ID','LATITUDE','LONGITUDE'] ,index_col='WGMS_ID').sort_index()
    id_coord_df_outpath = Path(out_dir,f'FOG_coord_{fog_version}.csv')
    id_coord_df.to_csv(id_coord_df_outpath)

    #################################
    #################################
    #### 1. EDIT GEODETIC DATA
    #################################
    #################################

    # # read global geodetic mass-balance data from csv into dataframe
    input_geo_df= pd.read_csv(c3s_geo_data, delimiter=',', header=0, index_col='WGMS_ID', low_memory=False).sort_index()
    input_geo_df=input_geo_df.dropna(subset = ['ELEVATION_CHANGE'])
    input_geo_df=input_geo_df.dropna(subset = ['SURVEY_DATE'])

    # Select CAU Ids
    cau_df = input_geo_df.loc[(input_geo_df['GLACIER_REGION_CODE'] == 'CAU')]
    cau_df = cau_df.rename(columns = {'GLIMS_ID': 'ID'}).drop(['RGI60_ID'], axis=1)

    # Remove glaciers with no RGI60 ID and glaciers from CAU
    geo_df = input_geo_df.dropna(subset=['RGI60_ID']).drop(['GLIMS_ID'], axis=1).rename(columns = {'RGI60_ID': 'ID'})
    geo_df = geo_df[(geo_df['GLACIER_REGION_CODE'] != 'CAU')]

    # Concatenate both to get all IDs
    input_geo_df = pd.concat([geo_df, cau_df])

    ## Get reference date (ini)
    input_geo_df['str_ref_date']= input_geo_df['REFERENCE_DATE'].astype(str)
    input_geo_df['ref_year']=input_geo_df['str_ref_date'].str.slice(0, 4)
    input_geo_df['ref_year']=pd.to_numeric(input_geo_df['ref_year'], downcast="float")
    input_geo_df['ref_month']=input_geo_df['str_ref_date'].str.slice(4, 6)
    input_geo_df['ref_month']=pd.to_numeric(input_geo_df['ref_month'], downcast="float")

    ## Get survey date (fin)
    input_geo_df['str_sur_date']= input_geo_df['SURVEY_DATE'].astype(str)
    input_geo_df['sur_year']=input_geo_df['str_sur_date'].str.slice(0, 4)
    input_geo_df['sur_year']=pd.to_numeric(input_geo_df['sur_year'], downcast="float")
    input_geo_df['sur_month']=input_geo_df['str_sur_date'].str.slice(4, 6)
    input_geo_df['sur_month']=pd.to_numeric(input_geo_df['sur_month'], downcast="float")

    ## Change date into decimal format
    input_geo_df['ini_date']=input_geo_df.apply(lambda x: date_format(x['ref_month'], x['ref_year']), axis=1)
    input_geo_df['fin_date']=input_geo_df.apply(lambda x: date_format(x['sur_month'], x['sur_year']), axis=1)

    ## Transform cumulative elevation changes to rate
    input_geo_df['elevation_chg_rate']=input_geo_df.apply(lambda x: cum_to_rate(x['ELEVATION_CHANGE'], x['fin_date'], x['ini_date']), axis=1)
    input_geo_df['sigma_elevation_chg']=input_geo_df.apply(lambda x: cum_to_rate(x['ELEVATION_CHANGE_UNC'], x['fin_date'], x['ini_date']), axis=1)

    ## Transform elevation change to specific mass balance: Apply density conversion factor
    f_dens = f_dens_input
    sig_dens = sig_dens_input

    input_geo_df['mb_chg_rate']= input_geo_df['elevation_chg_rate'] * f_dens
    input_geo_df['sigma_obs_mb_chg']= input_geo_df['sigma_elevation_chg'] * f_dens

    ## Calculate combined mass balance uncertainty: sigma dh + density transformation

    mb_rate = input_geo_df['mb_chg_rate']
    sigma_mb = input_geo_df['sigma_obs_mb_chg']
    sigma_mean = input_geo_df['sigma_obs_mb_chg'].mean()
    sigma_mb = sigma_mb.fillna(sigma_mean)
    input_geo_df['sigma_tot_mb_chg'] = abs(mb_rate) * np.sqrt((sigma_mb / mb_rate) ** 2 + (sig_dens / f_dens) ** 2)

    geo_df =input_geo_df.drop(['str_ref_date', 'ref_year', 'ref_month', 'str_sur_date', 'sur_year', 'sur_month',
                            'ELEVATION_CHANGE', 'ELEVATION_CHANGE_UNC', 'sigma_obs_mb_chg', 'SURVEY_ID', 'INVESTIGATOR',
                            'AREA_CHANGE', 'SURVEY_DATE', 'REFERENCE_DATE'], axis=1)

    geo_df.reset_index(inplace=True)

    # Remove unwanted variables and observations
    geo_edited_df =input_geo_df.drop(['str_ref_date', 'ref_year', 'ref_month', 'str_sur_date', 'sur_year', 'sur_month',
                            'ELEVATION_CHANGE', 'ELEVATION_CHANGE_UNC', 'sigma_obs_mb_chg', 'SURVEY_ID', 'AREA_CHANGE',
                            'SURVEY_DATE', 'REFERENCE_DATE'], axis=1)
    for p in provider_list_to_drop:
        geo_edited_df = geo_edited_df[geo_edited_df['INVESTIGATOR'] != p]

    geo_edited_df.reset_index(inplace=True)
    geo_edited_df_outpath = Path(out_dir,f'_FOG_GEO_MASS_BALANCE_DATA_{fog_version}.csv')
    geo_edited_df.to_csv(geo_edited_df_outpath, index=False)
    return id_coord_df, geo_edited_df

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Section: 2_Kriging_spatial_anomalies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


def calc_global_gla_spatial_anom(year_ini: int,
                                 year_fin: int,
                                 max_glac_anom: int,
                                 min_glac_anom: int,
                                 d_thresh_lst: List[int],
                                 max_d: int,
                                 fog_version: str,
                                 in_data_gla_path: str,
                                 ba_file_path: str,
                                 ba_unc_file_path: str,
                                 missing_years_path: str,
                                 in_gla_coord_path: str,
                                 fog_gmb_path: str,
                                 output_data_path_string: str) -> None:
    """
    Calculate the observational consensus estimate for every individual glacier

    calc_OCE_and_error_global_gla_reg_anom.py

    Author: idussa
    Date: Feb 2021
    Last changes: Feb 2021

    Scripted for Python 3.7

    Description:
    This script reads glacier-wide mass balance data edited from WGMS FoG database
    and regional glacier anomalies produced by calc_regional_anomalies_and_error.py
    and provides the observational consensus estimate for every individual glacier
    with available geodetic observations WGMS Id

    Input:  GEO_MASS_BALANCE_DATA_20200824.csv
            Regional_anomalies_ref_period_2009-2018.csv
            (UTF-8 encoding)

    Return: tbd.svg

    Parameters
    ----------
    fog_version_string: str
    A string input of the form 'YYYY-01' indicating the FOG version (e.g., '2025-01')

    Returns
    -------
    Tuple[pd.DataFrame,pd.DataFrame]
        Two pd.Dataframe objects (within a tuple); the first is a dataset of WGMS coordinates and
        the second is the outputted mass balance dataset

    """
    import time as time

    # Calculate the reference period
    reference_period = range(year_ini, year_fin + 1)

    # Define input
    start_time = time.time()

    path = os.getcwd()

    out_dir = Path(path,output_data_path_string,f'fog-{fog_version}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # create directory for regional glaciers anomalies
    out_reg_dir = Path(out_dir,'MEAN_spatial_gla_anom_ref_'+str(year_ini)+'-'+str(year_fin))
    if not os.path.exists(out_reg_dir):
        os.makedirs(out_reg_dir)

    out_anom_dir = Path(out_dir, 'LOOKUP_spatial_and_reg_ids_ref_'+str(year_ini)+'-'+str(year_fin))
    if not os.path.exists(out_anom_dir):
        os.makedirs(out_anom_dir)

    out_long_dir = Path(out_dir, 'LONG-NORM_spatial_gla_anom_ref_' + str(year_ini) + '-' + str(year_fin))
    if not os.path.exists(out_long_dir):
        os.makedirs(out_long_dir)

    ##### 2.1 READ MASS BALANCE DATA ######

    # read FoG file with global annual and seasonal mass-balance data
    # in_data_gla = os.path.join(path, 'in_data', 'fog-'+fog_version,'fog_bw-bs-ba_'+fog_version+'.csv')
    in_data_gla = in_data_gla_path
    if Path(in_data_gla).is_absolute() == True:
        raise ValueError('in_data_gla_path must be provided as a relative path from the working directory of your workflow script.')
    input_gla_df = pd.read_csv(in_data_gla, delimiter=',', header=0)

    ### create mass-balance data csv if it has not been produced before

    # create unique list of glacier ids and years with data
    all_fog_gla_id_lst = input_gla_df['WGMS_ID'].unique().tolist()
    yr_lst = list(range(1915, max(input_gla_df['YEAR']+1), 1))

    # !! These two lists were originally commented / switched; I restored the longer of the 2
    reg_lst= ['ASW', 'ANT', 'CEU', 'ASN', 'TRP', 'ASC', 'ASE', 'ACS', 'WNA', 'ACN', 'ALA', 'CAU', 'GRL', 'ISL', 'NZL', 'SCA', 'RUA', 'SJM', 'SA1', 'SA2']
    # reg_lst= ['GRL']

    # read in ba and ba_unc file data
    ba_file = Path(ba_file_path)
    if Path(ba_file_path).is_absolute() == True:
        raise ValueError('ba_file_path must be provided as a relative path from the working directory of your workflow script.')
    ba_unc_file = Path(ba_unc_file_path)
    if Path(in_data_gla).is_absolute() == True:
        raise ValueError('in_data_gla_path must be provided as a relative path from the working directory of your workflow script.')

    # read FoG file with global annual mass-balance data
    ba_df = pd.read_csv(ba_file, delimiter=',', header=0, index_col=0)
    ba_df.columns = ba_df.columns.map(int)  # make columns names great again

    ### Add missing years to Urumqi glacier fog_id 853
    missing_years_file = Path(missing_years_path)
    if Path(missing_years_path).is_absolute() == True:
        raise ValueError('missing_years_path must be provided as a relative path from the working directory of your workflow script.')
    missing_years_df = pd.read_csv(missing_years_file, delimiter=',', header=0, index_col=0)
    missing_years_df.columns = missing_years_df.columns.map(int)  # make columns names great again
    ba_df = ba_df.fillna(missing_years_df)

    ba_unc_df = pd.read_csv(ba_unc_file, delimiter=',', header=0, index_col=0)
    ba_unc_df.columns = ba_unc_df.columns.map(int)  # make columns names great again

    # in_gla_coord = os.path.join(path, 'in_data','fog-'+fog_version, 'FOG_coord_'+fog_version+'.csv')
    in_gla_coord = Path(in_gla_coord_path)
    if Path(in_gla_coord_path).is_absolute() == True:
        raise ValueError('in_gla_coord_path must be provided as a relative path from the working directory of your workflow script.')
    coord_gla_df= pd.read_csv(in_gla_coord, encoding='latin1', delimiter=',', header=0, index_col='WGMS_ID').sort_index()

    ##### 2.2 READ GEODETIC DATA ######

    # read FoG file with global geodetic data
    in_data_geo = Path(fog_gmb_path)
    if Path(fog_gmb_path).is_absolute() == True:
        raise ValueError('fog_gmb_path must be provided as a relative path from the working directory of your workflow script.')
    input_geo_df= pd.read_csv(in_data_geo, encoding='latin1', delimiter=',', header=0, index_col='WGMS_ID').sort_index()
    input_geo_df.reset_index(inplace=True)
    # print(geo_df)

    all_fog_geo_id_lst = input_geo_df['WGMS_ID'].unique().tolist()
    # print('Nb glaciers with geodetic obs C3S 2022: '+str(len(all_fog_geo_id_lst)))

    read_time = time.time()
    print("--- %s seconds ---" % (read_time - start_time))
    ############################################################################################################################

    for region in reg_lst:
        # region= 'GRL'
        print('Working on region, ', region)

        ## create empty dataframes for spatial anomalies and uncertainties
        spt_anom_df = pd.DataFrame(index=yr_lst)
        spt_anom_df.index.name = 'YEAR'
        spt_anom_lst = []

        spt_anom_err_df = pd.DataFrame(index=yr_lst)
        spt_anom_err_df.index.name = 'YEAR'
        sig_spt_anom_lst = []

        ## number crunching: SELECT GEODETIC DATA FOR GLACIER REGION GROUP

        if region == 'SA1':
            reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-01')]
        elif region == 'SA2':
            reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-02')]
        else:
            reg_geo_df = input_geo_df.loc[(input_geo_df['GLACIER_REGION_CODE'] == str(region))]

        ## create a list of fog_ids with geodetic data for the region group
        reg_fog_geo_id_lst = reg_geo_df['WGMS_ID'].unique().tolist()

        ############################################################################################################################
        ###### 3. CALCULATING SPATIAL ANOMALIES ######

        ## SELECT MASS BALANCE DATA FOR GLACIER REGION GROUP
        ## create list of glacier mass balance series ids possible to calculate the glacier temporal variabiity or anomaly
        ## remove or add neighbouring glacier mass balance series

        if region == 'ASN': # add Urumqui, remove Hamagury yuki, add
            add_id_lst = [853, 817]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
            rem_id = 897  # Hamagury yuki (ASN)
            rem_id_lst2 = [897, 1511, 1512]  # Hamagury yuki (ASN), Urumqi East and west branches (ASC)
            glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ALA')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASC')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst2)].index)
            glac_reg = glac_reg.drop(glac_reg[glac_reg['WGMS_ID'] == rem_id].index)
            # print(list(glac['WGMS_ID'].unique().tolist()))

        if region == 'ASE':
            add_id_lst = [817, 853]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
            rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
            glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ASC')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASW')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'ASC':
            rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
            glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ASE')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASW')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = glac_reg.drop(glac_reg[glac_reg['WGMS_ID'].isin(rem_id_lst)].index)

        if region == 'ASW':
            add_id_lst = [817, 853]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
            rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
            glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ASC')| (input_gla_df['GLACIER_REGION_CODE'] == 'ASE')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'CEU':
            glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'SA1':
            rem_id_lst = [3902, 3903, 3904, 3905, 1344, 3972]  # keep Martial Este only
            glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == 'SAN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = glac

        if region == 'SA2':  # keep Echaurren Norte only
            rem_id_lst = [3902, 3903, 3904, 3905, 2000, 3972]
            glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == 'SAN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = glac

        if region == 'NZL':
            add_id_lst = [2000]  # Martial Este (SAN-01)
            glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'ANT':
            rem_id_lst = [878, 3973]  # Dry valley glaciers
            glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = glac

        if region == 'RUA':
            glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'SJM') , ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'SJM':
            glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'ALA':
            glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'WNA') , ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'WNA':
            glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ALA'), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'TRP':
            rem_id = 226  # Yanamarey
            glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'] == rem_id].index)
            glac_reg = glac

        if region == 'ACS':
            glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) | (input_gla_df['GLACIER_REGION_CODE'] == 'ACN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'ACN':
            glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'GRL':
            glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'ACN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'ISL':
            glac = input_gla_df.loc[((input_gla_df['GLACIER_REGION_CODE'] == str(region)) |(input_gla_df['GLACIER_REGION_CODE'] == 'GRL')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'SCA':
            glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'CAU':
            glac = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = input_gla_df.loc[(input_gla_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        ## Find all possible individual glacier anomalies (with respect to reference period) for the given glacier id

        ## number crunching:   select mass-balance data for glacier region groups
        ba_glac_df = ba_df.loc[:, list(glac['WGMS_ID'].unique().tolist())]
        glac_anom = calc_anomalies(ba_glac_df, reference_period, region)
        unc_glac_anom = calc_spt_anomalies_unc(glac_anom, ba_unc_df, glac_anom.columns.to_list())

        # FOR SA2 ONLY: if no uncertainty measurement use the regional annual mean uncertainty of the glaciological sample
        if unc_glac_anom.isnull().sum().sum():
            for id in unc_glac_anom.columns.tolist():
                year_min = glac_anom[id].first_valid_index()
                yrs = list(range(1915, year_min))
                unc_glac_anom[id].fillna(np.nanmean(ba_unc_df), inplace=True)
                unc_glac_anom[id].mask(unc_glac_anom.index.isin(yrs), np.nan, inplace=True)
        else:
            continue

        ## Correct suspicious anomaly from Echaurren Norte by normalizing past period to present period amplitude.
        if region == 'SA2':
            STD_ech_ok = glac_anom.loc[glac_anom.index.isin(list(range(2004, (2022 + 1))))].std()
            STD_ech_bad = glac_anom.loc[glac_anom.index.isin(list(range(1980, (1999 + 1))))].std()
            glac_anom_pres_ok = glac_anom.loc[glac_anom.index >= 2004]
            norm_past = glac_anom.loc[glac_anom.index.isin(list(range(1885, (2003 + 1))))] / STD_ech_bad
            glac_anom_past_new = (norm_past * STD_ech_ok).round(decimals=1)
            glac_anom = pd.concat([glac_anom_past_new, glac_anom_pres_ok], axis = 0)

        # ## Filter series for regional anomaly to use
        ba_reg_glac_df = ba_df.loc[:, list(glac_reg['WGMS_ID'].unique().tolist())]
        reg_glac_anom = calc_anomalies(ba_reg_glac_df, reference_period, region)

        # ## Correct suspicious anomaly from Echaurren Norte by normalizing past period to present period amplitude.
        if region == 'SA2':
            STD_ech_ok = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(2004, (2022 + 1))))].std()
            STD_ech_bad = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(1980, (1999 + 1))))].std()
            reg_glac_anom_pres_ok = reg_glac_anom.loc[reg_glac_anom.index >= 2004]
            norm_past = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(1885, (2003 + 1))))] / STD_ech_bad
            reg_glac_anom_past_new = (norm_past * STD_ech_ok).round(decimals=1)
            reg_glac_anom = pd.concat([glac_anom_past_new, glac_anom_pres_ok], axis = 0)

        # ## select close anomalies for calculating the fog_id glacier anomaly

        spatial_id_fin_lst = glac_anom.columns.to_list()

        close_gla_weights = coord_gla_df.loc[spatial_id_fin_lst, :]
        lat_glac = close_gla_weights['LATITUDE']
        lon_glac= close_gla_weights['LONGITUDE']

        # ROMAIN: Replacing by inverse-distance weighting by kriging here
        anoms_4_fog_id_df = glac_anom[spatial_id_fin_lst]
        unc_anoms_4_fog_id_df = unc_glac_anom[spatial_id_fin_lst]

        # Get variance of anomalies in this region for the kriging algorithm
        var_anom = np.nanvar(anoms_4_fog_id_df)

        # We can't apply to the whole YEAR/ID dataframe at once here, we need to loop for each YEAR of the dataframes
        # to compute the kriging

        arr_mean_anom = np.ones((len(anoms_4_fog_id_df.index), len(reg_fog_geo_id_lst)), dtype=np.float32)
        arr_sig_anom = np.ones((len(anoms_4_fog_id_df.index), len(reg_fog_geo_id_lst)), dtype=np.float32)
        for i in range(len(anoms_4_fog_id_df.index)):
            print(f"Kriging region {region} for year {anoms_4_fog_id_df.index[i]}")

            # Create dataframe with anomalies, lat and lon
            yearly_anom_df = anoms_4_fog_id_df.iloc[i, :]

            obs_df = pd.DataFrame(data={"ba_anom": yearly_anom_df.values, "lat": np.array(lat_glac), "lon": np.array(lon_glac)})

            # print(obs_df)
            valids = np.isfinite(obs_df["ba_anom"])

            # If no data is valid, write NaNs
            if np.count_nonzero(valids) < 1:
                arr_mean_anom[i, :] = np.nan
                arr_sig_anom[i, :] = np.nan
                continue
            # Otherwise limit to valid data only
            else:
                obs_df = obs_df[valids]

            # Get latitude and longitude of unobserved glacier to predict
            lat_id = coord_gla_df.loc[reg_fog_geo_id_lst, 'LATITUDE']
            lon_id = coord_gla_df.loc[reg_fog_geo_id_lst, 'LONGITUDE']

            # Create dataframe with points where to predict (could be several at once but here always one)
            pred_df = pd.DataFrame(data={"lat": lat_id, "lon": lon_id})
            pred_df_path = Path(out_dir,'pred_',fog_version,'.csv')
            pred_df.to_csv()

            # Kriging at the coordinate of the current glacier
            mean_anom, sig_anom = wrapper_latlon_krige_ba_anom(df_obs=obs_df, df_pred=pred_df, var_anom=var_anom)
            arr_mean_anom[i, :] = mean_anom
            arr_sig_anom[i, :] = sig_anom

        # And write back the 1D list of uncertainties into an indexed (by YEAR) dataframe
        anom_fog_id_df = pd.DataFrame(index=anoms_4_fog_id_df.index, data=arr_mean_anom, columns=[str(fog_id) for fog_id in reg_fog_geo_id_lst])
        sig_anom_df = pd.DataFrame(index=anoms_4_fog_id_df.index, data=arr_sig_anom, columns=[str(fog_id) for fog_id in reg_fog_geo_id_lst])

        ## CALCULATE:  mean anomaly for fog_id
        ## if glacier has in situ measurements i.e. dist = 0 use the own glaciers anomaly
        anom_fog_id_df = anom_fog_id_df.loc[anom_fog_id_df.index >= 1915]
        spt_anom_lst.append(anom_fog_id_df)

        ## CALCULATE: Uncertainty for fog_id
        sig_anom_df = round(sig_anom_df, 2)
        sig_anom_df = sig_anom_df.loc[sig_anom_df.index >= 1915]
        sig_spt_anom_lst.append(sig_anom_df)

        # write the data for that entry
        glac_anom.to_csv(Path(out_anom_dir, region + '_all_SEL_gla_anomalies.csv'))
        reg_glac_anom.to_csv(Path(out_anom_dir, region + '_all_reg_gla_anomalies.csv'))
        unc_glac_anom.to_csv(Path(out_anom_dir, region + '_all_SEL_gla_anomalies_UNC.csv'))

        ### Save all glacier anomalies and uncertainties - exclude uncertainties from the SAN regions
        spt_anom_df = pd.concat(spt_anom_lst, axis='columns')
        spt_anom_path = Path(out_reg_dir, str(region) + '_spt_anoms_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
        spt_anom_df.to_csv(spt_anom_path)

        sig_spt_anom_df = pd.concat(sig_spt_anom_lst, axis='columns')
        sig_spt_path = Path(out_reg_dir, str(region) + '_spt_ERRORs_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
        sig_spt_anom_df.to_csv(sig_spt_path)

        print("--- %s seconds ---" % (time.time() - read_time))

        ### Save glacier anomalies and uncertainties OK with long time periods
        reg_ok_lst = ['ACS', 'ACN', 'ASW', 'ASE', 'ASC', 'ASN', 'ALA', 'SCA', 'SA2']
        if region in reg_ok_lst:
            spt_anom_df.to_csv(Path(out_long_dir, str(region) + '_spt_anoms_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv'))
            sig_spt_anom_df.to_csv(Path(out_long_dir, str(region) + '_spt_ERRORs_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv'))

    # reg_norm_lst = ['CAU', 'GRL', 'ISL', 'CEU', 'WNA']
    reg_norm_lst = ['ANT', 'RUA', 'SJM', 'SA1', 'ISL', 'NZL', 'TRP', 'CEU', 'WNA', 'CAU', 'GRL']

    ### 4. ADD NORMALIZED SERIES FROM NEIGHBOURING GLACIERS TO EXTEND ANOMALIES BACK IN TIME

    for region in reg_norm_lst:
        # region = 'SA1'
        spt_anom_fill_lst = []
        spt_anom_sig_fill_lst = []
        print('working on region, ', region)

        # !! The 'out_long_dir' had to be changed to 'out_reg_dir' for the code to run; this must have been a mistake from the
        # !! original code that was missed (likely via many subsequent runs and reruns of the code without refreshing input/output
        # !! data).
        spt_anom_in = Path(out_reg_dir, str(region) + '_spt_anoms_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
        spt_anom_df = pd.read_csv(spt_anom_in, delimiter=',', header=0, index_col=0)

        # !! And here, as well. (I.e., for both 'spt_anom_in' and 'sig_spt_anom_in'
        sig_spt_anom_in = Path(out_reg_dir, str(region) + '_spt_ERRORs_ref_' + str(year_ini) + '-' + str(year_fin) + '_' + fog_version + '.csv')
        sig_spt_anom_df = pd.read_csv(sig_spt_anom_in, delimiter=',', header=0, index_col=0)

        fog_id_lst = spt_anom_df.columns.to_list()

        for fog_id in fog_id_lst:
            print('working on id, ', fog_id)
            # fog_id='23697'
            max_sig = sig_spt_anom_df[fog_id].max().max()

            STD_id = spt_anom_df[fog_id].loc[spt_anom_df[fog_id].index.isin(list(reference_period))].std()
            print('std: ', STD_id)

            if region == 'ISL': ## Get series from Storbreen, Aalfotbreen and Rembesdalskaaka to normalize (SCA, fog_ids 302, 317, 2296)
                neighbour_anom_in = Path(out_anom_dir,'SCA_all_SEL_gla_anomalies.csv')
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','302','317','2296'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = Path(out_reg_dir,'SCA_spt_ERRORs_ref_2011-2020_'+fog_version+'.csv')
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR','302','317','2296'], index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour
                print('std: ', STD_neigbour)

            if region in ['SJM', 'RUA']: ## Get series from Storglacieren to normalize (SCA, fog_ids 332)
                neighbour_anom_in = Path(out_anom_dir,'SCA_all_reg_gla_anomalies.csv')
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','332'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = Path(out_reg_dir,'SCA_spt_ERRORs_ref_2011-2020_'+fog_version+'.csv')
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR', '332'], index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour
                print('std: ', STD_neigbour)

            if region == 'CEU':  ## Get series from Claridenfirn (CEU, fog_ids 2660)
                neighbour_anom_in = Path(out_anom_dir,'CEU_all_SEL_gla_anomalies.csv')
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols=['YEAR', '2660'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = Path(out_reg_dir,'CEU_spt_ERRORs_ref_2011-2020_'+fog_version+'.csv')
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols=['YEAR', '4617', '4619', '4620'], index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour
                print('std: ', STD_neigbour)

            if region == 'WNA':  ## Get series from Taku glacier (ALA, fog_ids 124)
                neighbour_anom_in = Path(out_anom_dir,'WNA_all_SEL_gla_anomalies.csv')
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols=['YEAR', '124'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = Path(out_reg_dir,'ALA_spt_ERRORs_ref_2011-2020_'+fog_version+'.csv')
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols=['YEAR', '124'], index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour
                print('std: ', STD_neigbour)

            if region == 'CAU':  ## Get series from Hinteeisferner, Kesselwand (CEU, fog_ids 491,507)
                neighbour_anom_in = Path(out_anom_dir,'CEU_all_SEL_gla_anomalies.csv')
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols=['YEAR', '491', '507'], index_col=['YEAR'])
                # print(neighbour_anom_df)
                neighbour_sig_mean_anom_in = Path(out_reg_dir,'CEU_spt_ERRORs_ref_2011-2020_'+fog_version+'.csv')
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0,usecols=['YEAR', '491', '507'], index_col=['YEAR'])
                # print(neighbour_sig_mean_anom_df)
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour
                print('std: ', STD_neigbour)

            if region == 'GRL':  ## Get series from Meighen and Devon Ice Caps to normalize (ACN, fog_ids 16, 39)
                neighbour_anom_in = Path(out_anom_dir,'GRL_all_SEL_gla_anomalies.csv')
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols=['YEAR', '16', '39'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = Path(out_reg_dir,'ACN_spt_ERRORs_ref_2011-2020_'+fog_version+'.csv')
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols=['YEAR', '102349', '104095'], index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour
                print('std: ', STD_neigbour)

            if region in ['ANT', 'NZL', 'SA1', 'TRP']: ## Get series from Echaurren to normalize (SA2, fog_id 1344)
                neighbour_anom_in = Path(out_anom_dir,'SA2_all_reg_gla_anomalies.csv')
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','1344'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = Path(out_reg_dir,'SA2_spt_ERRORs_ref_2011-2020_'+fog_version+'.csv')
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / (STD_neigbour)
                print('std: ', STD_neigbour)

            norm_all_neighbour_fog_id = norm_neighbour * STD_id

            norm_neighbour_fog_id = norm_all_neighbour_fog_id.mean(axis=1)
            norm_neighbour_fog_id = pd.DataFrame(norm_neighbour_fog_id, columns=[str(fog_id)])
            fog_id_spt_anom = spt_anom_df.filter([fog_id], axis=1)

            id_anom_fill = fog_id_spt_anom.fillna(norm_neighbour_fog_id)
            spt_anom_fill_lst.append(id_anom_fill)

            # fill past uncertainties
            id_sig_past_df = np.sqrt(max_neighbour_sig_mean_anom.pow(2) + max_sig ** 2)
            sig_spt_anom_df[fog_id] = sig_spt_anom_df[fog_id].fillna(id_sig_past_df)
            spt_anom_sig_fill_lst.append(sig_spt_anom_df[fog_id])


        reg_anom_fill_df = pd.concat(spt_anom_fill_lst, axis='columns')
        reg_anom_fill_path = Path(out_long_dir,str(region)+'_spt_anoms_fill_ref_'+str(year_ini)+'-'+str(year_fin)+'_'+fog_version+'.csv')
        reg_anom_fill_df.to_csv(reg_anom_fill_path)

        reg_anom_sig_fill_df = pd.concat(spt_anom_sig_fill_lst, axis='columns')
        reg_anom_sig_fill_path = Path(out_long_dir, str(region)+'_spt_ERRORs_fill_ref_'+str(year_ini)+'-'+str(year_fin)+'_'+fog_version+'.csv')
        reg_anom_sig_fill_df.to_csv(reg_anom_sig_fill_path)




    print('.........................................................................................')
    print('"The End"')
    print('.........................................................................................')


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Section: 3_Kriging_global_CE_spatial_anomaly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def calc_OCE_and_error_global_gla_reg_anom(fog_version: str,
                                           yr_ini: int,
                                           yr_end: int,
                                           path_spt_anom: str,
                                           in_data_geo: str,
                                           output_data_path_string: str,
                                           reg_lst: List[str]) -> None:
    """
    Calculate the observational consensus estimate for every individual glacier

    calc_OCE_and_error_global_gla_reg_anom.py

    Author: idussa
    Date: Feb 2021
    Last changes: Feb 2021

    Scripted for Python 3.7

    Description:
    This script reads glacier-wide mass balance data edited from WGMS FoG database
    and regional glacier anomalies produced by calc_regional_anomalies_and_error.py
    and provides the observational consensus estimate for every individual glacier
    with available geodetic observations WGMS Id

    Input:  GEO_MASS_BALANCE_DATA_20200824.csv
            Regional_anomalies_ref_period_2009-2018.csv
            (UTF-8 encoding)

    Parameters
    ----------
    fog_version_string: str
    A string input of the form 'YYYY-01' indicating the FOG version (e.g., '2025-01')

    Returns
    -------
    Tuple[pd.DataFrame,pd.DataFrame]
        Two pd.Dataframe objects (within a tuple); the first is a dataset of WGMS coordinates and
        the second is the outputted mass balance dataset

    """
    path = os.getcwd()

    out_dir = Path(path,output_data_path_string,f'fog-{fog_version}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # out = os.path.join(path, 'out_data_'+fog_version)
    out = os.path.join(path,output_data_path_string,f'out_data_{fog_version}')
    if not os.path.exists(out):
        os.mkdir(out)

    # out_dir = os.path.join(path, 'out_data_'+fog_version , 'OCE_files_by_region')
    out_dir = os.path.join(path,output_data_path_string, f'out_data_{fog_version}', 'OCE_files_by_region')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    #  Make a list of the full period of interest 1915 to present
    yr_lst = list(range(yr_ini, yr_end + 1))
    # print(yr_lst)

    min_year_geo_obs = 0 # start year of the period of interest, only geodetic from this year on will be considered, 0 if to use the start of the anomaly period
    min_lenght_geo= 5 # minimum lenght in years of the geodetic observations accounted for anomaly calibration
    run = 'spt_anom'

    # read global geodetic mass-balance data from csv into dataframe
    geo_df = pd.read_csv(in_data_geo, encoding='latin1', delimiter=',', header=0, index_col='WGMS_ID').sort_index()
    geo_df.reset_index(inplace=True)

    for region in reg_lst:
        start_time = time.time()
        print('working on region, ', region)

        # # create regional directory for regional glaciers OCE
        # out_reg_dir= os.path.join(path, 'out_data_'+fog_version, str(region) + '_oce_by_gla')
        # if not os.path.exists(out_reg_dir):
        #     os.mkdir(out_reg_dir)

        ## create regional OCE and sigma OCE empty dataframes, including 3 error sources dataframes and the full error
        reg_oce_df = pd.DataFrame(index=yr_lst)
        reg_oce_df.index.name = 'YEAR'
        reg_sig_dh_oce_df = pd.DataFrame(index=yr_lst)
        reg_sig_dh_oce_df.index.name = 'YEAR'
        reg_sig_rho_oce_df = pd.DataFrame(index=yr_lst)
        reg_sig_rho_oce_df.index.name = 'YEAR'
        reg_sig_anom_oce_df = pd.DataFrame(index=yr_lst)
        reg_sig_anom_oce_df.index.name = 'YEAR'

        reg_sig_oce_df = pd.DataFrame(index=yr_lst)
        reg_sig_oce_df.index.name = 'YEAR'

        ## number crunching: select geodetic data for glacier region group
        if region == 'SA1':
            geo_reg_df = geo_df.loc[(geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-01')]
            # print(geo_reg_df)
            # exit()

        elif region == 'SA2':
            geo_reg_df = geo_df.loc[(geo_df['GLACIER_SUBREGION_CODE'] == 'SAN-02')]
            # print(geo_reg_df)
            # exit()

        else:
            geo_reg_df = geo_df.loc[(geo_df['GLACIER_REGION_CODE'] == str(region))]

        geo_reg_df['sigma_elevation_chg'] = geo_reg_df['sigma_elevation_chg'].fillna(geo_reg_df['sigma_elevation_chg'].mean())

        ## create a list of wgms_ids belonging to the region group
        reg_wgms_id_lst= geo_reg_df['WGMS_ID'].unique().tolist()
        # print(reg_wgms_id_lst)
        # print('Nb glaciers in region ' + str(region) + ' with geodetic obs C3S 2020: ' + str(len(reg_wgms_id_lst)))

        # # read regional anomaly data and uncertainties from csv files into dataframe
        all_CE_files = [f for f in os.listdir(path_spt_anom) if f.endswith('.csv')]
        reg_spt_anom_name = fnmatch.filter(all_CE_files, region + '*.csv')

        reg_spt_anom_file = os.path.join(path_spt_anom, reg_spt_anom_name[0])
        reg_spt_anom_df = pd.read_csv(reg_spt_anom_file, encoding='utf-8', delimiter=',', header=0, index_col='YEAR')

        reg_spt_anom_err_file = os.path.join(path_spt_anom, reg_spt_anom_name[1])
        reg_spt_anom_err_df = pd.read_csv(reg_spt_anom_err_file, encoding='utf-8', delimiter=',', header=0, index_col='YEAR')

        ############################################################################################################################

        ###### CALCULATING OCE: Loop through all glaciers in the region with available geodetic estimates ######

        for fog_id in reg_wgms_id_lst:

            print('working on region, ', region, '- glacier Id, ', fog_id)

            # # create individual glacier directory
            # out_gla_dir = os.path.join(out_reg_dir, 'fog_Id_' + str(fog_id) + '_oce')
            # if not os.path.exists(out_gla_dir):
            #     os.mkdir(out_gla_dir)

            id_spt_anom_df = reg_spt_anom_df[[str(fog_id)]]
            id_spt_anom_err_df = reg_spt_anom_err_df[[str(fog_id)]]

            # # Define period of the complete anomaly series
            val = id_spt_anom_df[str(fog_id)].loc[id_spt_anom_df.first_valid_index()]

            if min_year_geo_obs == 0:
                if id_spt_anom_df.loc[id_spt_anom_df[str(fog_id)] == val].index[0] > 2000: ## For anomalies startig after 2000, use 2000 in order to use the geodetic estimates that start 2000
                    min_year = 2000
                else:
                    min_year = id_spt_anom_df.loc[id_spt_anom_df[str(fog_id)] == val].index[0]
            else:
                if id_spt_anom_df.loc[id_spt_anom_df[str(fog_id)] == val].index[0] > 2000:
                    min_year = 2000
                else:
                    min_year = min_year_geo_obs

            max_year = id_spt_anom_df.index.max()

            # # create geodetic mass balance series and geodetic Dataframe for selected glacier
            geo_mb_gla_df = geo_reg_df.loc[(geo_reg_df['WGMS_ID'] == fog_id)]
            geo_mb_gla_df['POR'] = geo_mb_gla_df['fin_date'] - geo_mb_gla_df['ini_date']
            geo_mb_gla_df = geo_mb_gla_df.loc[geo_mb_gla_df['POR'] > 5]

            # # Select geodetic estimates inside the period of interest and longer than min_lenght_geo

            geo_mb_gla_df = geo_mb_gla_df[['WGMS_ID', 'ini_date', 'fin_date', 'mb_chg_rate', 'sigma_tot_mb_chg', 'sigma_elevation_chg', 'elevation_chg_rate']]
            geo_ind_gla_sel_df = geo_mb_gla_df.loc[(geo_mb_gla_df['ini_date'] >= min_year - 2) & (geo_mb_gla_df['fin_date'] <= max_year + 1)]

            #create empty dataframes for calibrated series, calibrated series uncertainty, sigma geodetic uncertainty and distance to observation period
            cal_series_df = pd.DataFrame(index=yr_lst)
            cal_series_df.index.name='YEAR'

            sig_dh_cal_series_df = pd.DataFrame(index=yr_lst)
            sig_dh_cal_series_df.index.name='YEAR'

            sig_rho_cal_series_df = pd.DataFrame(index=yr_lst)
            sig_rho_cal_series_df.index.name='YEAR'

            sig_anom_cal_series_df = pd.DataFrame(index=yr_lst)
            sig_anom_cal_series_df.index.name = 'YEAR'

            sigma_geo_df = pd.DataFrame(index=yr_lst)
            sigma_geo_df.index.name= 'YEAR'

            dist_geo_df = pd.DataFrame(index=yr_lst)
            dist_geo_df.index.name= 'YEAR'
            dist_geo_df = dist_geo_df.reset_index()
            # print(geo_ind_gla_sel_df.columns)
            # exit()
            ###### Calculate the calibrated series #####
            for index, row in geo_ind_gla_sel_df.iterrows():
                if (int(row['fin_date'])-int(row['ini_date'])) > min_lenght_geo:
                    ref_period_geo_obs = range(int(row['ini_date']), int(row['fin_date']))
                    ref_anom_period_df = id_spt_anom_df.loc[id_spt_anom_df.index.isin(ref_period_geo_obs)]
                    ### -------here there is a problem with the dates !!!! need to correct geodetic values to hydrological years
                    avg_ref_anom = ref_anom_period_df.mean()
                    ref_anom= id_spt_anom_df - avg_ref_anom
                    # print(ref_anom)

                    cal_val = row['mb_chg_rate'] + ref_anom
                    cal_series_df['serie_'+str(index)] = cal_val[str(fog_id)]

                    # Three uncertainty sources: elevation change, density conversion, and anomaly

                    # We save them in the same unit of mass change, multiplying dh by density
                    # 1. Error in mass change from dh error sources
                    sig_dh_cal_series_df['sig_serie_'+str(index)] = row['sigma_elevation_chg'] * 0.85
                    # 2. Error in mass change from density error sources
                    sig_rho_cal_series_df['sig_serie_'+str(index)] = np.abs(row['elevation_chg_rate']) * 0.06
                    # 3. Error in mass change from anomalies
                    sig_anom_cal_series_df['sig_serie_'+str(index)] = id_spt_anom_err_df

                    # # Create geodetic estimate uncertainty dataframe
                    sigma_geo_df['serie_' + str(index)]=row['sigma_tot_mb_chg']
                    i_date=int(row['ini_date'])
                    f_date=int(row['fin_date'])

                    # # Create Distance to geodetic observation period dataframe
                    dist_geo_df['serie_' + str(index)]= dist_geo_df['YEAR'].apply(lambda row: dis_fil(row, i_date, f_date))
                else:
                    pass

            # if cal_series_df.empty == True:
            #     print('No calibrated series for glacier ' + str(wgms_id))

            if cal_series_df.empty == True:
                continue
                # os.rmdir(out_gla_dir)

            ###### Apply weights to calculate the mean calibrated series #####

            ## Calculate weight related to geodetic estimate uncertainty
            if cal_series_df.empty == False:

                if math.isnan(geo_ind_gla_sel_df['sigma_tot_mb_chg'].max()) == True:
                    fill_sigma = 1.0
                else:
                    fill_sigma = geo_ind_gla_sel_df['sigma_tot_mb_chg'].max()

                # weight_dir = out_gla_dir + 'wgmsId_' + str(wgms_id) + '_weights\\'
                # if not os.path.exists(weight_dir):
                #     os.mkdir(weight_dir)

                sigma_geo_df.fillna(fill_sigma, inplace=True)
                # sigma_geo_df.to_csv(weight_dir + 'Sigma_series_WGMS_ID_' + str(wgms_id) + '.csv', index=False)
                sigma_ratio_df=sigma_geo_df.apply(lambda x: (1 / x))
                wgt1_sigma_df=sigma_ratio_df.div(sigma_ratio_df.sum(axis=1), axis=0) # pass to percentage
                # wgt1_sigma_df.to_csv(weight_dir + 'Weight_percent_Sigma_series_WGMS_ID_' +str(wgms_id)+ '.csv', index=False)

                ## Calculate weight related to distance to the geodetic estimate survey period
                p=2 ## P value for inverse distance weighting
                dist_geo_df=dist_geo_df.set_index('YEAR')
                # dist_geo_df.to_csv(weight_dir + 'Distance_series_WGMS_ID_' + str(wgms_id) + '.csv', index=False)
                inv_dist_df=dist_geo_df.apply(lambda x: (1 / x) ** p)
                wgt2_dist_df=inv_dist_df.div(inv_dist_df.sum(axis=1), axis=0) # pass to percentage
                # wgt2_dist_df.to_csv(weight_dir + 'Weight_percent_Distance_series_WGMS_ID_' +str(wgms_id)+ '.csv', index=False)

                ## Calculate weight related to uncertaity and distance combined
                W1_W2_comb_df=wgt1_sigma_df.add(wgt2_dist_df)

                ##### Calculate MEANS of calibrated series: Artihmetic, Weight_1, Weight_2, Weight_combined #####

                cal_mean_df = pd.DataFrame(index=yr_lst)
                cal_mean_df.index.name='YEAR'
                # print(cal_series_df)

                ### Apply the weights to the calibrated series
                cal_series_W1_df = cal_series_df.mul(wgt1_sigma_df)
                cal_series_W2_df = cal_series_df.mul(wgt2_dist_df)
                cal_series_W1_W2_df=(cal_series_df.mul(W1_W2_comb_df))/2

                ## calibrated series means
                cal_mean_df['MEAN']=cal_series_df.mean(axis=1)
                cal_mean_df['MEAN_sigma_W']=cal_series_W1_df.sum(axis=1, min_count=1)
                cal_mean_df['MEAN_dist_W']=cal_series_W2_df.sum(axis=1, min_count=1)
                cal_mean_df['MEAN_combined_W']=cal_series_W1_W2_df.sum(axis=1, min_count=1)

                # # Plot the different means of the calibrated series
                # # print(cal_mean_df)
                # fig=cal_mean_df.plot()
                # plt.savefig(out_gla_dir + 'wgmsId_' + str(fog_id) + '_weights\\Fig_diff_mean_cal_series_glacier_id_' + str(fog_id) + '_' + region + '_Geo_+' + str(min_lenght_geo) + 'years.png')
                # plt.close()
                # exit()

                ## cumulative series of the different means
                cal_mean_cum_df = pd.DataFrame(index=yr_lst)
                cal_mean_cum_df.index.name='YEAR'

                cal_mean_cum_df['Cum_MEAN']=cal_mean_df['MEAN'].cumsum()
                cal_mean_cum_df['Cum_MEAN_sigma_W']=cal_mean_df['MEAN_sigma_W'].cumsum(skipna=True)
                cal_mean_cum_df['Cum_MEAN_dist_W']=cal_mean_df['MEAN_dist_W'].cumsum(skipna=True)
                cal_mean_cum_df['Cum_MEAN_combined_W']=cal_mean_df['MEAN_combined_W'].cumsum(skipna=True)

                ## Plot the cumulative values from the different means of the calibrated series
                # # print(cal_mean_cum_df)
                # fig=cal_mean_cum_df.plot()
                # plt.savefig(out_gla_dir + 'wgmsId_' + str(wgms_id) + '_weights\\Fig_cum_series_diff_mean_glacier_id_' + str(wgms_id) + '_Geo_+' + str(min_lenght_geo) + 'years.png')
                # plt.close()

                ############################################################################################################################
                # We take the mean error for each source, which is fully justified
                # (they will be correlated for the same glacier)

                reg_sig_dh_oce_df[fog_id] = sig_dh_cal_series_df.mean(axis=1)
                reg_sig_rho_oce_df[fog_id] = sig_rho_cal_series_df.mean(axis=1)
                reg_sig_anom_oce_df[fog_id] = sig_anom_cal_series_df.mean(axis=1)

                # Total error
                reg_sig_oce_df[fog_id] = np.sqrt(sig_anom_cal_series_df.mean(axis=1) ** 2 +
                                                 sig_rho_cal_series_df.mean(axis=1) ** 2 +
                                                 sig_dh_cal_series_df.mean(axis=1) ** 2)


                # sig_oce.to_csv(out_gla_dir + 'sigma_CE_fog_id_' + str(fog_id) + '_' + region + '.csv')
                #

                # TODO: INES I HAD TO PUT THIS OUT OF THE IF LOOP FOR PLOT BELOW, OTHERWISE "ISL_regional_CEs.csv" WAS EMPTY
                #  OR IT WAS DOING AN EXIT()

                oce_df = pd.DataFrame()
                oce_df['weighted_MEAN'] = cal_series_W1_W2_df.sum(axis=1, min_count=1)
                oce_df['normal_MEAN'] = cal_series_df.mean(axis=1)
                oce_df = oce_df.rename(columns={'weighted_MEAN': fog_id})
                oce_df = oce_df[fog_id]
                reg_oce_df[fog_id] = oce_df
                # print(oce_df)

                # ## Plot and save individual glacier OCE
                #
                # plot_gla_oce(cal_series_df, geo_ind_gla_sel_df, fog_id, min_year, max_year, min_lenght_geo, region, run, out_gla_dir)
                # plot_gla_oce_and_unc(cal_series_df, oce_df, geo_ind_gla_sel_df, reg_sig_oce_df, fog_id, min_year, max_year, min_lenght_geo, region, run, out_gla_dir)
                # exit()

            read_time4 = time.time()
            print("--- %s seconds ---" % (read_time4 - start_time))


        ### Save regional OCEs
        reg_oce_df.to_csv(os.path.join(out_dir, region + '_regional_CEs.csv'))
        reg_sig_dh_oce_df.to_csv(os.path.join(out_dir, region + '_regional_sigma_dh_CEs.csv'))
        reg_sig_rho_oce_df.to_csv(os.path.join(out_dir, region + '_regional_sigma_rho_CEs.csv'))
        reg_sig_anom_oce_df.to_csv(os.path.join(out_dir, region + '_regional_sigma_anom_CEs.csv'))


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Section: 4_Kriging_regional_mass_balance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def calc_reg_ba(fog_version: str,
                path_oce: str,
                output_data_path_string: str,
                reg_lst: List[str],
                rgi_reg: dict,
                rgi_code: dict,
                input_b_and_sigma_path: str,
                DM_series_min_yr: int,
                in_data_rgi_area: str,
                in_data_glims: str,
                in_data_attribs: str,
                in_data_regional_series: str) -> None:

    """
    Calculate the regional mass loss

    calc_reg_mass_loss.py

    Author: idussa
    Date: Feb 2021
    Last changes: Feb 2021

    Scripted for Python 3.7

    Description:
    This script reads glacier-wide mass balance data edited from WGMS FoG database
    and regional glacier anomalies produced by calc_regional_anomalies_and_error.py
    and provides the observational consensus estimate for every individual glacier
    with available geodetic observations WGMS Id

    Input:  C3S_GLACIER_DATA_20200824.csv
            OCE_files_by_region\\
            (UTF-8 encoding)

    Return: tbd.svg
    """

    #################################################################################################
    ##    Define input datasets
    #################################################################################################
    path = os.getcwd()

    #################################################################################################
    ##    Define parameters
    #################################################################################################
    if Path(output_data_path_string).is_absolute() == True:
        raise ValueError('output_data_path_string must be provided as a relative path from the working directory of your workflow script.')
    out_dir = Path(path,output_data_path_string,f'fog-{fog_version}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ###################################################################################
    ## If not done before: Bring files together
    Reg_mb_lst = []
    Reg_sig_mb_lst = []

    if Path(input_b_and_sigma_path).is_absolute() == True:
        raise ValueError('input_b_and_sigma_path must be provided as a relative path from the working directory of your workflow script.')

    # !! If logic explicitly added by Devin as a quick way to save on computation

    regional_AreaWeighted_path = Path(input_b_and_sigma_path, 'Regional_B_series_AreaWeighted.csv')
    regional_uncertainty_path = Path(input_b_and_sigma_path, 'Regional_B_series_uncertainty.csv')

    if not os.path.exists(regional_AreaWeighted_path and regional_uncertainty_path):

        for region in reg_lst:
            in_data = Path(input_b_and_sigma_path, region +'_B_and_sigma.csv')
            data_df = pd.read_csv(in_data, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
            mb_df = pd.DataFrame(data_df['Aw_B m w.e.'])
            mb_df = mb_df.rename(columns={'Aw_B m w.e.': region})
            sig_df = pd.DataFrame(data_df['sigma_B m w.e.'])
            sig_df = sig_df.rename(columns={'sigma_B m w.e.': region})
            Reg_mb_lst.append(mb_df)
            Reg_sig_mb_lst.append(sig_df)

        Reg_mb_df = pd.concat(Reg_mb_lst, axis = 1)
        Reg_sig_mb_df = pd.concat(Reg_sig_mb_lst, axis = 1)

        ### Save regional Mass balance series
        Reg_mb_df.to_csv()
        Reg_sig_mb_df.to_csv()
    else:
        print('Regional AreaWeighted and uncertainty files already exist')


    #################################################################################################
    ##    READ ID links and areas
    #################################################################################################

    if Path(in_data_rgi_area).is_absolute() == True:
        raise ValueError('in_data_rgi_area must be provided as a relative path from the working directory of your workflow script.')
    id_rgi_area_df = pd.read_csv(in_data_rgi_area, encoding='latin1', delimiter=',', header=0)

    if Path(in_data_glims).is_absolute() == True:
        raise ValueError('in_data_glims must be provided as a relative path from the working directory of your workflow script.')
    id_glims_coords_df = pd.read_csv(in_data_glims, encoding='latin1', delimiter=',', header=0 ,usecols= ['glac_id','db_area','CenLat', 'CenLon'])
    id_glims_coords_df = id_glims_coords_df.rename(columns={'glac_id': 'RGIId'}).set_index('RGIId')

    if Path(in_data_attribs).is_absolute() == True:
        raise ValueError('in_data_attribs must be provided as a relative path from the working directory of your workflow script.')
    rgi_path= Path(in_data_attribs)

    if Path(in_data_regional_series).is_absolute() == True:
        raise ValueError('in_data_regional_series must be provided as a relative path from the working directory of your workflow script.')
    in_data_zemp = Path(in_data_regional_series)

    ###### Calculate specific glacier mass balance by region ######

    Reg_mb_df = pd.DataFrame()
    Reg_sig_mb_df = pd.DataFrame()
    Reg_sumary_df = pd.DataFrame()

    for region in reg_lst:
        print('working on region, ', region)

        ## Define and read input:   regional OCE series and three sources of uncertainty
        in_oce_file = Path(path_oce, region + '_regional_CEs.csv')
        in_sig_dh_oce_file = Path(path_oce, region + '_regional_sigma_dh_CEs.csv')
        in_sig_rho_oce_file = Path(path_oce, region + '_regional_sigma_rho_CEs.csv')
        in_sig_anom_oce_file = Path(path_oce, region + '_regional_sigma_anom_CEs.csv')

        oce_df = pd.read_csv(in_oce_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
        sig_anom_oce_df = pd.read_csv(in_sig_anom_oce_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
        yr = sig_anom_oce_df.first_valid_index()

        sig_anom_oce_df = sig_anom_oce_df.loc[sig_anom_oce_df.index >= yr]
        sig_dh_oce_df = pd.read_csv(in_sig_dh_oce_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
        sig_dh_oce_df = sig_dh_oce_df.loc[sig_dh_oce_df.index >= yr]
        sig_rho_oce_df = pd.read_csv(in_sig_rho_oce_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
        sig_rho_oce_df = sig_rho_oce_df.loc[sig_rho_oce_df.index >= yr]

        sig_oce_df = sig_dh_oce_df.copy()
        sig_oce_df = np.sqrt(sig_dh_oce_df**2 + sig_rho_oce_df**2 + sig_anom_oce_df**2)

        nan_lst = sig_oce_df.columns[sig_oce_df.isna().any()].tolist()
        sig_oce_df = sig_oce_df.drop(columns = nan_lst)
        oce_df = oce_df.drop(columns = nan_lst)
        sig_rho_oce_df = sig_rho_oce_df.drop(columns = nan_lst)
        sig_anom_oce_df = sig_anom_oce_df.drop(columns = nan_lst)
        sig_dh_oce_df = sig_dh_oce_df.drop(columns = nan_lst)

        filename = Path(rgi_path , rgi_code[region]+'_rgi60_'+rgi_reg[region]+'.csv')
        if region == 'GRL':
            rgi_df_all = pd.read_csv(filename, encoding='latin1', delimiter=',', header=0,usecols=['RGIId', 'CenLat', 'CenLon', 'Connect'], index_col=[0])
            rgi_df = rgi_df_all.loc[rgi_df_all['Connect'] != 2]
            l1l2_lst = rgi_df.index.to_list()

        else:
            rgi_df = pd.read_csv(filename, encoding='latin1', delimiter=',', header=0, usecols= ['RGIId', 'CenLat', 'CenLon'], index_col=[0])

        # Keep only glaciers in the region
        if region == 'SA1':
            rgi_area_df= id_rgi_area_df.loc[id_rgi_area_df['GLACIER_SUBREGION_CODE']== 'SAN-01']
        elif region == 'SA2':
            rgi_area_df= id_rgi_area_df.loc[id_rgi_area_df['GLACIER_SUBREGION_CODE']== 'SAN-02']
        elif region == 'GRL':
            rgi_area_df = id_rgi_area_df.loc[id_rgi_area_df['RGIId'].isin(l1l2_lst)]
        else:
            rgi_area_df = id_rgi_area_df.loc[(id_rgi_area_df['GLACIER_REGION_CODE'] == region)]


        nb_gla_reg = len(rgi_area_df.index)
        tot_area_rgi_reg = rgi_area_df['AREA'].sum()

        ## select wgms_ids belonging to the region group
        wgms_id_lst = oce_df.columns.to_list()
        wgms_id_lst = [int(i) for i in wgms_id_lst]

        ## Calculate total area of observed glaciers presenting an area value in FoG

        ## Remove glacier IDS with no Area, only for FoG areas
        rgi_area_df = rgi_area_df.set_index('WGMS_ID')
        id_lst=[]
        for id in wgms_id_lst:
            if id in rgi_area_df.index:
                id_lst.append(id)
            else:
                pass

        gla_obs_df = rgi_area_df.loc[id_lst]
        tot_area_obs = gla_obs_df['AREA'].sum()
        nb_gla_obs = len(gla_obs_df)

        gla_obs_df = gla_obs_df.reset_index().set_index('RGIId')

        if region == 'CAU':
            tot_area_rgi_reg = id_glims_coords_df['db_area'].sum()
            gla_obs_area_coord_df = pd.merge(gla_obs_df, id_glims_coords_df, left_index=True, right_index=True).drop_duplicates()
        else:
            gla_obs_area_coord_df = pd.merge(gla_obs_df, rgi_df, left_index=True, right_index=True)

        gla_obs_area_coord_df =gla_obs_area_coord_df.reset_index().set_index('WGMS_ID')
        gla_obs_area_coord_df = gla_obs_area_coord_df[~gla_obs_area_coord_df.index.duplicated()]

        print('total area region / tot nb glaciers in region :  ', tot_area_rgi_reg, ' / ', nb_gla_reg)
        print('total area glaciers observed / number glaciers with observations :  ', tot_area_obs, ' / ', nb_gla_obs)

        ####### Calculate all glaciers time series and uncertainties ##########

        ## 1. Calculate OCE series for unobserved glaciers as the Weigthed mean from the regional glacier sample with observations
        rel_mb_df = pd.DataFrame()
        rel_sig_dh_mb_df = pd.DataFrame()
        rel_sig_rho_mb_df = pd.DataFrame()
        rel_sig_anom_mb_df = pd.DataFrame()
        rel_sig_mb_df = pd.DataFrame()

        list_df = []
        list_areas = []
        list_lat = []
        list_lon = []
        for id in id_lst:

            # Read area, mass balance estimate and three uncertainty sources
            area= gla_obs_area_coord_df.loc[id, 'AREA']
            lon= gla_obs_area_coord_df.loc[id, 'CenLon']
            lat= gla_obs_area_coord_df.loc[id, 'CenLat']
            mb_oce= oce_df[str(id)]
            sig_dh_oce = sig_dh_oce_df[str(id)]
            sig_rho_oce = sig_rho_oce_df[str(id)]
            sig_anom_oce = sig_anom_oce_df[str(id)]

            # Area weighting for all (uncertainties are also combined pairwise with area-weight in exact propagation)
            mb_oce_rel = (mb_oce * area) / tot_area_obs
            sig_dh_rel = (sig_dh_oce * area)/tot_area_obs
            sig_rho_rel = (sig_rho_oce * area)/tot_area_obs
            sig_anom_rel = (sig_anom_oce * area)/tot_area_obs
            # obs_df["id"] = mb_oce_rel
            # obs_df[id] = sig_rel

            # Dataframes per ID
            rel_mb_df[id] = mb_oce_rel
            # The three error sources
            rel_sig_dh_mb_df[id] = sig_dh_rel
            rel_sig_rho_mb_df[id] = sig_rho_rel
            rel_sig_anom_mb_df[id] = sig_anom_rel

            # The total error
            rel_sig_mb_df[id] = np.sqrt(sig_dh_rel**2 + sig_rho_rel**2 + sig_anom_rel**2)

            # Store lat/lon in a dataframe for "observed" glaciers
            list_lat.append(lat)
            list_lon.append(lon)

        # Area-weighted OCE for observed glaciers
        Aw_oce_obs_df = rel_mb_df.sum(axis=1, min_count=1)


        ## 2. Calculate OCE uncertainties for observed glaciers

        # Weighted mean Sigma OCE of observed glaciers (only to use later for unobserved glaciers)
        Sig_oce_obs_gla = rel_sig_mb_df.sum(axis=1, min_count=1)
        Sig_dh_mb_gla = rel_sig_dh_mb_df.sum(axis=1, min_count=1)
        Sig_rho_mb_gla = rel_sig_rho_mb_df.sum(axis=1, min_count=1)
        Sig_anom_mb_gla = rel_sig_anom_mb_df.sum(axis=1, min_count=1)

        ## 3. Add OCE series and uncertainties for unobserved glaciers

        # Id -9999 for unobserved glaciers, OCE is the area weighthed average of the regional observed series

        out_oce = Path(out_dir, 'spt_CEs_obs-unobs_per_region')
        if not os.path.exists(out_oce):
            os.makedirs(out_oce)

        oce_df['unobs_gla'] = Aw_oce_obs_df
        oce_df.to_csv(Path(out_oce, region +'_CEs_obs-unobs.csv'))

        sig_oce_df['unobs_gla'] = Sig_oce_obs_gla
        sig_oce_df.to_csv(Path(out_oce, region + '_sigma_tot_CEs_obs-unobs.csv'))

        sig_rho_oce_df['unobs_gla'] = Sig_rho_mb_gla
        sig_rho_oce_df.to_csv(Path(out_oce, region + '_sig_rho_CEs_obs-unobs.csv'))
        sig_anom_oce_df['unobs_gla'] = Sig_anom_mb_gla
        sig_anom_oce_df.to_csv(Path(out_oce, region + '_sigma_anom_CEs_obs-unobs.csv'))
        sig_dh_oce_df['unobs_gla'] = Sig_dh_mb_gla
        sig_dh_oce_df.to_csv(Path(out_oce, region + '_sigma_dh_CEs_obs-unobs.csv'))

        ####### Calculate Regional specific mass balance time series ##########

        Reg_mb_df[region] = Aw_oce_obs_df
        nb_unobs_gla = nb_gla_reg - nb_gla_obs

        # # Fully correlated propagation for residual error of anomaly
        # Aw_sig_anom_obs_df = rel_sig_anom_mb_df.sum(axis=1, min_count=1)

        # We can't apply to the whole YEAR/ID dataframe at once here, we need to loop for each YEAR of the dataframes
        # to compute the pairwise error propagation for dh and density across all glaciers of that year
        list_sig_dh_yearly = []
        list_sig_rho_yearly = []
        list_sig_anom_yearly = []

        for i in range(len(rel_sig_dh_mb_df.index)):

            print(f"Propagating uncertainties from glaciers to region for year {rel_sig_dh_mb_df.index[i]}")

            # Create dataframe with dh errors, lat and lon
            yearly_dh_df = rel_sig_dh_mb_df.iloc[i, :]
            yearly_dh_df["errors"] = yearly_dh_df.values.flatten()
            yearly_dh_df["lat"] = np.array(list_lat)
            yearly_dh_df["lon"] = np.array(list_lon)

            # Spatial correlations for dh
            sig_dh_obs = wrapper_latlon_double_sum_covar(yearly_dh_df, spatialcorr_func=sig_dh_spatialcorr)

            # Check propagation works as intended: final estimate is between fully correlated and independent
            sig_dh_fullcorr = np.sum(yearly_dh_df["errors"])
            sig_dh_uncorr = np.sqrt(np.sum(yearly_dh_df["errors"]**2))
            print(f"{sig_dh_uncorr}, {sig_dh_obs}, {sig_dh_fullcorr}")
            assert sig_dh_uncorr <= sig_dh_obs <= sig_dh_fullcorr

            # Create dataframe with rho errors, lat and lon
            yearly_rho_df = rel_sig_rho_mb_df.iloc[i, :]
            yearly_rho_df["errors"] = yearly_rho_df.values
            yearly_rho_df["lat"] = np.array(list_lat)
            yearly_rho_df["lon"] = np.array(list_lon)

            # Spatial correlation for rho for a 1-year period
            def sig_rho_dv_spatialcorr_yearly(d):
                return sig_rho_dv_spatialcorr(d, dt=1)
            sig_rho_obs = wrapper_latlon_double_sum_covar(yearly_rho_df, spatialcorr_func=sig_rho_dv_spatialcorr_yearly)

            # Check propagation works as intended: final estimate is between fully correlated and independent
            sig_rho_fullcorr = np.sum(yearly_rho_df["errors"])
            sig_rho_uncorr = np.sqrt(np.sum(yearly_rho_df["errors"] ** 2))
            # print(f"{sig_rho_uncorr}, {sig_rho_obs}, {sig_rho_fullcorr}")
            assert sig_rho_uncorr <= sig_rho_obs <= sig_rho_fullcorr

            # Create dataframe with anom errors, lat and lon
            yearly_anom_df = rel_sig_anom_mb_df.iloc[i, :]
            # print(yearly_anom_df)
            # exit()
            yearly_anom_df["errors"] = yearly_anom_df.values
            yearly_anom_df["lat"] = np.array(list_lat)
            yearly_anom_df["lon"] = np.array(list_lon)

            # Spatial correlations for anom
            sig_anom_obs = wrapper_latlon_double_sum_covar(yearly_anom_df, spatialcorr_func=ba_anom_spatialcorr)

            # Check propagation works as intended: final estimate is between fully correlated and independent
            sig_anom_fullcorr = np.sum(yearly_anom_df["errors"])
            sig_anom_uncorr = np.sqrt(np.sum(yearly_anom_df["errors"]**2))
            # print(f"{sig_anom_uncorr}, {sig_anom_obs}, {sig_anom_fullcorr}")
            assert sig_anom_uncorr <= sig_anom_obs <= sig_anom_fullcorr

            # Append to list for each yearly period
            list_sig_dh_yearly.append(sig_dh_obs)
            list_sig_rho_yearly.append(sig_rho_obs)
            list_sig_anom_yearly.append(sig_anom_obs)

        # And write back the 1D list of uncertainties into an indexed (by YEAR) dataframe
        Aw_sig_dh_obs_df =  pd.DataFrame(index=sig_anom_oce_df.index.copy())
        Aw_sig_dh_obs_df['dh']= list_sig_dh_yearly

        Aw_sig_rho_obs_df = pd.DataFrame(index=sig_anom_oce_df.index.copy())
        Aw_sig_rho_obs_df['rho'] = list_sig_rho_yearly

        Aw_sig_anom_obs_df = pd.DataFrame(index=sig_anom_oce_df.index.copy())
        Aw_sig_anom_obs_df['anom']= list_sig_anom_yearly

        Sig_oce_obs_propag = np.sqrt(Aw_sig_dh_obs_df['dh']**2 + Aw_sig_rho_obs_df['rho']**2 + Aw_sig_anom_obs_df['anom']**2)
        # print(Sig_oce_obs_propag)
        # exit()

        # Defining area-weighted uncertainty of unobserved glaciers based on the mean uncertainty of observed glaciers
        area_unobs = round(tot_area_rgi_reg, 2) - round(tot_area_obs, 2)
        sig_W_unobs = Sig_oce_obs_gla * (area_unobs / tot_area_rgi_reg)

        # Area-weight the observed glaciers before combining in final uncertainty
        sig_W_obs = Sig_oce_obs_propag * (tot_area_obs / tot_area_rgi_reg)


        # Final regional uncertainty!
        reg_sig = np.sqrt(sig_W_obs**2 + sig_W_unobs**2)

        Reg_sig_mb_df[region] = reg_sig

        Reg_sumary_df['Aw_B m w.e.'] = Aw_oce_obs_df/1000
        Reg_sumary_df['sigma_B m w.e.'] = reg_sig / 1000
        Reg_sumary_df['sigma_propagated m w.e.'] = Sig_oce_obs_propag / 1000
        Reg_sumary_df['sigma_dh m w.e.'] = Aw_sig_dh_obs_df / 1000
        Reg_sumary_df['sigma_rho m w.e.'] = Aw_sig_rho_obs_df / 1000
        Reg_sumary_df['sigma_anom m w.e.'] = Aw_sig_anom_obs_df / 1000
        Reg_sumary_df.to_csv(Path(out_dir, region +'_B_and_sigma.csv'))


    Reg_mb_df = Reg_mb_df.loc[(Reg_mb_df.index >= DM_series_min_yr)] / 1000
    Reg_sig_mb_df = Reg_sig_mb_df.loc[(Reg_sig_mb_df.index >= DM_series_min_yr)] / 1000
    ### Save regional Mass balance series
    Reg_mb_df.to_csv(Path(out_dir, 'Regional_B_series_AreaWeighted_code.csv'))
    Reg_sig_mb_df.to_csv(Path(out_dir, 'Regional_B_series_uncertainty_code.csv'))


    print('.........................................................................................')
    print('"The End - Part 4"')
    print('.........................................................................................')


"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Section: 5_Kriging_regional_mass_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

def calc_reg_mass_loss(fog_version: str,
                       in_areaweighted_file: str,
                       in_uncertainty_file: str,
                       path_oce: str,
                       regional_area_change_file: str,
                       in_data_area: str,
                       regional_area_directory: str,
                       ini_yr: int,
                       fin_yr: int,
                       reg_lst: List[str],
                       rgi_code: dict,
                       rgi_region: dict,
                       output_data_path_string: str) -> None:

    """
    Calculate the regional mass loss

    calc_reg_mass_loss.py

    Author: idussa
    Date: Feb 2021
    Last changes: Feb 2021

    Scripted for Python 3.7

    Description:
    This script reads glacier-wide mass balance data edited from WGMS FoG database
    and regional glacier anomalies produced by calc_regional_anomalies_and_error.py
    and provides the observational consensus estimate for every individual glacier
    with available geodetic observations WGMS Id

    Input:  C3S_GLACIER_DATA_20200824.csv
            OCE_files_by_region\\
            (UTF-8 encoding)

    Return: tbd.svg
    """

    ##########################################
    ##########################################
    """main code"""
    ##########################################
    ##########################################

    #################################################################################################
    ##    Define input datasets
    #################################################################################################

    path = os.getcwd()

    # Affirm input file and directory paths
    if Path(in_areaweighted_file).is_absolute() == True:
        raise ValueError('in_areaweighted_file must be provided as a relative path from the working directory of your workflow script.')
    if Path(in_uncertainty_file).is_absolute() == True:
        raise ValueError('in_uncertainty_file must be provided as a relative path from the working directory of your workflow script.')
    if Path(path_oce).is_absolute() == True:
        raise ValueError('path_oce must be provided as a relative path from the working directory of your workflow script.')
    if Path(regional_area_change_file).is_absolute() == True:
        raise ValueError('regional_area_change_file must be provided as a relative path from the working directory of your workflow script.')
    if Path(in_data_area).is_absolute() == True:
        raise ValueError('in_data_area must be provided as a relative path from the working directory of your workflow script.')
    if Path(regional_area_directory).is_absolute() == True:
        raise ValueError('regional_area_directory must be provided as a relative path from the working directory of your workflow script.')


    #################################################################################################
    ##    Define parameters
    #################################################################################################

    # period to calculate the cumulative mass loss
    ini_yr_full_obs = ini_yr
    fin_yr_obs = fin_yr

    # FULL period
    PoR = list(range(ini_yr_full_obs, fin_yr_obs +1))
    PoR_full = list(range(ini_yr_full_obs, fin_yr_obs+1))

    S_ocean = 362.5 * 10**6 # Cogley et al 2012
    sig_area = 0.05 # Paul et al 2015

    # Define Output parameters
    if Path(output_data_path_string).is_absolute() == True:
        raise ValueError('output_data_path_string must be provided as a relative path from the working directory of your workflow script.')
    out_dir = Path(path,output_data_path_string,f'fog-{fog_version}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    #################################################################################################
    ##    READ input files
    #################################################################################################

    ba_df = pd.read_csv(in_areaweighted_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
    sig_ba_df = pd.read_csv(in_uncertainty_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')

    # ba_df = ba_df / 1000
    # sig_ba_df = sig_ba_df / 1000

    reg_area_zemp_df = pd.read_csv(regional_area_change_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')
    id_area_df = pd.read_csv(in_data_area, encoding='utf-8', delimiter=',', header=0)

    ############################################################################################################################

    ###### Calculate total glacier mass loss by region ######

    cols = ['region', 'area_mean_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + ' [km2]', 'area_mean_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [km2]',
            'percentage_area_obs' ,'DM [Gt yr-1]', 'sigma_DM [Gt yr-1]', 'CUM_DM_'+str(min(PoR))+'_'+str(max(PoR))+' [Gt]', 'sigma_CUM_DM_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [Gt]',
            'B [mwe yr-1]', 'sigma_B [mwe yr-1]',  'SLE [mm yr-1]', 'sigma_SLE [mm yr-1]', 'zemp_DM [Gt yr-1]', 'zemp_sigma_DM [Gt yr-1]',
            'zemp_CUM_DM_'+str(min(PoR))+'_'+str(max(PoR))+' [Gt]']


    glob_cum_df = pd.DataFrame(index=reg_lst, columns=cols)

    Reg_DM_df = pd.DataFrame()
    Reg_sig_DM_df = pd.DataFrame()

    for region in reg_lst:
        print('working in region: ', region)

        out_DM_series = Path(out_dir,'regional_mass_loss_series')
        if not os.path.exists(out_DM_series):
            os.mkdir(out_DM_series)

        in_oce_file = Path(path_oce,region + '_regional_CEs.csv')
        oce_df = pd.read_csv(in_oce_file, encoding='latin1', delimiter=',', header=0, index_col='YEAR')

        rgi_area_df = id_area_df.loc[(id_area_df['GLACIER_REGION_CODE'] == region)].set_index('WGMS_ID')

        if region == 'SA1':
            rgi_area_df = id_area_df.loc[(id_area_df['GLACIER_REGION_CODE'] == 'SAN')].set_index('WGMS_ID')
            rgi_area_df= rgi_area_df.loc[rgi_area_df['GLACIER_SUBREGION_CODE']== 'SAN-01']
        #
        if region == 'SA2':
            rgi_area_df = id_area_df.loc[(id_area_df['GLACIER_REGION_CODE'] == 'SAN')].set_index('WGMS_ID')
            rgi_area_df= rgi_area_df.loc[rgi_area_df['GLACIER_SUBREGION_CODE']== 'SAN-02']

        ## select wgms_ids belonging to the region group
        wgms_id_lst = oce_df.columns.to_list()
        wgms_id_lst = [int(i) for i in wgms_id_lst]
        print(len(wgms_id_lst))

        id_lst=[]
        for id in wgms_id_lst:
            if id in rgi_area_df.index:
                id_lst.append(id)
            else:
                pass
        ## Calculate total area of observed glaciers presenting an area value in FoG

        nb_gla_reg = len(rgi_area_df)
        print(nb_gla_reg)
        gla_obs_df = rgi_area_df.loc[id_lst]

        tot_area_rgi_reg = rgi_area_df['AREA'].sum()
        tot_area_obs_reg = gla_obs_df['AREA'].sum()

        nb_gla_obs = len(gla_obs_df)
        ba_mwe = ba_df[region]
        ba_kmwe = ba_mwe/10**3

        sig_ba_mwe = sig_ba_df[region]
        sig_ba_kmwe = sig_ba_mwe/10**3

        area = reg_area_zemp_df[region]
        # area_reg_for_weight = round(area.loc[2022])

        dm_Gt = ba_kmwe * area

        sig_dm_sum = np.sqrt((sig_ba_kmwe/ba_kmwe)**2 + sig_area**2)
        sig_dm = np.abs(dm_Gt) * sig_dm_sum

        reg_file = pd.DataFrame()
        reg_file['Aw_mwe'] = ba_df[region]
        reg_file['sig_tot_mwe'] = sig_ba_df[region]
        reg_file['area_tot_km2'] = reg_area_zemp_df[region]
        reg_file['DM_Gt'] = dm_Gt
        reg_file['sig_tot_DM'] = sig_dm

        reg_file.to_csv(Path(out_DM_series, 'results_region_' + rgi_code[region] + '_' + region + '_' + rgi_region[region] + '.csv'))

        Reg_DM_df[region] = dm_Gt
        Reg_sig_DM_df[region] = sig_dm

    ####### Calculate cumulative mass loss for PoR ###################################

        mean_area = reg_file['area_tot_km2'].loc[(reg_file.index.isin(PoR))].mean()
        mean_area_full = reg_file['area_tot_km2'].loc[(reg_file.index.isin(PoR_full))].mean()

        B_mwe_yr = reg_file['Aw_mwe'].loc[(reg_file.index.isin(PoR))].mean()
        sigma_B_mwe_yr = np.sqrt(reg_file['sig_tot_mwe'].loc[(reg_file.index.isin(PoR))].pow(2).sum() / len(PoR))

        DM_Gt_yr = reg_file['DM_Gt'].loc[(reg_file.index.isin(PoR))].sum() / len(PoR)
        CUM_DM_Gt = reg_file['DM_Gt'].loc[(reg_file.index.isin(PoR))].sum()

        sigma_DM_Gt_yr = np.sqrt(reg_file['sig_tot_DM'].loc[(reg_file.index.isin(PoR))].pow(2).sum() / len(PoR))  # maybe not divided by the lenght of the period!
        sigma_cum = reg_file['sig_tot_DM'].pow(2).cumsum().pow(0.5)
        sigma_CUM_DM_Gt = sigma_cum.loc[2023]

        per_obs = nb_gla_obs * 100 / nb_gla_reg
        per_area = tot_area_obs_reg * 100 / tot_area_rgi_reg
        SLE = (-DM_Gt_yr / S_ocean) * 10**6
        Sigma_SLE = (sigma_DM_Gt_yr / S_ocean) * 10**6

        if region in ['SA1', 'SA2']:
            in_regional_series_df = Path(regional_area_directory, 'Zemp_etal_results_region_' + str(rgi_code[region]).lstrip('0') + '_SAN.csv')
        else:
            in_regional_series_df = Path(regional_area_directory, 'Zemp_etal_results_region_' + str(rgi_code[region]).lstrip('0') + '_' + region + '.csv')


        zemp_df = pd.read_csv(in_regional_series_df, encoding='utf-8', delimiter=',', header=26, index_col='Year')
        zemp_2019_df = zemp_df[[' INT_Gt', ' sig_Int_Gt', ' sig_Total_Gt']]
        zemp_2019_df.index.name = 'Year'
        zemp_2019_df.columns = ['MB_Gt', 'sig_Int_Gt', 'MB_sigma_Gt']

        zemp_1976_2016 = zemp_2019_df['MB_Gt'].loc[(zemp_2019_df.index.isin(PoR))].sum() / len(PoR)
        zemp_sig_1976_2016 = np.sqrt(zemp_2019_df['MB_sigma_Gt'].loc[zemp_2019_df.index.isin(PoR)].pow(2).sum() / len(PoR))
        zemp_cum_1976_2016 = zemp_2019_df['MB_Gt'].loc[(zemp_2019_df.index.isin(PoR))].sum()


        for index, row in glob_cum_df.iterrows():
            if index == region:
                row['region']= rgi_region[region]
                row['area_mean_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + ' [km2]'] = "{:.0f}".format(mean_area_full)
                row['area_mean_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [km2]'] = "{:.0f}".format(mean_area)
                row['percentage_area_obs'] = per_area
                row['DM [Gt yr-1]'] = "{:.2f}".format(DM_Gt_yr)
                row['sigma_DM [Gt yr-1]'] = "{:.2f}".format(sigma_DM_Gt_yr)
                row['CUM_DM_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [Gt]'] = "{:.2f}".format(CUM_DM_Gt)
                row['sigma_CUM_DM_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [Gt]'] = "{:.2f}".format(sigma_CUM_DM_Gt)
                row['B [mwe yr-1]'] = "{:.2f}".format(B_mwe_yr)
                row['sigma_B [mwe yr-1]'] = "{:.2f}".format(sigma_B_mwe_yr)
                row['SLE [mm yr-1]'] = "{:.3f}".format(SLE)
                row['sigma_SLE [mm yr-1]'] = "{:.3f}".format(Sigma_SLE)

                row['zemp_DM [Gt yr-1]'] = "{:.2f}".format(zemp_1976_2016)
                row['zemp_sigma_DM [Gt yr-1]'] = "{:.2f}".format(zemp_sig_1976_2016)
                row['zemp_CUM_DM_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [Gt]'] = "{:.2f}".format(zemp_cum_1976_2016)


    #################################################################################################
    ########## Calculate global glacier mass balance and mass loss series ############################

    area_per_region = glob_cum_df['area_mean_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + ' [km2]'].astype(float)
    world_area = glob_cum_df['area_mean_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + ' [km2]'].astype(float).sum()
    rel_area_per_region = area_per_region / world_area
    rel_area_dict = rel_area_per_region.to_dict()

    ba_df0 = pd.DataFrame()
    for region in ba_df.columns:
        ba_df0[region] = ba_df[region] * rel_area_dict[region]


    glob_ba_df = ba_df0.loc[(ba_df0.index >= ini_yr_full_obs)].sum(axis=1).rename('B [m w.e.]')

    sig_ba_df0 = pd.DataFrame()
    for region in sig_ba_df.columns:
        sig_ba_df0[region] = sig_ba_df[region] * rel_area_dict[region]

    glob_sig_ba = sig_ba_df0.loc[(sig_ba_df0.index >= ini_yr_full_obs)].sum(axis=1).rename('sigma B [m w.e.]')
    glob_sig_ba_df = np.sqrt(glob_sig_ba.pow(2))

    glob_DM_df = Reg_DM_df.loc[(Reg_DM_df.index >= ini_yr_full_obs) & (Reg_DM_df.index <= fin_yr_obs)].sum(axis=1).rename('DM [Gt]')
    glob_sig_DM = Reg_sig_DM_df.loc[(Reg_DM_df.index >= ini_yr_full_obs) & (Reg_DM_df.index <= fin_yr_obs)].pow(2).sum(axis=1)
    glob_sig_DM_df = np.sqrt(glob_sig_DM).rename('sigma_DM [Gt]')

    glob_slr_df = (-glob_DM_df/ S_ocean) * 10 ** 6
    glob_slr_df = glob_slr_df.rename('SLE [mm]')
    glob_sig_slr_df = (glob_sig_DM_df / S_ocean) * 10 ** 6
    glob_sig_slr_df = glob_sig_slr_df.rename('sigma_SLE [mm]')

    glob_df = pd.concat([glob_ba_df, glob_sig_ba_df, glob_DM_df, glob_sig_DM_df, glob_slr_df, glob_sig_slr_df ], axis=1)

    Reg_DM_cum_df = Reg_DM_df.loc[Reg_DM_df.index >= 1976]
    Reg_DM_cum_df = Reg_DM_cum_df.cumsum()

    Reg_B_cum_df = ba_df.loc[ba_df.index >= 1976]
    Reg_B_cum_df = Reg_B_cum_df.cumsum()
    Reg_B_cum_df.to_csv(Path(out_dir, 'Cumulative_Regional_Bmwe_series.csv'))

    Reg_DM_df.to_csv(Path(out_dir,'Regional_DM_series.csv'))
    Reg_DM_cum_df.to_csv(Path(out_dir,'Cumulative_Regional_DM_series.csv'))
    Reg_sig_DM_df.to_csv(Path(out_dir,'Regional_DM_series_uncertainty.csv'))

    ########## Calculate global glacier cumulative mass loss ############################

    glob_area_full = glob_cum_df['area_mean_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + ' [km2]'].astype(float).sum()
    glob_area_mean = glob_cum_df['area_mean_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [km2]'].astype(float).sum()
    glob_per_obs = glob_cum_df['percentage_area_obs'].mean()
    glob_DM_yr = glob_cum_df['DM [Gt yr-1]'].astype(float).sum()
    glob_sig_DM_yr = "{:.1f}".format((np.sqrt(glob_cum_df['sigma_DM [Gt yr-1]'].astype(float).pow(2).sum()/20)))
    glob_DM_CUM = glob_cum_df['CUM_DM_'+str(min(PoR))+'_'+str(max(PoR))+' [Gt]'].astype(float).sum()
    glob_SLE_yr = glob_cum_df['SLE [mm yr-1]'].astype(float).sum()
    glob_sig_SLE_yr = "{:.3f}".format((np.sqrt(glob_cum_df['sigma_SLE [mm yr-1]'].astype(float).pow(2).sum())))

    glob_lst = ['GLOBAL', '' , glob_area_full, glob_area_mean , glob_per_obs, glob_DM_yr, glob_sig_DM_yr, glob_DM_CUM, '', '', '', glob_SLE_yr, glob_sig_SLE_yr, '', '', '' ]
    glob_cum_df = glob_cum_df.reset_index()
    glob_cum_df.loc[len(glob_cum_df.index)] = glob_lst
    glob_cum_df = glob_cum_df.rename(columns={'index':'region_code'})

    glob_df['DM_cum [Gt]']= glob_df['DM [Gt]'].cumsum()

    sig = glob_df['sigma_DM [Gt]'].pow(2).cumsum()
    glob_df['sigma_DM_cum [Gt]'] = sig.pow(0.5)

    glob_df['B_cum [m w.e.]']= glob_df['B [m w.e.]'].cumsum()

    glob_df['SLE_cum [Gt]']= glob_df['SLE [mm]'].cumsum()
    sig_sle = glob_df['sigma_SLE [mm]'].pow(2).cumsum()
    glob_df['sigma_SLE_cum [mm]'] = sig_sle.pow(0.5)

    glob_df.to_csv(Path(out_dir,'Global_DM_series_year_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + '.csv'))
    glob_cum_df.to_csv(Path(out_dir,'Cum_DM_Gt_per_region_PoR_'+str(min(PoR))+'_'+str(max(PoR))+'.csv'), index=False)
