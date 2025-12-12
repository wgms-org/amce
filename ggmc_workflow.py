# %%
from ggmc.functions import *
from ggmc.kriging import *


# %% [markdown]
# ### 1 Glacier Change Data
#

# %%
# Global variables for first step

# Input the fog version
FOG_VERSION = '2025-01'

# Input the path (as a string) to the WGMS Fog input data (relative path only)
WGMS_INPUT = './data/input/1_glacier_change_data/fog_bw-bs-ba_2025-01.csv'

# Input the path (as a string) of a directory where you'd like the outputs to be delivered
OUTPUT_DATA_PATH = './data/output/1_glacier_change_data/'



# %%
# Run the first data prep function for the first step
data_prep_spt_anom(FOG_VERSION,
                   WGMS_INPUT,
                   OUTPUT_DATA_PATH)



# %%
# Input files for the elevation change function (as relative paths)
ELEV_CHANGE_INPUT = './data/input/1_glacier_change_data/FOG_ELEVATION_CHANGE_DATA_2025-01.csv'
GLAC_SERIES_INPUT = './data/input/1_glacier_change_data/FOG_GLACIER_SERIES_2025-01.csv'

# Input a list of providers to clean with the function
PROVIDERS_TO_DROP = ['Robert McNabb','Thorsten Seehaus']



# %%
# Run the elevation change data prep function
data_prep_elevation_change(FOG_VERSION,
                           ELEV_CHANGE_INPUT,
                           GLAC_SERIES_INPUT,
                           PROVIDERS_TO_DROP,
                           OUTPUT_DATA_PATH)



# %% [markdown]
# ### 2 Kriging spatial anomalies

# %%
# Specify the inputs to the functions

YEAR_INI = 2011
YEAR_FIN = 2020
MAX_GLAC_ANOM = 5 # maximum number of closer individual glacier anomalies used to calculate the glacier temporal variability
MIN_GLAC_ANOM = 3 # minimum number of closer individual glacier anomalies to calculate the glacier temporal variability, if less anomalies are available, regional anomaly is used
D_THRESH_LST = [60, 120, 250, 500, 1000] # search distances (km) for finding close mass balance anomalies
MAX_D = 1000 # maximum distance (km) allowed for finding close mass balance anomalies, if no anomalies are found, regional anomaly is used
FOG_VERSION = '2025-01'

# Input/Output data paths
IN_DATA_GLA_PATH = './data/input/2_Kriging_spatial_anomalies/fog_bw-bs-ba_2025-01.csv'
BA_FILE_PATH = './data/input/2_Kriging_spatial_anomalies/fog_2025-01_ba.csv'
BA_UNC_FILE_PATH = './data/input/2_Kriging_spatial_anomalies/fog_2025-01_ba_unc.csv'
MISSING_YEARS_PATH = './data/input/2_Kriging_spatial_anomalies/urumqi_missing_years.csv'
IN_GLA_COORD_PATH = './data/input/2_Kriging_spatial_anomalies/FOG_coord_2025-01.csv'
FOG_GMB_PATH = './data/input/2_Kriging_spatial_anomalies/_FOG_GEO_MASS_BALANCE_DATA_2025-01.csv'
OUTPUT_DATA_PATH = './data/output/2_Kriging_spatial_anomalies/'



# %%
# Run the function
calc_global_gla_spatial_anom(YEAR_INI,
                             YEAR_FIN,
                             MAX_GLAC_ANOM,
                             MIN_GLAC_ANOM,
                             D_THRESH_LST,
                             MAX_D,
                             FOG_VERSION,
                             IN_DATA_GLA_PATH,
                             BA_FILE_PATH,
                             BA_UNC_FILE_PATH,
                             MISSING_YEARS_PATH,
                             IN_GLA_COORD_PATH,
                             FOG_GMB_PATH,
                             OUTPUT_DATA_PATH)



# %% [markdown]
# ### 3 Kriging global CE spatial anomaly
#

# %%
# Specify the inputs to the functions
fin_hydro_yr= {'ALA' : 0.75,'WNA' : 0.75,'ACN' : 0.75,'ACS' : 0.75,'GRL' : 0.75,
           'ISL' : 0.75,'SJM' : 0.75,'SCA' : 0.75,'RUA' : 0.75,'ASN' : 0.75,
           'CEU' : 0.75,'CAU' : 0.75,'ASC' : 0.75,'ASW' : 0.75,'ASE' : 0.75,
           'TRP' : 0,'SA1' : 0.25,'SA2' : 0.25,'NZL' : 0.25,'ANT' : 0.25}

ini_hydro_yr= {'ALA' : 0.25,'WNA' : 0.25,'ACN' : 0.75,'ACS' : 0.75,'GRL' : 0.75,
           'ISL' : 0.25,'SJM' : 0.25,'SCA' : 0.25,'RUA' : 0.25,'ASN' : 0.25,
           'CEU' : 0.25,'CAU' : 0.25,'ASC' : 0.25,'ASW' : 0.25,'ASE' : 0.25,
           'TRP' : 0,'SA1' : 0.75,'SA2' : 0.75,'NZL' : 0.75,'ANT' : 0.75}



# %%
# Specify the inputs to the functions
FOG_VERSION = '2025-01'
YR_INI = 1915 # define begining year of anomaly files, determined by longer anomally from CEU starting in 1915
YR_END = 2024 # define the end year with data, determied by the last call for data in WGMS

# Input/Output data paths
PATH_SPT_ANOM = "./data/output/2_Kriging_spatial_anomalies/fog-2025-01/LONG-NORM_spatial_gla_anom_ref_2011-2020"
IN_DATA_GEO = "./data/input/3_Kriging_global_CE_spatial_anomaly/_FOG_GEO_MASS_BALANCE_DATA_2025-01.csv"
OUTPUT_DATA_PATH_STRING = "./data/output/3_Kriging_global_CE_spatial_anomaly"
REG_LST = ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'SA1', 'SA2', 'NZL', 'ANT']



# %%
calc_OCE_and_error_global_gla_reg_anom(FOG_VERSION,
                                       YR_INI,
                                       YR_END,
                                       PATH_SPT_ANOM,
                                       IN_DATA_GEO,
                                       OUTPUT_DATA_PATH_STRING,
                                       REG_LST)



# %%


# %% [markdown]
# ### 4 Kriging regional mass balance

# %%
# Input region information
REG_LST = ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'SA1', 'SA2', 'NZL', 'ANT']

RGI_REG = {'ACN' : 'ArcticCanadaNorth', 'WNA' : 'WesternCanadaUS', 'ALA' : 'Alaska', 'ACS' : 'ArcticCanadaSouth', 'TRP' : 'LowLatitudes', 'SCA' : 'Scandinavia',
             'SJM' : 'Svalbard', 'CEU' : 'CentralEurope', 'CAU' : 'CaucasusMiddleEast', 'ASC' : 'CentralAsia', 'ASN' : 'NorthAsia', 'ASE' : 'SouthAsiaEast',
             'NZL' : 'NewZealand', 'ASW' : 'SouthAsiaWest', 'GRL' : 'GreenlandPeriphery', 'ANT' : 'AntarcticSubantarctic', 'ISL' : 'Iceland', 'RUA' : 'RussianArctic',
             'SAN' : 'SouthernAndes', 'SA1' : 'SouthernAndes', 'SA2' : 'SouthernAndes'}

RGI_CODE = {'ALA' : '01', 'WNA' : '02', 'ACN' : '03', 'ACS' : '04', 'GRL' : '05', 'ISL' : '06', 'SJM' : '07', 'SCA' : '08', 'RUA' : '09', 'ASN' : '10',
           'CEU' : '11', 'CAU' : '12', 'ASC' : '13', 'ASW' : '14', 'ASE' : '15', 'TRP' : '16', 'SA1' : '17', 'SA2' : '17', 'NZL' : '18', 'ANT' : '19'}



# %%
# Input FOG version and path specs
FOG_VERSION = '2025-01'

DM_SERIES_MIN_YR = 1915

PATH_OCE = "data/input/4_Kriging_regional_mass_balance/OCE_files_by_region" #path to regional OCE files

OUTPUT_DATA_PATH_STRING = "./data/output/4_Kriging_regional_mass_balance"

INPUT_B_AND_SIGMA_PATH = "./data/input/4_Kriging_regional_mass_balance/b_and_sigma"

IN_DATA_RGI_AREA_PATH = "./data/input/4_Kriging_regional_mass_balance/_RGI_All_ID_Area.csv"

IN_DATA_GLIMS = "./data/input/4_Kriging_regional_mass_balance/CAU_glims_attribute_table.csv"

IN_DATA_ATTRIBS = "./data/input/4_Kriging_regional_mass_balance/00_rgi60/00_rgi60_attribs"

IN_DATA_REGIONAL_SERIES = "./data/input/4_Kriging_regional_mass_balance/zemp_etal_regional_series"



# %%
calc_reg_ba(FOG_VERSION,
            PATH_OCE,
            OUTPUT_DATA_PATH_STRING,
            REG_LST,
            RGI_REG,
            RGI_CODE,
            INPUT_B_AND_SIGMA_PATH,
            DM_SERIES_MIN_YR,
            IN_DATA_RGI_AREA_PATH,
            IN_DATA_GLIMS,
            IN_DATA_ATTRIBS,
            IN_DATA_REGIONAL_SERIES)



# %% [markdown]
# ### 5 Kriging regional mass loss
#

# %%
# period to calculate the cumulative mass loss
INI_YR = 1976
FIN_YR = 2024

REG_LST = ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'SA1', 'SA2', 'NZL', 'ANT']

RGI_CODE = {'ALA' : '01', 'WNA' : '02', 'ACN' : '03', 'ACS' : '04', 'GRL' : '05', 'ISL' : '06', 'SJM' : '07', 'SCA' : '08', 'RUA' : '09', 'ASN' : '10',
           'CEU' : '11', 'CAU' : '12', 'ASC' : '13', 'ASW' : '14', 'ASE' : '15', 'TRP' : '16', 'SA1' : '17', 'SA2' : '17', 'NZL' : '18', 'ANT' : '19'}

RGI_REG = {'ACN' : 'ArcticCanadaNorth', 'WNA' : 'WesternCanadaUS', 'ALA' : 'Alaska', 'ACS' : 'ArcticCanadaSouth', 'TRP' : 'LowLatitudes', 'SCA' : 'Scandinavia',
             'SJM' : 'Svalbard', 'CEU' : 'CentralEurope', 'CAU' : 'CaucasusMiddleEast', 'ASC' : 'CentralAsia', 'ASN' : 'NorthAsia', 'ASE' : 'SouthAsiaEast',
             'NZL' : 'NewZealand', 'ASW' : 'SouthAsiaWest', 'GRL' : 'GreenlandPeriphery', 'ANT' : 'AntarcticSubantarctic', 'ISL' : 'Iceland', 'RUA' : 'RussianArctic',
             'SAN' : 'SouthernAndes', 'SA1' : 'SouthernAndes', 'SA2' : 'SouthernAndes'}



# %%
# Input FOG version and path specs
FOG_VERSION = '2025-01'

IN_AREAWEIGHTED_FILE = "./data/input/5_Kriging_regional_mass_loss/Regional_B_series_AreaWeighted.csv"

IN_UNCERTAINTY_FILE = "./data/input/5_Kriging_regional_mass_loss/Regional_B_series_uncertainty.csv"

PATH_OCE = "./data/output/3_Kriging_global_CE_spatial_anomaly/out_data_2025-01/OCE_files_by_region"

REGIONAL_AREA_CHANGE_FILE = "./data/input/5_Kriging_regional_mass_loss/Regional_area_change_Zemp_for_spt_CEs.csv"

IN_DATA_AREA = "./data/input/5_Kriging_regional_mass_loss/_RGI_All_ID_Area.csv"

REGIONAL_SERIES_DIRECTORY = "./data/input/5_Kriging_regional_mass_loss/zemp_etal_regional_series"

OUTPUT_DATA_PATH_STRING = "./data/output/5_Kriging_regional_mass_loss"



# %%
calc_reg_mass_loss(FOG_VERSION,
                   IN_AREAWEIGHTED_FILE,
                   IN_UNCERTAINTY_FILE,
                   PATH_OCE,
                   REGIONAL_AREA_CHANGE_FILE,
                   IN_DATA_AREA,
                   REGIONAL_SERIES_DIRECTORY,
                   INI_YR,
                   FIN_YR,
                   REG_LST,
                   RGI_CODE,
                   RGI_REG,
                   OUTPUT_DATA_PATH_STRING)



# %% [markdown]
# ## Section 2
#

# %%
from ggmc.creation import *


# %% [markdown]
# ### 0_v1.6_grid_tiles_per_region.py
#

# %%
# Function inputs
FOG_VERSION = '2025-01'

RGI_REGION = {'ACN' : 'ArcticCanadaNorth', 'WNA' : 'WesternCanadaUS', 'ALA' : 'Alaska', 'ACS' : 'ArcticCanadaSouth', 'TRP' : 'LowLatitudes', 'SCA' : 'Scandinavia',
             'SJM' : 'Svalbard', 'CEU' : 'CentralEurope', 'CAU' : 'CaucasusMiddleEast', 'ASC' : 'CentralAsia', 'ASN' : 'NorthAsia', 'ASE' : 'SouthAsiaEast',
             'NZL' : 'NewZealand', 'ASW' : 'SouthAsiaWest', 'GRL' : 'GreenlandPeriphery', 'ANT' : 'AntarcticSubantarctic', 'ISL' : 'Iceland', 'RUA' : 'RussianArctic',
             'SAN' : 'SouthernAndes'}

RGI_CODE = {'ALA' : '01', 'WNA' : '02', 'ACN' : '03', 'ACS' : '04', 'GRL' : '05', 'ISL' : '06', 'SJM' : '07', 'SCA' : '08', 'RUA' : '09', 'ASN' : '10',
           'CEU' : '11', 'CAU' : '12', 'ASC' : '13', 'ASW' : '14', 'ASE' : '15', 'TRP' : '16', 'SAN' : '17', 'NZL' : '18', 'ANT' : '19'}

REG_LST = ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'SAN', 'NZL', 'ANT']

GRID_RESOLUTION = '0.5'

INI_YR_OBS = 1976 # initial year of series
END_YR_OBS = 2024 # year of last observation

# Paths
RGI_PATH = "./data/input/Creation_workflow/00_rgi60_attribs"
GLIMS_PATH = "./data/input/Creation_workflow/glims_CAU_attributes.csv"
OUTPUT_DATA_PATH_STRING = "./data/output/Creation_workflow"



# %%
grid_tiles_per_region(FOG_VERSION,
                      RGI_REGION,
                      RGI_CODE,
                      REG_LST,
                      INI_YR_OBS,
                      END_YR_OBS,
                      RGI_PATH, # !! pointing to the original `00_rgi60_attribs` subdirectory, specifically
                      GLIMS_PATH,
                      OUTPUT_DATA_PATH_STRING)



# %% [markdown]
# ### 0_v1.6_oce2tiles_0.5_grid_per_region.py
#

# %%
# Inputs
RGI_CODE = {'ALA' : '01', 'WNA' : '02', 'ACN' : '03', 'ACS' : '04', 'GRL' : '05', 'ISL' : '06', 'SJM' : '07', 'SCA' : '08', 'RUA' : '09', 'ASN' : '10',
           'CEU' : '11', 'CAU' : '12', 'ASC' : '13', 'ASW' : '14', 'ASE' : '15', 'TRP' : '16', 'SA1' : '17', 'SA2' : '17', 'NZL' : '18', 'ANT' : '19',
           'HMA' : '20', 'ALA-WNA' : '21'}


# REG_LST= ['ALA-WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'HMA', 'TRP', 'SA1', 'SA2', 'NZL', 'ANT']
# REG_LST = ['ALA-WNA', 'HMA']
REG_LST = ['ACN']

YMIN = 1976
YMAX = 2024
FOG_VERSION = '2025-01'

# Paths
GRID_PATH = "./data/output/Creation_workflow/Tiles_by_region_0.5"
OCE_PATH = "./data/input/Creation_workflow/out_data_2025-01_Dussaillant_etal_format"
OUTPUT_DATA_PATH_STRING = "./data/output/Creation_workflow"



# %%
oce2tiles_05_grid_per_region(FOG_VERSION,
                             RGI_CODE,
                             REG_LST,
                             YMIN,
                             YMAX,
                             GRID_PATH,
                             OCE_PATH,
                             OUTPUT_DATA_PATH_STRING)



# %% [markdown]
# ### 1_v1.5_mwe2Gt_AreaChange_0.5_grid_per_region

# %%
# Inputs
RGI_CODE = {'ALA' : '01', 'WNA' : '02', 'ACN' : '03', 'ACS' : '04', 'GRL' : '05', 'ISL' : '06', 'SJM' : '07', 'SCA' : '08', 'RUA' : '09', 'ASN' : '10',
           'CEU' : '11', 'CAU' : '12', 'ASC' : '13', 'ASW' : '14', 'ASE' : '15', 'TRP' : '16', 'SA1' : '17', 'SA2' : '17', 'NZL' : '18', 'ANT' : '19'}

REG_LST= ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'SA1', 'SA2', 'NZL', 'ANT']

AREA_REF_YEAR = {
        'ACN': 2000,
        'ACS': 2000,
        'ALA': 2009,
        'ANT': 1989,
        'ASC': 2003,
        'ASE': 2003,
        'ASN': 2011,
        'ASW': 2003,
        'CAU': 2014,
        'CEU': 2003,
        'GRL': 2001,
        'ISL': 2000,
        'NZL': 1978,
        'RUA': 2006,
        'SA1': 2000,
        'SA2': 2000,
        'SCA': 2002,
        'SJM': 2001,
        'TRP': 2000,
        'WNA': 2006}

AREA_CHG_RATE = {
        'ACN': -0.07,
        'ACS': -0.08,
        'ALA': -0.48,
        'ANT': -0.27,
        'ASC': -0.18,
        'ASE': -0.47,
        'ASN': -0.43,
        'ASW': -0.36,
        'CAU': -0.53,
        'CEU': -0.93,
        'GRL': -0.82,
        'ISL': -0.36,
        'NZL': -0.69,
        'RUA': -0.08,
        'SA1': -0.18,
        'SA2': -0.18,
        'SCA': -0.27,
        'SJM': -0.26,
        'TRP': -1.19,
        'WNA': -0.54}

YMIN = 1976
YMAX = 2024
FOG_VERSION = '2025-01'

# Paths
GRID_PATH = "./data/input/Creation_workflow/Tiles_by_region_0.5"
MASS_BALANCE_MWE_PATH = "./data/input/Creation_workflow/MB_mwe_gridded_0.5_by_region" # path + '\\out_data_fog_'+fog_version+'\\0_Output_Regional_MB_mwe_by_gridpoint_fog_'+fog_version+'\\'
MASS_BALANCE_SIGMA_PATH = "./data/input/Creation_workflow/MB_sigma_mwe_gridded_0.5_by_region/" # path_mb + 'MB_mwe_gridded_0.5_by_region\\' # Specific mb files
OUTPUT_DATA_PATH_STRING = "./data/output/Creation_workflow"



# %%
areachange_grid_per_region(FOG_VERSION,
                           RGI_CODE,
                           REG_LST,
                           AREA_REF_YEAR,
                           AREA_CHG_RATE,
                           YMIN,
                           YMAX,
                           GRID_PATH,
                           MASS_BALANCE_MWE_PATH,
                           MASS_BALANCE_SIGMA_PATH,
                           OUTPUT_DATA_PATH_STRING)



# %% [markdown]
# ### 2_v1.5_tiles2globalGridd.py

# %%
# Inputs
FOG_VERSION = '2025-01'
YMIN = 1976
YMAX = 2024

# Paths
TOTAL_MASS_LOSS_MWE_PATH = "./data/input/Creation_workflow/dM_Gt_gridded_0.5_by_region"
TOTAL_MASS_LOSS_SIGMA_PATH = "./data/input/Creation_workflow/dM_sigma_Gt_gridded_0.5_by_region.5"
SPECIFIC_MASS_LOSS_MWE_PATH = "./data/input/Creation_workflow/MB_mwe_gridded_0.5_by_region"
SPECIFIC_MASS_LOSS_SIGMA_PATH = "./data/input/Creation_workflow/MB_sigma_mwe_gridded_0.5_by_region"
GRIDDED_AREA_CHANGE_FILES_PATH = "./data/input/Creation_workflow/area_changes_gridded_0.5_by_region"
OUTPUT_DATA_PATH_STRING = "./data/output/Creation_workflow"



# %%
tiles_to_global_grid(FOG_VERSION,
                     YMIN,
                     YMAX,
                     TOTAL_MASS_LOSS_MWE_PATH,
                     TOTAL_MASS_LOSS_SIGMA_PATH,
                     SPECIFIC_MASS_LOSS_MWE_PATH,
                     SPECIFIC_MASS_LOSS_SIGMA_PATH,
                     GRIDDED_AREA_CHANGE_FILES_PATH,
                     OUTPUT_DATA_PATH_STRING)



# %% [markdown]
# ### 3_v1.5_csv2netcdf4_globalGrid_0.5.py

# %%
# Inputs
FOG_VERSION = '2025-01'
YMIN = 1976
# !! 2024 was changed to 2023 because the input files did not contain data for year 2024
YMAX = 2023

# Paths
TOTAL_MASS_LOSS_MWE_PATH = "./data/input/Creation_workflow/dM_Gt_globalGrid_0.5"
TOTAL_MASS_LOSS_SIGMA_PATH = "./data/input/Creation_workflow/sigma_dM_Gt_globalGrid_0.5"
SPECIFIC_MASS_LOSS_MWE_PATH = "./data/input/Creation_workflow/MB_mwe_globalGrid_0.5"
SPECIFIC_MASS_LOSS_SIGMA_PATH = "./data/input/Creation_workflow/sigma_MB_mwe_globalGrid_0.5"
GRIDDED_AREA_CHANGE_FILES_PATH = "./data/input/Creation_workflow/Area_km2_globalGrid_0.5"
OUTPUT_DATA_PATH_STRING = "./data/output/Creation_workflow/NetCDFs"



# %%
csv2netcdf4_globalGrid(FOG_VERSION,
                       YMIN,
                       YMAX,
                       TOTAL_MASS_LOSS_MWE_PATH,
                       TOTAL_MASS_LOSS_SIGMA_PATH,
                       SPECIFIC_MASS_LOSS_MWE_PATH,
                       SPECIFIC_MASS_LOSS_SIGMA_PATH,
                       GRIDDED_AREA_CHANGE_FILES_PATH,
                       OUTPUT_DATA_PATH_STRING)
