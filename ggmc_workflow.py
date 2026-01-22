from typing import Dict, List, Tuple
from pathlib import Path

import ggmc.creation
import ggmc.functions


# ---- Constants ----

INPUT_PATH: Path = Path('data/_input')
OUTPUT_PATH: Path = Path('data/_output')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

DENSITY_FACTOR: Tuple[float, float] = (0.85, 0.06)
"""Density of ice (mean, sigma) relative to water (1000 kg m-3)."""

BEGIN_YEAR: int = 1915
"""Earliest year (determined by longer anomally from CEU starting in 1915)."""

# ---- 1. Glacier change data ----

MASS_BALANCE_FILE: Path = INPUT_PATH / 'fog_bw-bs-ba.csv'

ggmc.functions.format_mass_balance_data(
    input_file=MASS_BALANCE_FILE,
    begin_year=BEGIN_YEAR,
    output_dir=OUTPUT_PATH,
)

# Input
ELEVATION_CHANGE_FILE: Path = INPUT_PATH / 'FOG_ELEVATION_CHANGE_DATA.csv'
GLACIER_SERIES_FILE: Path = INPUT_PATH / 'FOG_GLACIER_SERIES.csv'
INVESTIGATORS_TO_DROP: List[str] = ['Robert McNabb', 'Thorsten Seehaus']

# Output
GLACIER_COORDINATE_FILE: Path = OUTPUT_PATH / 'FOG_coord.csv'
GEODETIC_CHANGE_FILE: Path = OUTPUT_PATH / '_FOG_GEO_MASS_BALANCE_DATA.csv'

ggmc.functions.format_elevation_change(
    elevation_change_file=ELEVATION_CHANGE_FILE,
    glacier_series_file=GLACIER_SERIES_FILE,
    investigators_to_drop=INVESTIGATORS_TO_DROP,
    glacier_coordinate_file=GLACIER_COORDINATE_FILE,
    geodetic_change_file=GEODETIC_CHANGE_FILE,
    density_factor=DENSITY_FACTOR
)

# ---- 2. Kriging spatial anomalies ----

YEAR_INI: int = 2011
YEAR_FIN: int = 2020

# Input
URUMQI_MISSING_YEARS_FILE: Path = INPUT_PATH / 'urumqi_missing_years.csv'

# Output
BA_FILE: Path = OUTPUT_PATH / 'ba.csv'
BA_UNC_FILE: Path = OUTPUT_PATH / 'ba_unc.csv'
MEAN_ANOMALY_DIR: Path = OUTPUT_PATH / 'MEAN_spatial_gla_anom'
LOOKUP_ANOMALY_DIR: Path = OUTPUT_PATH / 'LOOKUP_spatial_and_reg_ids'
LONG_NORM_ANOMALY_DIR: Path = OUTPUT_PATH / 'LONG-NORM_spatial_gla_anom'
REGIONS: List[str] = ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'NZL', 'ANT', 'SA1', 'SA2']

# TODO: Expose region configuration as parameter
ggmc.functions.calculate_global_glacier_spatial_anomaly(
    year_ini=YEAR_INI,
    year_fin=YEAR_FIN,
    begin_year=BEGIN_YEAR,
    mass_balance_file=MASS_BALANCE_FILE,
    ba_file=BA_FILE,
    ba_unc_file=BA_UNC_FILE,
    urumqi_missing_years_file=URUMQI_MISSING_YEARS_FILE,
    glacier_coordinate_file=GLACIER_COORDINATE_FILE,
    geodetic_change_file=GEODETIC_CHANGE_FILE,
    regions=REGIONS,
    mean_anomaly_dir=MEAN_ANOMALY_DIR,
    lookup_anomaly_dir=LOOKUP_ANOMALY_DIR,
    long_norm_anomaly_dir=LONG_NORM_ANOMALY_DIR
)

# ---- 3. Kriging global CE spatial anomaly ----

# TODO: Determine from data
END_YEAR: int = 2025
REGION_OCE_DIR: Path = OUTPUT_PATH / 'OCE_files_by_region'
MIN_YEAR_GEO_OBS: int = 0

# NOTE: Hugonnet 5-year estimates are only dropped if min_length_geo >= 5
MIN_LENGTH_GEO: float = 5.0

ggmc.functions.calculate_consensus_estimate_and_error_global_glacier_regional_anomaly(
    begin_year=BEGIN_YEAR,
    end_year=END_YEAR,
    min_year_geo_obs=MIN_YEAR_GEO_OBS,
    min_length_geo=MIN_LENGTH_GEO,
    long_norm_anomaly_dir=LONG_NORM_ANOMALY_DIR,
    geodetic_change_file=GEODETIC_CHANGE_FILE,
    region_oce_dir=REGION_OCE_DIR,
    regions=REGIONS
)

# ---- 4. Kriging regional mass balance ----

# TODO: Merge with REGIONS and RGI_CODE and move to configuration
RGI_REG: Dict[str, str] = {
  'ACN' : 'ArcticCanadaNorth',
  'WNA' : 'WesternCanadaUS',
  'ALA' : 'Alaska',
  'ACS' : 'ArcticCanadaSouth',
  'TRP' : 'LowLatitudes',
  'SCA' : 'Scandinavia',
  'SJM' : 'Svalbard',
  'CEU' : 'CentralEurope',
  'CAU' : 'CaucasusMiddleEast',
  'ASC' : 'CentralAsia',
  'ASN' : 'NorthAsia',
  'ASE' : 'SouthAsiaEast',
  'NZL' : 'NewZealand',
  'ASW' : 'SouthAsiaWest',
  'GRL' : 'GreenlandPeriphery',
  'ANT' : 'AntarcticSubantarctic',
  'ISL' : 'Iceland',
  'RUA' : 'RussianArctic',
  'SAN' : 'SouthernAndes',
  'SA1' : 'SouthernAndes',
  'SA2' : 'SouthernAndes'
}

RGI_CODE: Dict[str, str] = {
  'ALA' : '01',
  'WNA' : '02',
  'ACN' : '03',
  'ACS' : '04',
  'GRL' : '05',
  'ISL' : '06',
  'SJM' : '07',
  'SCA' : '08',
  'RUA' : '09',
  'ASN' : '10',
  'CEU' : '11',
  'CAU' : '12',
  'ASC' : '13',
  'ASW' : '14',
  'ASE' : '15',
  'TRP' : '16',
  'SA1' : '17',
  'SA2' : '17',
  'NZL' : '18',
  'ANT' : '19'
}

RGI_AREA_FILE: Path = INPUT_PATH / '_RGI_All_ID_Area.csv'
GLIMS_ATTRIBUTE_FILE: Path = INPUT_PATH / 'CAU_glims_attribute_table.csv'
RGI_ATTRIBUTE_DIR: Path = INPUT_PATH / '00_rgi60/00_rgi60_attribs'
REGIONAL_BALANCE_DIR: Path = OUTPUT_PATH / 'regional_balance'

ggmc.functions.calculate_regional_mass_balance(
    region_oce_dir=REGION_OCE_DIR,
    regional_balance_dir=REGIONAL_BALANCE_DIR,
    regions=REGIONS,
    rgi_reg=RGI_REG,
    rgi_code=RGI_CODE,
    begin_year=BEGIN_YEAR,
    rgi_area_file=RGI_AREA_FILE,
    glims_attribute_file=GLIMS_ATTRIBUTE_FILE,
    rgi_attribute_dir=RGI_ATTRIBUTE_DIR
)

ggmc.functions.compile_regional_mass_balance(
    regional_balance_dir=REGIONAL_BALANCE_DIR,
    regions=REGIONS
)

REGIONAL_BALANCE_ESSD_DIR: Path = OUTPUT_PATH / 'regional_balance_essd'
GLACIER_ID_LUT_FILE: Path = INPUT_PATH / 'GLACIER_ID_LUT_links.csv'
GLIMS_ATTRIBUTE_AREA_FILE: Path = INPUT_PATH / 'glims_CAU_attributes.csv'

ggmc.functions.calculate_regional_mass_balance_essd(
    regional_balance_dir=REGIONAL_BALANCE_DIR,
    rgi_code=RGI_CODE,
    rgi_region=RGI_REG,
    glacier_id_lut_file=GLACIER_ID_LUT_FILE,
    glims_attribute_file=GLIMS_ATTRIBUTE_AREA_FILE,
    rgi_attribute_dir=RGI_ATTRIBUTE_DIR,
    regional_balance_essd_dir=REGIONAL_BALANCE_ESSD_DIR,
    regions=REGIONS,
    runs=['cal_series', 'error_dh', 'error_anom', 'error_rho', 'error_tot']
)

# ---- 5. Kriging regional mass loss ----

# Period to calculate the cumulative mass loss
INI_YR: int = 1976
FIN_YR: int = 2025

ZEMP_REGIONAL_SERIES_DIR: Path = INPUT_PATH / 'zemp_etal_regional_series'
REGIONAL_AREA_CHANGE_FILE: Path = INPUT_PATH / 'Regional_area_change_Zemp_for_spt_CEs.csv'
MASS_LOSS_DIR: Path = OUTPUT_PATH / 'mass_loss'

ggmc.functions.calculate_regional_mass_loss(
    regional_balance_dir=REGIONAL_BALANCE_DIR,
    region_oce_dir=REGION_OCE_DIR,
    regional_area_change_file=REGIONAL_AREA_CHANGE_FILE,
    rgi_area_file=RGI_AREA_FILE,
    zemp_regional_series_dir=ZEMP_REGIONAL_SERIES_DIR,
    ini_yr=INI_YR,
    fin_yr=FIN_YR,
    regions=REGIONS,
    rgi_code=RGI_CODE,
    rgi_reg=RGI_REG,
    mass_loss_dir=MASS_LOSS_DIR
)

#####################################
# ---- Part 2: Creation workflow ----

# ---- 0. Grid tiles per region ----

RGI_REGION: Dict[str, str] = {
    'ACN' : 'ArcticCanadaNorth',
    'WNA' : 'WesternCanadaUS',
    'ALA' : 'Alaska',
    'ACS' : 'ArcticCanadaSouth',
    'TRP' : 'LowLatitudes',
    'SCA' : 'Scandinavia',
    'SJM' : 'Svalbard',
    'CEU' : 'CentralEurope',
    'CAU' : 'CaucasusMiddleEast',
    'ASC' : 'CentralAsia',
    'ASN' : 'NorthAsia',
    'ASE' : 'SouthAsiaEast',
    'NZL' : 'NewZealand',
    'ASW' : 'SouthAsiaWest',
    'GRL' : 'GreenlandPeriphery',
    'ANT' : 'AntarcticSubantarctic',
    'ISL' : 'Iceland',
    'RUA' : 'RussianArctic',
    'SAN' : 'SouthernAndes'
}

RGI_CODE: Dict[str, str] = {
    'ALA' : '01',
    'WNA' : '02',
    'ACN' : '03',
    'ACS' : '04',
    'GRL' : '05',
    'ISL' : '06',
    'SJM' : '07',
    'SCA' : '08',
    'RUA' : '09',
    'ASN' : '10',
    'CEU' : '11',
    'CAU' : '12',
    'ASC' : '13',
    'ASW' : '14',
    'ASE' : '15',
    'TRP' : '16',
    'SAN' : '17',
    'NZL' : '18',
    'ANT' : '19'
}

# TODO: Why is SAN used here instead of SA1 and SA2?
REGIONS_SAN: List[str] = ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'SAN', 'NZL', 'ANT']

REGIONAL_TILE_DIR: Path = OUTPUT_PATH / 'Tiles_by_region_0.5'

ggmc.creation.grid_tiles_per_region(
    rgi_region=RGI_REGION,
    rgi_code=RGI_CODE,
    regions=REGIONS_SAN,
    rgi_attribute_dir=RGI_ATTRIBUTE_DIR,
    glims_attribute_area_file=GLIMS_ATTRIBUTE_AREA_FILE,
    regional_tile_dir=REGIONAL_TILE_DIR
)


# ---- 2. OCE to tiles 0.5 grid per region ----

YMIN: int = 1976
YMAX: int = 2025

OCE_TILE_DIR: Path = OUTPUT_PATH / 'OCE_tiles_by_region_0.5'

ggmc.creation.oce2tiles_05_grid_per_region(
    regions=REGIONS,
    ymin=YMIN,
    ymax=YMAX,
    regional_tile_dir=REGIONAL_TILE_DIR,
    oce_dir=REGIONAL_BALANCE_ESSD_DIR,
    oce_tile_dir=OCE_TILE_DIR
)

# ---- 3. Meters water equivalent to gigatonnes and area change 0.5 grid per region ----

# NOTE: Why is the Regional_area_change_Zemp_for_spt_CEs.csv not used here?
AREA_REF_YEAR: Dict[str, int] = {
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
    'WNA': 2006
}

AREA_CHG_RATE: Dict[str, float] = {
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
    'WNA': -0.54
}

AREA_CHANGE_GRID_DIR: Path = OUTPUT_PATH / 'area_change_by_region_0.5'
MASS_CHANGE_GRID_DIR: Path = OUTPUT_PATH / 'mass_change_by_region_0.5'

ggmc.creation.areachange_grid_per_region(
    regions=REGIONS,
    area_ref_year=AREA_REF_YEAR,
    area_chg_rate=AREA_CHG_RATE,
    ymin=YMIN,
    ymax=YMAX,
    regional_tile_dir=REGIONAL_TILE_DIR,
    oce_tile_dir=OCE_TILE_DIR,
    area_change_grid_dir=AREA_CHANGE_GRID_DIR,
    mass_change_grid_dir=MASS_CHANGE_GRID_DIR
)

# ---- Tiles to global grid 0.5 ----

# Output
GLOBAL_GRID_DIR: Path = OUTPUT_PATH / 'global_grid_0.5'

ggmc.creation.tiles_to_global_grid(
    ymin=YMIN,
    ymax=YMAX,
    mass_change_grid_dir=MASS_CHANGE_GRID_DIR,
    oce_tile_dir=OCE_TILE_DIR,
    area_change_grid_dir=AREA_CHANGE_GRID_DIR,
    global_grid_dir=GLOBAL_GRID_DIR
)

# ---- CSV to NetCDF4 global grid 0.5 ----

# Output
GLOBAL_GRID_NETCDF_DIR: Path = OUTPUT_PATH / 'global_grid_netcdf_0.5'

ggmc.creation.csv2netcdf4_globalGrid(
    ymin=YMIN,
    ymax=YMAX,
    global_grid_dir=GLOBAL_GRID_DIR,
    global_grid_netcdf_dir=GLOBAL_GRID_NETCDF_DIR
)
