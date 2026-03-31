from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd


VERSION: str = '2026-02-10'


# ---- Inputs ----

INPUT_PATH: Path = Path('data/input')

MASS_BALANCE_FILE: Path = INPUT_PATH / 'fog_bw-bs-ba.csv'
ELEVATION_CHANGE_FILE: Path = INPUT_PATH / 'FOG_ELEVATION_CHANGE_DATA.csv'
GLACIER_SERIES_FILE: Path = INPUT_PATH / 'FOG_GLACIER_SERIES.csv'
REGIONAL_AREA_CHANGE_RATE_FILE: Path = INPUT_PATH / 'regional_area_change.csv'
URUMQI_MISSING_YEARS_FILE: Path = INPUT_PATH / 'urumqi_missing_years.csv'
RGI_AREA_FILE: Path = INPUT_PATH / '_RGI_All_ID_Area.csv'
GLIMS_ATTRIBUTE_FILE: Path = INPUT_PATH / 'CAU_glims_attribute_table.csv'
RGI_ATTRIBUTE_DIR: Path = INPUT_PATH / '00_rgi60/00_rgi60_attribs'
GLACIER_ID_LUT_FILE: Path = INPUT_PATH / 'GLACIER_ID_LUT_links.csv'
GLIMS_ATTRIBUTE_AREA_FILE: Path = INPUT_PATH / 'glims_CAU_attributes.csv'
ZEMP_REGIONAL_SERIES_DIR: Path = INPUT_PATH / 'zemp_etal_regional_series'


# ---- Outputs ----

OUTPUT_PATH: Path = Path('data/output')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

GLACIER_COORDINATE_FILE: Path = OUTPUT_PATH / 'FOG_coord.csv'
GEODETIC_CHANGE_FILE: Path = OUTPUT_PATH / '_FOG_GEO_MASS_BALANCE_DATA.csv'
REGIONAL_AREA_FILE: Path = OUTPUT_PATH / 'regional_area.csv'
BA_FILE: Path = OUTPUT_PATH / 'ba.csv'
BA_UNC_FILE: Path = OUTPUT_PATH / 'ba_unc.csv'
MEAN_ANOMALY_DIR: Path = OUTPUT_PATH / 'MEAN_spatial_gla_anom'
LOOKUP_ANOMALY_DIR: Path = OUTPUT_PATH / 'LOOKUP_spatial_and_reg_ids'
LONG_NORM_ANOMALY_DIR: Path = OUTPUT_PATH / 'LONG-NORM_spatial_gla_anom'
REGION_OCE_DIR: Path = OUTPUT_PATH / 'OCE_files_by_region'
REGIONAL_BALANCE_DIR: Path = OUTPUT_PATH / 'regional_balance'
REGIONAL_BALANCE_ESSD_DIR: Path = OUTPUT_PATH / 'regional_balance_essd'
MASS_LOSS_DIR: Path = OUTPUT_PATH / 'mass_loss'
REGIONAL_TILE_DIR: Path = OUTPUT_PATH / 'Tiles_by_region_0.5'
OCE_TILE_DIR: Path = OUTPUT_PATH / 'OCE_tiles_by_region_0.5'
AREA_CHANGE_GRID_DIR: Path = OUTPUT_PATH / 'area_change_by_region_0.5'
MASS_CHANGE_GRID_DIR: Path = OUTPUT_PATH / 'mass_change_by_region_0.5'
GLOBAL_GRID_DIR: Path = OUTPUT_PATH / 'global_grid_0.5'
GLOBAL_GRID_NETCDF_DIR: Path = OUTPUT_PATH / 'global_grid_netcdf_0.5'


# ---- Parameters ----

INVESTIGATORS_TO_DROP: List[str] = ['Robert McNabb', 'Thorsten Seehaus']

DENSITY_FACTOR: Tuple[float, float] = (0.85, 0.06)
"""Density of ice (mean, sigma) relative to water (1000 kg m-3)."""

BEGIN_YEAR: int = 1915
"""Earliest year (determined by longer anomally from CEU starting in 1915)."""

# TODO: Determine from data
END_YEAR: int = 2025

YEAR_INI: int = 2011
YEAR_FIN: int = 2020

REGIONS: List[str] = ['ALA', 'WNA', 'ACN', 'ACS', 'GRL', 'ISL', 'SJM', 'SCA', 'RUA', 'ASN', 'CEU', 'CAU', 'ASC', 'ASW', 'ASE', 'TRP', 'NZL', 'ANT', 'SA1', 'SA2']

MIN_YEAR_GEO_OBS: int = 0

# NOTE: Hugonnet 5-year estimates are only dropped if min_length_geo >= 5
MIN_LENGTH_GEO: float = 5.0

# TODO: Merge with REGIONS and RGI_CODE
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

# Period to calculate the cumulative mass loss
INI_YR: int = 1976

# TODO: Merge with END_YEAR
FIN_YR: int = 2025

RGI_REGION_SAN: Dict[str, str] = {
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

RGI_CODE_SAN: Dict[str, str] = {
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

# TODO: Merge with INI_YR and FIN_YR
YMIN: int = 1976
YMAX: int = 2025

AREA_REF_YEAR: Dict[str, int] = (
  pd.read_csv(REGIONAL_AREA_CHANGE_RATE_FILE)
  .set_index('region_id')['reference_year']
  .to_dict()
)

AREA_CHG_RATE: Dict[str, float] = (
  pd.read_csv(REGIONAL_AREA_CHANGE_RATE_FILE)
  .set_index('region_id')['change_rate_percentyear']
  .to_dict()
)
