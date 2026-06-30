import amce.creation
import amce.functions
import amce.publish

from constants import *


# ---- 1. Glacier change data ----

amce.functions.format_mass_balance_data(
    input_file=MASS_BALANCE_FILE,
    begin_year=BEGIN_YEAR,
    output_dir=OUTPUT_PATH,
)

amce.functions.format_elevation_change(
    elevation_change_file=ELEVATION_CHANGE_FILE,
    glacier_series_file=GLACIER_SERIES_FILE,
    investigators_to_drop=INVESTIGATORS_TO_DROP,
    glacier_coordinate_file=GLACIER_COORDINATE_FILE,
    geodetic_change_file=GEODETIC_CHANGE_FILE,
    density_factor=DENSITY_FACTOR
)

amce.functions.format_regional_area(
    regional_area_change_rate_file=REGIONAL_AREA_CHANGE_RATE_FILE,
    begin_year=BEGIN_YEAR,
    end_year=END_YEAR,
    regional_area_file=REGIONAL_AREA_FILE
)

# ---- 2. Kriging spatial anomalies ----

# TODO: Expose region configuration as parameter
amce.functions.calculate_global_glacier_spatial_anomaly(
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

amce.functions.calculate_consensus_estimate_and_error_global_glacier_regional_anomaly(
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

amce.functions.calculate_regional_mass_balance(
    region_oce_dir=REGION_OCE_DIR,
    regional_balance_dir=REGIONAL_BALANCE_DIR,
    regions=REGIONS,
    rgi_region=RGI_REGION,
    rgi_code=RGI_CODE,
    rgi_area_file=RGI_AREA_FILE,
    glims_attribute_file=GLIMS_ATTRIBUTE_FILE,
    rgi_attribute_dir=RGI_ATTRIBUTE_DIR
)

amce.functions.compile_regional_mass_balance(
    regional_balance_dir=REGIONAL_BALANCE_DIR,
    regions=REGIONS
)

amce.functions.calculate_regional_mass_balance_essd(
    regional_balance_dir=REGIONAL_BALANCE_DIR,
    rgi_code=RGI_CODE,
    rgi_region=RGI_REGION,
    glacier_id_lut_file=GLACIER_ID_LUT_FILE,
    glims_attribute_file=GLIMS_ATTRIBUTE_AREA_FILE,
    rgi_attribute_dir=RGI_ATTRIBUTE_DIR,
    regional_balance_essd_dir=REGIONAL_BALANCE_ESSD_DIR,
    regions=REGIONS,
    runs=['cal_series', 'error_dh', 'error_anom', 'error_rho', 'error_tot']
)

# ---- 5. Kriging regional mass loss ----

amce.functions.calculate_regional_mass_loss(
    regional_balance_dir=REGIONAL_BALANCE_DIR,
    region_oce_dir=REGION_OCE_DIR,
    regional_area_file=REGIONAL_AREA_FILE,
    rgi_area_file=RGI_AREA_FILE,
    zemp_regional_series_dir=ZEMP_REGIONAL_SERIES_DIR,
    ini_yr=INI_YR,
    fin_yr=FIN_YR,
    regions=REGIONS,
    rgi_code=RGI_CODE,
    rgi_region=RGI_REGION,
    mass_loss_dir=MASS_LOSS_DIR
)

#####################################
# ---- Part 2: Creation workflow ----

# ---- 0. Grid tiles per region ----

amce.creation.grid_tiles_per_region(
    rgi_region=RGI_REGION_SAN,
    rgi_code=RGI_CODE_SAN,
    regions=REGIONS_SAN,
    rgi_attribute_dir=RGI_ATTRIBUTE_DIR,
    glims_attribute_area_file=GLIMS_ATTRIBUTE_AREA_FILE,
    regional_tile_dir=REGIONAL_TILE_DIR
)


# ---- 2. OCE to tiles 0.5 grid per region ----

amce.creation.oce2tiles_05_grid_per_region(
    regions=REGIONS,
    ymin=YMIN,
    ymax=YMAX,
    regional_tile_dir=REGIONAL_TILE_DIR,
    oce_dir=REGIONAL_BALANCE_ESSD_DIR,
    oce_tile_dir=OCE_TILE_DIR
)

# ---- 3. Meters water equivalent to gigatonnes and area change 0.5 grid per region ----

amce.creation.areachange_grid_per_region(
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

amce.creation.tiles_to_global_grid(
    ymin=YMIN,
    ymax=YMAX,
    mass_change_grid_dir=MASS_CHANGE_GRID_DIR,
    oce_tile_dir=OCE_TILE_DIR,
    area_change_grid_dir=AREA_CHANGE_GRID_DIR,
    global_grid_dir=GLOBAL_GRID_DIR
)

# ---- CSV to NetCDF4 global grid 0.5 ----

amce.creation.csv2netcdf4_globalGrid(
    ymin=YMIN,
    ymax=YMAX,
    global_grid_dir=GLOBAL_GRID_DIR,
    global_grid_netcdf_dir=GLOBAL_GRID_NETCDF_DIR
)

#####################################
# ---- Part 3: Publishing workflow --

amce.publish.build_doi_release(
    mass_loss_dir=MASS_LOSS_DIR,
    regional_balance_essd_dir=REGIONAL_BALANCE_ESSD_DIR,
    global_grid_netcdf_dir=GLOBAL_GRID_NETCDF_DIR,
    version=VERSION
)

amce.publish.build_website_figures()

amce.publish.build_glambie_submission(
    mass_loss_dir=MASS_LOSS_DIR,
    lookup_anomaly_dir=LOOKUP_ANOMALY_DIR,
    rgi_code=RGI_CODE,
    mass_balance_area_file=MASS_BALANCE_AREA_FILE,
    regional_area_change_rate_file=REGIONAL_AREA_CHANGE_RATE_FILE,
    mass_balance_file=MASS_BALANCE_FILE,
    begin_year=BEGIN_YEAR,
    year_ini=YEAR_INI,
    year_fin=YEAR_FIN,
    elevation_change_file=ELEVATION_CHANGE_FILE,
    investigators_to_drop=INVESTIGATORS_TO_DROP,
    min_length_geo=MIN_LENGTH_GEO,
    long_norm_anomaly_dir=LONG_NORM_ANOMALY_DIR,
    glacier_series_file=GLACIER_SERIES_FILE,
    rgi_area_file=RGI_AREA_FILE,
    glambie_begin_year=1976
)
