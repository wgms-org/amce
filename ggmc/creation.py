from typing import Dict, List
from pathlib import Path
import os

import numpy as np
import pandas as pd
import rioxarray  # Register raterio drivers
import xarray as xr

from ggmc import propagation


def grid_tiles_per_region(
    rgi_region: Dict[str, str],
    rgi_code: Dict[str, str],
    regions: List[str],
    rgi_attribute_dir: Path,
    glims_attribute_area_file: Path,
    regional_tile_dir: Path
) -> None:
    """
    Computes glacier area on a 0.5 x 0.5 degree grid by region based on RGI/GLIMS.
    """
    regional_tile_dir.mkdir(parents=True, exist_ok=True)

    for region in regions:
        print(region)

        if region == 'CAU':
            rgi_df = pd.read_csv(
                glims_attribute_area_file, usecols=['glac_id', 'CenLat', 'CenLon', 'Area']
            )
            rgi_df.rename(columns={'glac_id': 'GLIMS_ID'}, inplace=True)
            rgi_df.set_index('GLIMS_ID', inplace=True)
        else:
            rgi_file = rgi_attribute_dir / f'{rgi_code[region]}_rgi60_{rgi_region[region]}.csv'
            rgi_df = pd.read_csv(rgi_file, usecols=['RGIId', 'CenLon', 'CenLat', 'Area', 'Connect'], encoding='latin1', index_col='RGIId')
            if region == 'GRL':
                rgi_df = rgi_df[rgi_df['Connect'] != 2].copy()

        # TODO: Why would this be necessary?
        # Drop duplicates
        rgi_df.drop_duplicates(subset=None, keep='first', inplace=True)

        # Define coordinates of 0.5 x 0.5 degrees grid cell where the glacier belongs
        # TODO: Vectorise and generalise to arbitrary grid resolution
        cen_lon_grid_lst = []
        cen_lat_grid_lst = []
        for _, row in rgi_df.iterrows():
            lon = row['CenLon']
            lat = row['CenLat']
            if round(lon) < lon:
                if round(lat) > lat:
                    cen_lon_grid = round(lon) + 0.25
                    cen_lat_grid = round(lat) - 0.25
                else:
                    cen_lon_grid = round(lon) + 0.25
                    cen_lat_grid = round(lat) + 0.25
            else:
                if round(lat) > lat:
                    cen_lon_grid = round(lon) - 0.25
                    cen_lat_grid = round(lat) - 0.25
                else:
                    cen_lon_grid = round(lon) - 0.25
                    cen_lat_grid = round(lat) + 0.25

            cen_lon_grid_lst.append(cen_lon_grid)
            cen_lat_grid_lst.append(cen_lat_grid)

        rgi_df['LON_GRID'] = cen_lon_grid_lst
        rgi_df['LAT_GRID'] = cen_lat_grid_lst

        rgi_grid_df = rgi_df.groupby(['LON_GRID', 'LAT_GRID'])['Area'].sum()
        rgi_grid_df.to_csv(regional_tile_dir / f'{region}.csv')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 0_v1.6_oce2tiles_0.5_grid_per_region.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def oce2tiles_05_grid_per_region(
    regions: List[str],
    ymin: int,
    ymax: int,
    regional_tile_dir: Path,
    oce_dir: Path,
    oce_tile_dir: Path
) -> None:
    oce_tile_dir.mkdir(parents=True, exist_ok=True)

    ######### work on regions with fix lat-lon regional limits
    for region in regions:
        print(region)

        # Read mean estimate and errors
        file_oce = oce_dir / f'{region}_gla_MEAN-CAL-mass-change-series_obs_unobs.csv'
        file_sig_dh = oce_dir / f'{region}_gla_mean-cal-mass-change_DH-ERROR_obs_unobs.csv'
        file_sig_rho = oce_dir / f'{region}_gla_mean-cal-mass-change_RHO-ERROR_obs_unobs.csv'
        file_sig_anom = oce_dir / f'{region}_gla_mean-cal-mass-change_ANOM-ERROR_obs_unobs.csv'

        oce_df = pd.read_csv(file_oce)
        sig_dh_df = pd.read_csv(file_sig_dh)
        sig_rho_df = pd.read_csv(file_sig_rho)
        sig_anom_df = pd.read_csv(file_sig_anom)

        # Read tiles for this region
        if region == 'SA1':
            file_grid = regional_tile_dir / 'SAN.csv'
            grid_df = pd.read_csv(file_grid)
            grid_df = grid_df.loc[grid_df['LAT_GRID'] < -45.50]
        elif region == 'SA2':
            file_grid = regional_tile_dir / 'SAN.csv'
            grid_df = pd.read_csv(file_grid)
            grid_df = grid_df.loc[grid_df['LAT_GRID'] > -45.50]
            grid_df = grid_df.reset_index()
        else:
            file_grid = regional_tile_dir / f'{region}.csv'
            grid_df = pd.read_csv(file_grid)

        # List years to process
        list_years = np.arange(ymin, ymax + 1)

        # We can't apply to the whole YEAR/ID dataframe at once here, we need to loop for each YEAR of the dataframes
        # to compute the pairwise error propagation for dh and density across all glaciers of that year
        list_df_out = []

        # Loop through all tiles
        for idx_tile in grid_df.index:
            grid_lat = grid_df["LAT_GRID"].loc[idx_tile]
            grid_lon = grid_df["LON_GRID"].loc[idx_tile]
            grid_area = grid_df["Area"].loc[idx_tile]
            print(f'{grid_lon}, {grid_lat}')

            # Glaciers that belong in this tile
            idx_within_tile = np.logical_and.reduce((sig_dh_df["CenLat"] >= grid_lat - 0.25,
                                                     sig_dh_df["CenLat"] < grid_lat + 0.25,
                                                     sig_dh_df["CenLon"] >= grid_lon - 0.25,
                                                     sig_dh_df["CenLon"] < grid_lon + 0.25))

            if idx_within_tile.sum() == 0:
                continue

            # Subset to tile
            oce_df_tile = oce_df[idx_within_tile]
            sig_dh_df_tile = sig_dh_df[idx_within_tile]
            sig_anom_df_tile = sig_anom_df[idx_within_tile]
            sig_rho_df_tile = sig_rho_df[idx_within_tile]
            lats_tile = oce_df_tile["CenLat"]
            lons_tile = oce_df_tile["CenLon"]
            areas = oce_df_tile["Area"]

            # TODO: Pre-format oce_df_tile outside tile loop
            area_weights = np.atleast_2d(areas / grid_area)
            year_columns = [str(year) for year in list_years]
            mean_df = oce_df_tile[year_columns].transpose().values * area_weights
            mb = mean_df.sum(axis=1)
            sigma_dh = sig_dh_df_tile[year_columns].transpose().values * area_weights
            sigma_rho = sig_rho_df_tile[year_columns].transpose().values * area_weights
            sigma_anom = sig_anom_df_tile[year_columns].transpose().values * area_weights

            sig_dh_obs, sig_rho_obs, sig_anom_obs = propagation.regional_sigma_wrapper(
                latitude=np.asarray(lats_tile),
                longitude=np.asarray(lons_tile),
                sigma_dh=sigma_dh,
                sigma_rho=sigma_rho,
                sigma_anom=sigma_anom,
                by_year=False,
                verbose=False
            )
            list_df_out.append(pd.DataFrame({
                "tile_lon": grid_lon,
                "tile_lat": grid_lat,
                "year": list_years,
                "mb": mb,
                "sig_dh": sig_dh_obs,
                "sig_rho": sig_rho_obs,
                "sig_anom": sig_anom_obs
            }))

        # Concatenate all outputs for the region
        df_out_reg = pd.concat(list_df_out)

        # Add total error
        df_out_reg["sig_tot"] = np.sqrt(df_out_reg["sig_dh"] ** 2 + df_out_reg["sig_rho"] ** 2 + df_out_reg["sig_anom"] ** 2)

        # Reshape to format expected by Ines
        df_mb_out = df_out_reg[["tile_lat", "tile_lon", "year", "mb"]]
        df_mb_out = df_mb_out.pivot(index=["tile_lat", "tile_lon"], columns='year', values='mb')

        df_sig_dh_out = df_out_reg[["tile_lat", "tile_lon", "year", "sig_dh"]]
        df_sig_dh_out = df_sig_dh_out.pivot(index=["tile_lat", "tile_lon"], columns='year', values='sig_dh')

        df_sig_anom_out = df_out_reg[["tile_lat", "tile_lon", "year", "sig_anom"]]
        df_sig_anom_out = df_sig_anom_out.pivot(index=["tile_lat", "tile_lon"], columns='year', values='sig_anom')

        df_sig_rho_out = df_out_reg[["tile_lat", "tile_lon", "year", "sig_rho"]]
        df_sig_rho_out = df_sig_rho_out.pivot(index=["tile_lat", "tile_lon"], columns='year', values='sig_rho')

        df_sig_tot_out = df_out_reg[["tile_lat", "tile_lon", "year", "sig_tot"]]
        df_sig_tot_out = df_sig_tot_out.pivot(index=["tile_lat", "tile_lon"], columns='year', values='sig_tot')

        df_mb_out.to_csv(oce_tile_dir / f'{region}_MB_mwe_grid_0.5.csv', index_label=["LAT_GRID", "LON_GRID"])
        df_sig_dh_out.to_csv(oce_tile_dir / f'{region}_sigma_dh_grid_0.5.csv', index_label=["LAT_GRID", "LON_GRID"])
        df_sig_anom_out.to_csv(oce_tile_dir / f'{region}_sigma_anom_grid_0.5.csv', index_label=["LAT_GRID", "LON_GRID"])
        df_sig_rho_out.to_csv(oce_tile_dir / f'{region}_sigma_rho_grid_0.5.csv', index_label=["LAT_GRID", "LON_GRID"])
        df_sig_tot_out.to_csv(oce_tile_dir / f'{region}_sigma_TOTAL_grid_0.5.csv', index_label=["LAT_GRID", "LON_GRID"])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1_v1.5_mwe2Gt_AreaChange_0.5_grid_per_region.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def areachange_grid_per_region(
    regions: List[str],
    area_ref_year: dict,
    area_chg_rate: dict,
    ymin: int,
    ymax: int,
    regional_tile_dir: Path,
    oce_tile_dir: Path,
    area_change_grid_dir: Path,
    mass_change_grid_dir: Path
) -> None:
    area_change_grid_dir.mkdir(parents=True, exist_ok=True)
    mass_change_grid_dir.mkdir(parents=True, exist_ok=True)

    def read_tiles_area_data(region):
        ## Create Regional grid point df: with all glacierized grid points and areas in the region from rgi
        if region == 'SA1':
            file_grid = regional_tile_dir / 'SAN.csv'
            grid_df = pd.read_csv(file_grid, sep=',', header=0)
            grid_df= grid_df.loc[(grid_df['LAT_GRID'] < -45.50)].set_index(['LAT_GRID', 'LON_GRID'])
        elif region == 'SA2':
            file_grid = regional_tile_dir / 'SAN.csv'
            grid_df = pd.read_csv(file_grid, sep=',', header=0)
            grid_df= grid_df.loc[(grid_df['LAT_GRID'] > -45.50)].set_index(['LAT_GRID', 'LON_GRID'])
        else:
            file_grid = regional_tile_dir / f'{region}.csv'
            grid_df = pd.read_csv(file_grid, sep=',', header=0, index_col=['LAT_GRID', 'LON_GRID'])

        return grid_df, dict(zip(grid_df.index, grid_df['Area']))

    def calc_current_area_li(dict_grid, tile, sample_year):
        """Calculate regional tile glacier area for a given year based on linear equation"""
        # find grid point rgi area
        rgi_area = dict_grid[tile]
        change_rate_km2 = area_chg_rate[region] / 100 * rgi_area

        # calculate tile glacier area for sample year
        area_sample_year = round(rgi_area + (sample_year - area_ref_year[region]) * change_rate_km2, 3)

        return area_sample_year

    yr_lst = list(range(ymin, ymax+1))

    for region in regions:
        print(region)

        ## 1. Calculate tile glacier area for a given year
        grid_df, dict_grid = read_tiles_area_data(region)
        tile_area_lst=[]

        for tile in grid_df.index:
            print(tile, end='\r')
            tile_grid_df = grid_df.loc[[tile]]
            for year in yr_lst:
                area_sample_year = calc_current_area_li(dict_grid, tile, year)
                tile_grid_df[str(year)]= area_sample_year
            tile_area_lst.append(tile_grid_df)
        print('')

        # Save area change file (by tile and year)
        area_chg_grid_df = pd.concat(tile_area_lst).drop(columns='Area')
        area_chg_grid_df.to_csv(area_change_grid_dir / f'{region}.csv')

        ## 2. Calculate tile glacier Total mass loss in Gt for a given year

        # Read specific mass balance file (by tile and year)
        file_mb_mwe = oce_tile_dir / f'{region}_MB_mwe_grid_0.5.csv'
        mb_grid_df = pd.read_csv(file_mb_mwe, sep=',', header=0, index_col= ['LAT_GRID', 'LON_GRID'])

        # multiply by area change to get total mass loss in Gt
        dM_Gt_grid_df = (mb_grid_df/ 10**3) * area_chg_grid_df
        dM_Gt_grid_df.to_csv(mass_change_grid_dir / f'{region}_mean.csv')

        ## 3. Calculate tile glacier Total mass loss uncertainties in Gt for a given year
        file_sig_mb_mwe = oce_tile_dir / f'{region}_sigma_TOTAL_grid_0.5.csv'
        sig_mb_grid_df = pd.read_csv(file_sig_mb_mwe, sep=',', header=0, index_col= ['LAT_GRID', 'LON_GRID'])
        dM_sig_Gt_grid_df = abs(dM_Gt_grid_df) * np.sqrt( (sig_mb_grid_df/mb_grid_df)**2 + (0.05)**2 )
        dM_sig_Gt_grid_df.to_csv(mass_change_grid_dir / f'{region}_sigma.csv')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2_v1.5_tiles2globalGridd.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def tiles_to_global_grid(
    ymin: int,
    ymax: int,
    mass_change_grid_dir: Path,
    oce_tile_dir: Path,
    area_change_grid_dir: Path,
    global_grid_dir: Path
) -> None:
    global_grid_dir.mkdir(parents=True, exist_ok=True)

    # gridded Total mass loss files
    filenames_dM_Gt = sorted(mass_change_grid_dir.glob('*_mean.csv'))
    filenames_sig_dM = sorted(mass_change_grid_dir.glob('*_sigma.csv'))

    # gridded Specific mass balance files
    filenames_mb_mwe = sorted(oce_tile_dir.glob('*_MB_mwe_grid_0.5.csv'))
    filenames_sig_mb = sorted(oce_tile_dir.glob('*_sigma_TOTAL_grid_0.5.csv'))

    # gridded area change files
    filenames_area = sorted(area_change_grid_dir.glob('*.csv'))

    # Create an empty dataframe with gridded 0.5 x 0.5 degrees
    lat_lst = np.arange(-89.75, 89.751, 0.5).tolist()
    lat_lst = lat_lst[::-1]
    lon_lst = np.arange(-179.75, 179.751, 0.5).tolist()
    lat_lon_df = pd.DataFrame(columns=lon_lst, index=lat_lst)

    years = np.arange(ymin, ymax+1).tolist()

    for year in years:
        print(year)

        # Prepare dataframes
        lat_lon_dM_df = lat_lon_df.copy()
        lat_lon_sig_dM_df = lat_lon_df.copy()
        lat_lon_MB_df = lat_lon_df.copy()
        lat_lon_sig_MB_df = lat_lon_df.copy()
        lat_lon_area_df = lat_lon_df.copy()

        # Create Total mass loss Global grid
        for file in filenames_dM_Gt:
            dM_df = pd.read_csv(file)

            for _, row in dM_df.iterrows():
                lat = row['LAT_GRID']
                lon = row['LON_GRID']
                dM = row[str(year)]
                lat_lon_dM_df.loc[lat, lon] = dM

        lat_lon_dM_df.to_csv(global_grid_dir / f'mass_change_{year}_mean.csv')

        # Create Total mass loss uncertainty Global grid
        for file in filenames_sig_dM:
            sig_dM_df = pd.read_csv(file)

            for _, row in sig_dM_df.iterrows():
                lat = row['LAT_GRID']
                lon = row['LON_GRID']
                sig_dM = row[str(year)]
                lat_lon_sig_dM_df.loc[lat, lon] = sig_dM

        lat_lon_sig_dM_df.to_csv(global_grid_dir / f'mass_change_{year}_sigma.csv')

        # Create Specific mass balance Global grid
        for file in filenames_mb_mwe:
            mb_df = pd.read_csv(file)

            for _, row in mb_df.iterrows():
                lat = row['LAT_GRID']
                lon = row['LON_GRID']
                mb = row[str(year)]
                lat_lon_MB_df.loc[lat, lon] = round(mb, 3)

        lat_lon_MB_df.to_csv(global_grid_dir / f'mass_balance_{year}_mean.csv')

        # Create Specific mass balance uncertainty Global grid
        for file in filenames_sig_mb:
            sig_mb_df = pd.read_csv(file)

            for _, row in sig_mb_df.iterrows():
                lat = row['LAT_GRID']
                lon = row['LON_GRID']
                sig_mb = row[str(year)]
                lat_lon_sig_MB_df.loc[lat, lon] = round(sig_mb, 3)

        lat_lon_sig_MB_df.to_csv(global_grid_dir / f'mass_balance_{year}_sigma.csv')

        # Create Area change Global grid
        for file in filenames_area:
            area_df= pd.read_csv(file)

            for _, row in area_df.iterrows():
                lat = row['LAT_GRID']
                lon = row['LON_GRID']
                area = row[str(year)]
                lat_lon_area_df.loc[lat, lon] = round(area,3)

        lat_lon_area_df.to_csv(global_grid_dir / f'area_change_{year}.csv')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3_v1.5_csv2netcdf4_globalGrid_0.5.py
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def csv2netcdf4_globalGrid(
    ymin: int,
    ymax: int,
    global_grid_dir: Path,
    global_grid_netcdf_dir: Path
) -> None:
    global_grid_netcdf_dir.mkdir(parents=True, exist_ok=True)
    # Clear existing files in output to avoid xarray write errors
    for existing_file in global_grid_netcdf_dir.glob('*.nc'):
        existing_file.unlink()

    years = np.arange(ymin, ymax + 1).tolist()

    for year in years:
        print(year)

        file_gla_gt = global_grid_dir / f'mass_change_{year}_mean.csv'
        file_sig_gt = global_grid_dir / f'mass_change_{year}_sigma.csv'

        file_gla_mwe = global_grid_dir / f'mass_balance_{year}_mean.csv'
        file_sig_mwe = global_grid_dir / f'mass_balance_{year}_sigma.csv'
        file_gla_area = global_grid_dir / f'area_change_{year}.csv'

        # Read glacier mass change data as dataframe
        gla_gt_df = pd.read_csv(file_gla_gt, index_col=0)
        gla_mwe_df = pd.read_csv(file_gla_mwe, index_col=0)
        gla_area_df = pd.read_csv(file_gla_area, index_col=0)

        # Define dimensions: Latitude, Longitude and Time
        lon1 = gla_gt_df.columns.values
        lon = []
        for i in range(len(lon1)):
            lon.append(float(float(lon1[i])))
        lat = gla_gt_df.index

        time = pd.date_range(start=str(year), periods=12, freq='MS')

        # Read glacier mass change data as xarray with lat, lon, time dimensions
        Glacier_newformat = xr.DataArray(data=gla_gt_df, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})
        Glacier_newformat_mwe = xr.DataArray(data=gla_mwe_df, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})
        Glacier_newformat_area = xr.DataArray(data=gla_area_df, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})

        # Give a name to the variable (here Glacier mass change)
        Glacier_newformat = Glacier_newformat.to_dataset(name='glacier_mass_change_gt')
        Glacier_newformat_mwe = Glacier_newformat_mwe.to_dataset(name='glacier_mass_change_mwe')
        Glacier_newformat_area = Glacier_newformat_area.to_dataset(name='glacier_area_km2')

        # Glacier_newformat = Glacier_newformat.expand_dims(time=time)

        # Read and add the uncertainty data to the Glacier change dataset
        sig_gt_df = pd.read_csv(file_sig_gt, index_col=0)
        sig_mwe_df = pd.read_csv(file_sig_mwe, index_col=0)
        # data = sig_gt_df.values
        # data = sig_gt_df.all()

        Glacier_newformat_Uncer_gt = xr.DataArray(data=sig_gt_df, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})
        Glacier_newformat_Uncer_mwe = xr.DataArray(data=sig_mwe_df, dims=['lat', 'lon'], coords={'lat': lat, 'lon': lon})

        # Give a name to the variable (here Glacier change Uncertainty)
        Glacier_newformat_Uncer_gt = Glacier_newformat_Uncer_gt.to_dataset(name='uncertainty_gt')
        Glacier_newformat_Uncer_mwe = Glacier_newformat_Uncer_mwe.to_dataset(name='uncertainty_mwe')

        # Add the uncertainty data (as a new variable) to the Glacier mass change variable
        Glacier_newformat = Glacier_newformat.assign(glacier_mass_change_mwe=Glacier_newformat_mwe.glacier_mass_change_mwe)
        Glacier_newformat = Glacier_newformat.assign(glacier_area_km2=Glacier_newformat_area.glacier_area_km2)
        Glacier_newformat = Glacier_newformat.assign(uncertainty_gt=Glacier_newformat_Uncer_gt.uncertainty_gt)
        Glacier_newformat = Glacier_newformat.assign(uncertainty_mwe=Glacier_newformat_Uncer_mwe.uncertainty_mwe)

        # add the time variable to the data
        Glacier_newformat = Glacier_newformat.expand_dims(time=time)
        Glacier_newformat.time.attrs['standard_name']= 'time'
        Glacier_newformat.time.attrs['long_name']= 'time axis'

        # add some information (attributes) related to Longitude and Latitude
        Glacier_newformat.lon.attrs['standard_name'] = 'longitude'
        Glacier_newformat.lon.attrs['long_name'] = 'longitude'
        Glacier_newformat.lon.attrs['units'] = 'degrees_east'
        Glacier_newformat.lon.attrs['valid_min'] = -179.75
        Glacier_newformat.lon.attrs['valid_max'] = 179.75

        Glacier_newformat.lat.attrs['standard_name'] = 'latitude'
        Glacier_newformat.lat.attrs['long_name'] = 'latitude'
        Glacier_newformat.lat.attrs['units'] = 'degrees_north'
        Glacier_newformat.lat.attrs['valid_min'] = -89.75
        Glacier_newformat.lat.attrs['valid_max'] = 89.75

        # add some information (attributes) related to Glacier_gt mass change variable
        Glacier_newformat.glacier_mass_change_gt.attrs['standard_name'] = 'mass_change_gt'
        Glacier_newformat.glacier_mass_change_gt.attrs['long_name'] = 'glacier_mass_change_gigatons'
        Glacier_newformat.glacier_mass_change_gt.attrs['units'] = 'gt'
        Glacier_newformat.glacier_mass_change_gt.attrs['_FillValue'] = 'NaN'

        # add some information (attributes) related to mass change Uncertainty_gt variable
        Glacier_newformat.uncertainty_gt.attrs['standard_name'] = 'mass_change_uncertainty_gt'
        Glacier_newformat.uncertainty_gt.attrs['long_name'] = 'glacier_mass_change_uncertainty_gigatons'
        Glacier_newformat.uncertainty_gt.attrs['units'] = 'gt'
        Glacier_newformat.uncertainty_gt.attrs['_FillValue'] = 'NaN'

        # add some information (attributes) related to Glacier_mwe mass change variable
        Glacier_newformat.glacier_mass_change_mwe.attrs['standard_name'] = 'mass_change_mwe'
        Glacier_newformat.glacier_mass_change_mwe.attrs['long_name'] = 'glacier_mass_change_meter_water_equivalent'
        Glacier_newformat.glacier_mass_change_mwe.attrs['units'] = 'm w.e.'
        Glacier_newformat.glacier_mass_change_mwe.attrs['_FillValue'] = 'NaN'

        # add some information (attributes) related to mass change Uncertainty_mwe variable
        Glacier_newformat.uncertainty_mwe.attrs['standard_name'] = 'mass_change_uncertainty_mwe'
        Glacier_newformat.uncertainty_mwe.attrs['long_name'] = 'glacier_mass_change_uncertainty_meter_water_equivalent'
        Glacier_newformat.uncertainty_mwe.attrs['units'] = 'm w.e.'
        Glacier_newformat.uncertainty_mwe.attrs['_FillValue'] = 'NaN'

        # add some information (attributes) related to Glacier_area variable
        Glacier_newformat.glacier_area_km2.attrs['standard_name'] = 'glacier_area'
        Glacier_newformat.glacier_area_km2.attrs['long_name'] = 'glacier_area_square_kilometers'
        Glacier_newformat.glacier_area_km2.attrs['units'] = 'km2'
        Glacier_newformat.glacier_area_km2.attrs['_FillValue'] = 'NaN'

        # Write some global attributes (with respect to the whole data sets)
        Glacier_newformat.attrs['title'] = 'Global gridded annual glacier mass change'
        Glacier_newformat.attrs['data_version'] = 'version-wgms-fog-2025-01'
        Glacier_newformat.attrs['project'] = 'Dussaillant et al. dataset'
        Glacier_newformat.attrs['institution'] = 'World Glacier Monitoring Service - Geography Department - University of Zurich - Zurich - Switzerland'
        Glacier_newformat.attrs['created_by'] = 'Dr. Ines Dussaillant - ines.dussaillant@geo.uzh.ch, ines.dussaillant@gmail.com'
        Glacier_newformat.attrs['references'] = 'Fluctuation of Glaciers (FoG) database version wgms-fog-2025-01'
        Glacier_newformat.attrs['citation'] = 'Dussaillant et al. 2025'
        Glacier_newformat.attrs['conventions'] = 'CF Version CF-1.8'
        Glacier_newformat.attrs['dataset_description'] = (
            'Horizontal resolution: 0.5° (latitude - longitude), GCS_WGS_1984' \
            'Temporal resolution: Annual, hydrological year' \
            'Temporal coverage: hydrological years from 1976 to 2024' \
            'Observational sample: 96% of world glaciers with valid observations'\
            'Spatial interpolation method: Kriging'\
        )

        crs = 'EPSG:4326'  # WGS84
        Glacier_newformat = Glacier_newformat.rio.write_crs(crs, inplace=True)

        Glacier_newformat = Glacier_newformat.sel(time=str(year) + '-01')
        Glacier_newformat.to_netcdf(global_grid_netcdf_dir / f'{year}.nc', unlimited_dims='time', mode='w')
