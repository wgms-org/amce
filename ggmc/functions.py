import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyproj

from . import helpers, kriging, propagation


def format_mass_balance_data(
    input_file: Path,
    begin_year: int,
    output_dir: Path
) -> None:
    """
    Format mass balance data into separate files by variable.

    Each variable is written to a file with glaciers as columns and years as rows.

    Parameters
    ----------
    input_file
        Path to CSV file with columns WGMS_ID, YEAR,
        ANNUAL_BALANCE, SUMMER_BALANCE, WINTER_BALANCE, and ANNUAL_BALANCE_UNC.
    begin_year
        Earliest year to include in output files.
    output_dir
        Path to directory where output files will be saved.
    """
    df = pd.read_csv(input_file)
    column_keys = {
        'ANNUAL_BALANCE': 'ba',
        'SUMMER_BALANCE': 'bs',
        'WINTER_BALANCE': 'bw',
        'ANNUAL_BALANCE_UNC': 'ba_unc'
    }
    for column, key in column_keys.items():
        temp = df.pivot(
            index='YEAR',
            columns='WGMS_ID',
            values=column
        )
        # HACK: Ensure WGMS_ID columns are preserved, even if empty
        # Avoids missing index error on line
        # unc_glac_anom = helpers.calc_spt_anomalies_unc(glac_anom, ba_unc_df, glac_anom.columns.to_list())
        temp = temp[temp.index >= begin_year]
        temp.to_csv(output_dir / f'{key}.csv', index=True)


def format_elevation_change(
    elevation_change_file: Path,
    glacier_series_file: Path,
    investigators_to_drop: List[str],
    glacier_coordinate_file: Path,
    geodetic_change_file: Path,
    density_factor: Tuple[float, float]
) -> None:
    """
    Format elevation change data into specific mass balance with uncertainty.

    Writes a file with glacier coordinates and other with geodetic mass change rates.

    Parameters
    ----------
    elevation_change_file
        Path to elevation change data.
    glacier_series_file
        Path to glacier series data.
    investigators_to_drop
        Investigators records should be removed.
    glacier_coordinate_file
        Path to output glacier coordinate file.
    geodetic_change_file
        Path to output geodetic mass balance change file.
    density_factor
        Density conversion factor and its sigma.
    """
    # Glacier coordinates
    df = pd.read_csv(glacier_series_file, usecols=['WGMS_ID', 'LATITUDE', 'LONGITUDE'])
    df.set_index('WGMS_ID', inplace=True)
    df.sort_index(inplace=True)
    df.to_csv(glacier_coordinate_file, index=True)

    # Geodetic data
    df = pd.read_csv(elevation_change_file, low_memory=False)
    df.set_index('WGMS_ID', inplace=True)
    df.sort_index(inplace=True)
    mask = (
        df['ELEVATION_CHANGE'].notnull() &
        df['SURVEY_DATE'].notnull() &
        df['REFERENCE_DATE'].notnull() &
        (
            (df['GLACIER_REGION_CODE'].eq('CAU') & df['GLIMS_ID'].notnull()) |
            (df['GLACIER_REGION_CODE'].ne('CAU') & df['RGI60_ID'].notnull())
        )
    )
    df: pd.DataFrame = df[mask]
    df['ID'] = df['GLIMS_ID'].where(df['GLACIER_REGION_CODE'] == 'CAU', df['RGI60_ID'])
    # TODO: Migrate to fuzzy dates (begin_date_min, begin_date_max, ...)
    df['ini_date'] = helpers.wgms_date_to_decimal_year(df['REFERENCE_DATE'])
    df['fin_date'] = helpers.wgms_date_to_decimal_year(df['SURVEY_DATE'])
    df['elevation_chg_rate'] = helpers.change_to_rate(
        change=df['ELEVATION_CHANGE'],
        begin_date=df['ini_date'],
        end_date=df['fin_date']
    )
    df['sigma_elevation_chg'] = helpers.change_to_rate(
        change=df['ELEVATION_CHANGE_UNC'],
        begin_date=df['ini_date'],
        end_date=df['fin_date']
    )
    df['mb_chg_rate'] = df['elevation_chg_rate'] * density_factor[0]
    sigma_obs_mb_chg = df['sigma_elevation_chg'] * density_factor[0]
    # Fill missing elevation change uncertainties with the global mean
    # TODO: Is it not better to fill after dropping records with short durations?
    sigma_obs_mb_chg.fillna(sigma_obs_mb_chg.mean(), inplace=True)
    df['sigma_tot_mb_chg'] = df['mb_chg_rate'].abs() * (
        (sigma_obs_mb_chg / df['mb_chg_rate']) ** 2 +
        (density_factor[1] / density_factor[0]) ** 2
    ) ** 0.5
    # TODO: Move to mask above
    mask = ~df['INVESTIGATOR'].isin(investigators_to_drop)
    df = df[mask]
    # TODO: Limit to the columns actually needed downstream
    df[[
        'GLACIER_REGION_CODE', 'GLACIER_SUBREGION_CODE', 'POLITICAL_UNIT',
        'NAME', 'ID', 'AREA_SURVEY_YEAR', 'LATITUDE', 'LONGITUDE', 'INVESTIGATOR',
        'REFERENCE', 'ini_date', 'fin_date', 'elevation_chg_rate', 'sigma_elevation_chg',
        'mb_chg_rate', 'sigma_tot_mb_chg'
    ]].to_csv(geodetic_change_file, index=True)


def calculate_global_glacier_spatial_anomaly(
    year_ini: int,
    year_fin: int,
    # TODO: Determine from data
    begin_year: int,
    mass_balance_file: Path,
    ba_file: Path,
    ba_unc_file: Path,
    urumqi_missing_years_file: Path,
    glacier_coordinate_file: Path,
    geodetic_change_file: Path,
    regions: List[str],
    mean_anomaly_dir: Path,
    lookup_anomaly_dir: Path,
    long_norm_anomaly_dir: Path
) -> None:
    reference_period = range(year_ini, year_fin + 1)
    for path in (mean_anomaly_dir, lookup_anomaly_dir, long_norm_anomaly_dir):
        path.mkdir(parents=True, exist_ok=True)

    ##### 2.1 READ MASS BALANCE DATA ######

    # read FoG file with global annual and seasonal mass-balance data
    # TODO: Why not use ba_file?
    mb_df = pd.read_csv(mass_balance_file)
    years = list(range(begin_year, mb_df['YEAR'].max() + 1))

    ba_df = pd.read_csv(ba_file, index_col='YEAR')
    ba_df.columns = ba_df.columns.astype(int)

    ### Add missing years to Urumqi glacier fog_id 853
    # TODO: Apply patch upstream
    missing_years_df = pd.read_csv(urumqi_missing_years_file, index_col='YEAR')
    missing_years_df.columns = missing_years_df.columns.astype(int)
    ba_df = ba_df.fillna(missing_years_df)

    ba_unc_df = pd.read_csv(ba_unc_file, index_col='YEAR')
    ba_unc_df.columns = ba_unc_df.columns.astype(int)

    # TODO: Is sorting necessary here?
    coord_gla_df = pd.read_csv(glacier_coordinate_file, index_col='WGMS_ID')
    coord_gla_df.sort_index(inplace=True)

    ##### 2.2 READ GEODETIC DATA ######

    # TODO: Is sorting necessary here?
    geo_df = pd.read_csv(geodetic_change_file)
    geo_df.sort_values(by=['WGMS_ID'], inplace=True)
    geo_glacier_ids = geo_df['WGMS_ID'].unique().tolist()

    for region in regions:
        print(region)

        ## create empty dataframes for spatial anomalies and uncertainties
        spt_anom_df = pd.DataFrame(index=years)
        spt_anom_df.index.name = 'YEAR'
        spt_anom_lst = []

        spt_anom_err_df = pd.DataFrame(index=years)
        spt_anom_err_df.index.name = 'YEAR'
        sig_spt_anom_lst = []

        ## Select geodetic data for region
        # TODO: Move to constant
        subregion_mapping = {
            'SA1': 'SAN-01',
            'SA2': 'SAN-02'
        }
        if region in subregion_mapping:
            mask = geo_df['GLACIER_SUBREGION_CODE'].eq(subregion_mapping[region])
        else:
            mask = geo_df['GLACIER_REGION_CODE'].eq(region)
        # create a list of fog_ids with geodetic data for the region group
        geo_glacier_ids = geo_df.loc[mask, 'WGMS_ID'].unique().tolist()

        ############################################################################################################################
        ###### 3. CALCULATING SPATIAL ANOMALIES ######

        ## SELECT MASS BALANCE DATA FOR GLACIER REGION GROUP
        ## create list of glacier mass balance series ids possible to calculate the glacier temporal variabiity or anomaly
        ## remove or add neighbouring glacier mass balance series

        # TODO: Move to parameters
        if region == 'ASN': # add Urumqui, remove Hamagury yuki, add
            add_id_lst = [853, 817]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
            rem_id = 897  # Hamagury yuki (ASN)
            rem_id_lst2 = [897, 1511, 1512]  # Hamagury yuki (ASN), Urumqi East and west branches (ASC)
            glac = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region)) |(mb_df['GLACIER_REGION_CODE'] == 'ALA')| (mb_df['GLACIER_REGION_CODE'] == 'ASC')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region)) | (mb_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst2)].index)
            glac_reg = glac_reg.drop(glac_reg[glac_reg['WGMS_ID'] == rem_id].index)

        if region == 'ASE':
            add_id_lst = [817, 853]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
            rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
            glac = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region)) |(mb_df['GLACIER_REGION_CODE'] == 'ASC')| (mb_df['GLACIER_REGION_CODE'] == 'ASW')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region)) | (mb_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'ASC':
            rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
            glac = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region)) |(mb_df['GLACIER_REGION_CODE'] == 'ASE')| (mb_df['GLACIER_REGION_CODE'] == 'ASW')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = glac_reg.drop(glac_reg[glac_reg['WGMS_ID'].isin(rem_id_lst)].index)

        if region == 'ASW':
            add_id_lst = [817, 853]  # Ts. Tuyuksuyskiy (ASC), Urumqi (ASC)
            rem_id_lst = [1511, 1512]  # Urumqi East and west branches (ASC)
            glac = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region)) |(mb_df['GLACIER_REGION_CODE'] == 'ASC')| (mb_df['GLACIER_REGION_CODE'] == 'ASE')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region)) | (mb_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'CEU':
            glac = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'SA1':
            rem_id_lst = [3902, 3903, 3904, 3905, 1344, 3972]  # keep Martial Este only
            glac = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == 'SAN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = glac

        if region == 'SA2':  # keep Echaurren Norte only
            rem_id_lst = [3902, 3903, 3904, 3905, 2000, 3972]
            glac = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == 'SAN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = glac

        if region == 'NZL':
            add_id_lst = [2000]  # Martial Este (SAN-01)
            glac = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region)) | (mb_df['WGMS_ID'].isin(add_id_lst))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'ANT':
            rem_id_lst = [878, 3973]  # Dry valley glaciers
            glac = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region))), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'].isin(rem_id_lst)].index)
            glac_reg = glac

        if region == 'RUA':
            glac = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)) |(mb_df['GLACIER_REGION_CODE'] == 'SJM') , ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'SJM':
            glac = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'ALA':
            glac = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)) |(mb_df['GLACIER_REGION_CODE'] == 'WNA') , ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'WNA':
            glac = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)) |(mb_df['GLACIER_REGION_CODE'] == 'ALA'), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'TRP':
            rem_id = 226  # Yanamarey
            glac = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac = glac.drop(glac[glac['WGMS_ID'] == rem_id].index)
            glac_reg = glac

        if region == 'ACS':
            glac = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region)) | (mb_df['GLACIER_REGION_CODE'] == 'ACN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'ACN':
            glac = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'GRL':
            glac = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region)) |(mb_df['GLACIER_REGION_CODE'] == 'ACN')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'ISL':
            glac = mb_df.loc[((mb_df['GLACIER_REGION_CODE'] == str(region)) |(mb_df['GLACIER_REGION_CODE'] == 'GRL')), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        if region == 'SCA':
            glac = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = glac

        if region == 'CAU':
            glac = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]
            glac_reg = mb_df.loc[(mb_df['GLACIER_REGION_CODE'] == str(region)), ['GLACIER_REGION_CODE', 'WGMS_ID']]

        ## Find all possible individual glacier anomalies (with respect to reference period) for the given glacier id

        ## number crunching:   select mass-balance data for glacier region groups
        ba_glac_df = ba_df.loc[:, list(glac['WGMS_ID'].unique().tolist())]
        glac_anom = helpers.calc_anomalies(ba_glac_df, reference_period, region)
        unc_glac_anom = helpers.calc_spt_anomalies_unc(glac_anom, ba_unc_df, glac_anom.columns.to_list())

        # FOR SA2 ONLY: if no uncertainty measurement use the regional annual mean uncertainty of the glaciological sample
        if unc_glac_anom.isnull().sum().sum():
            for id in unc_glac_anom.columns.tolist():
                year_min = glac_anom[id].first_valid_index()
                yrs = list(range(begin_year, year_min))
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
        reg_glac_anom = helpers.calc_anomalies(ba_reg_glac_df, reference_period, region)

        # ## Correct suspicious anomaly from Echaurren Norte by normalizing past period to present period amplitude.
        # TODO: Is this really needed again?
        if region == 'SA2':
            STD_ech_ok = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(2004, (2022 + 1))))].std()
            STD_ech_bad = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(1980, (1999 + 1))))].std()
            reg_glac_anom_pres_ok = reg_glac_anom.loc[reg_glac_anom.index >= 2004]
            norm_past = reg_glac_anom.loc[reg_glac_anom.index.isin(list(range(1885, (2003 + 1))))] / STD_ech_bad
            reg_glac_anom_past_new = (norm_past * STD_ech_ok).round(decimals=1)
            reg_glac_anom = pd.concat([reg_glac_anom_past_new, reg_glac_anom_pres_ok], axis = 0)

        # ## select close anomalies for calculating the fog_id glacier anomaly

        spatial_id_fin_lst = glac_anom.columns.to_list()

        close_gla_weights = coord_gla_df.loc[spatial_id_fin_lst, :]
        lat_glac = close_gla_weights['LATITUDE']
        lon_glac= close_gla_weights['LONGITUDE']

        # ROMAIN: Replacing by inverse-distance weighting by kriging here
        anoms_4_fog_id_df = glac_anom[spatial_id_fin_lst]

        # Get variance of anomalies in this region for the kriging algorithm
        var_anom = np.nanvar(anoms_4_fog_id_df)

        # We can't apply to the whole YEAR/ID dataframe at once here, we need to loop for each YEAR of the dataframes
        # to compute the kriging

        arr_mean_anom = np.ones((len(anoms_4_fog_id_df.index), len(geo_glacier_ids)), dtype=np.float32)
        arr_sig_anom = np.ones((len(anoms_4_fog_id_df.index), len(geo_glacier_ids)), dtype=np.float32)
        for i in range(len(anoms_4_fog_id_df.index)):
            print(anoms_4_fog_id_df.index[i])

            # Create dataframe with anomalies, lat and lon
            yearly_anom_df = anoms_4_fog_id_df.iloc[i, :]
            obs_df = pd.DataFrame(data={"ba_anom": yearly_anom_df.values, "lat": np.array(lat_glac), "lon": np.array(lon_glac)})
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
            lat_id = coord_gla_df.loc[geo_glacier_ids, 'LATITUDE']
            lon_id = coord_gla_df.loc[geo_glacier_ids, 'LONGITUDE']

            # Create dataframe with points where to predict (could be several at once but here always one)
            pred_df = pd.DataFrame(data={"lat": lat_id, "lon": lon_id})

            # Kriging at the coordinate of the current glacier
            mean_anom, sig_anom = kriging.wrapper_latlon_krige_ba_anom(df_obs=obs_df, df_pred=pred_df, var_anom=var_anom)
            arr_mean_anom[i, :] = mean_anom
            arr_sig_anom[i, :] = sig_anom

        # And write back the 1D list of uncertainties into an indexed (by YEAR) dataframe
        anom_fog_id_df = pd.DataFrame(index=anoms_4_fog_id_df.index, data=arr_mean_anom, columns=[str(fog_id) for fog_id in geo_glacier_ids])
        sig_anom_df = pd.DataFrame(index=anoms_4_fog_id_df.index, data=arr_sig_anom, columns=[str(fog_id) for fog_id in geo_glacier_ids])

        ## CALCULATE:  mean anomaly for fog_id
        ## if glacier has in situ measurements i.e. dist = 0 use the own glaciers anomaly
        anom_fog_id_df = anom_fog_id_df.loc[anom_fog_id_df.index >= begin_year]
        spt_anom_lst.append(anom_fog_id_df)

        ## CALCULATE: Uncertainty for fog_id
        sig_anom_df = round(sig_anom_df, 2)
        sig_anom_df = sig_anom_df.loc[sig_anom_df.index >= begin_year]
        sig_spt_anom_lst.append(sig_anom_df)

        # write the data for that entry
        glac_anom.to_csv(lookup_anomaly_dir / f'{region}_all_SEL_gla_anomalies.csv')
        reg_glac_anom.to_csv(lookup_anomaly_dir / f'{region}_all_reg_gla_anomalies.csv')
        unc_glac_anom.to_csv(lookup_anomaly_dir / f'{region}_all_SEL_gla_anomalies_UNC.csv')

        ### Save all glacier anomalies and uncertainties - exclude uncertainties from the SAN regions
        spt_anom_df = pd.concat(spt_anom_lst, axis='columns')
        spt_anom_path = mean_anomaly_dir / f'{region}_spt_anoms.csv'
        spt_anom_df.to_csv(spt_anom_path)

        sig_spt_anom_df = pd.concat(sig_spt_anom_lst, axis='columns')
        sig_spt_path = mean_anomaly_dir / f'{region}_spt_ERRORs.csv'
        sig_spt_anom_df.to_csv(sig_spt_path)

        ### Save glacier anomalies and uncertainties OK with long time periods
        reg_ok_lst = ['ACS', 'ACN', 'ASW', 'ASE', 'ASC', 'ASN', 'ALA', 'SCA', 'SA2']
        if region in reg_ok_lst:
            spt_anom_df.to_csv(long_norm_anomaly_dir / f'{region}_spt_anoms.csv')
            sig_spt_anom_df.to_csv(long_norm_anomaly_dir / f'{region}_spt_ERRORs.csv')

    ### 4. ADD NORMALIZED SERIES FROM NEIGHBOURING GLACIERS TO EXTEND ANOMALIES BACK IN TIME
    reg_norm_lst = ['ANT', 'RUA', 'SJM', 'SA1', 'ISL', 'NZL', 'TRP', 'CEU', 'WNA', 'CAU', 'GRL']

    for region in reg_norm_lst:
        if region not in regions:
            continue
        spt_anom_fill_lst = []
        spt_anom_sig_fill_lst = []
        print(region)

        # TODO: Avoid reading files back in
        spt_anom_in = mean_anomaly_dir / f'{region}_spt_anoms.csv'
        spt_anom_df = pd.read_csv(spt_anom_in, index_col=0)
        sig_spt_anom_in = mean_anomaly_dir / f'{region}_spt_ERRORs.csv'
        sig_spt_anom_df = pd.read_csv(sig_spt_anom_in, index_col=0)

        fog_id_lst = spt_anom_df.columns.to_list()

        for fog_id in fog_id_lst:
            max_sig = sig_spt_anom_df[fog_id].max().max()
            STD_id = spt_anom_df[fog_id].loc[spt_anom_df[fog_id].index.isin(list(reference_period))].std()

            # TODO: Avoid reading files back in
            # TODO: Move to parameters
            if region == 'ISL': ## Get series from Storbreen, Aalfotbreen and Rembesdalskaaka to normalize (SCA, fog_ids 302, 317, 2296)
                neighbour_anom_in = lookup_anomaly_dir / 'SCA_all_SEL_gla_anomalies.csv'
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','302','317','2296'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = mean_anomaly_dir / f'SCA_spt_ERRORs.csv'
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR','302','317','2296'], index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour

            if region in ['SJM', 'RUA']: ## Get series from Storglacieren to normalize (SCA, fog_ids 332)
                neighbour_anom_in = lookup_anomaly_dir / 'SCA_all_reg_gla_anomalies.csv'
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, delimiter=',', header=0, usecols= ['YEAR','332'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = mean_anomaly_dir / f'SCA_spt_ERRORs.csv'
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, delimiter=',', header=0, usecols= ['YEAR', '332'], index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour

            if region == 'CEU':  ## Get series from Claridenfirn (CEU, fog_ids 2660)
                neighbour_anom_in = lookup_anomaly_dir / 'CEU_all_SEL_gla_anomalies.csv'
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, usecols=['YEAR', '2660'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = mean_anomaly_dir / f'CEU_spt_ERRORs.csv'
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, usecols=['YEAR', '4617', '4619', '4620'], index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour

            if region == 'WNA':  ## Get series from Taku glacier (ALA, fog_ids 124)
                neighbour_anom_in = lookup_anomaly_dir / 'WNA_all_SEL_gla_anomalies.csv'
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, usecols=['YEAR', '124'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = mean_anomaly_dir / f'ALA_spt_ERRORs.csv'
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, usecols=['YEAR', '124'], index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour

            if region == 'CAU':  ## Get series from Hinteeisferner, Kesselwand (CEU, fog_ids 491,507)
                neighbour_anom_in = lookup_anomaly_dir / 'CEU_all_SEL_gla_anomalies.csv'
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, usecols=['YEAR', '491', '507'], index_col=['YEAR'])
                # print(neighbour_anom_df)
                neighbour_sig_mean_anom_in = mean_anomaly_dir / f'CEU_spt_ERRORs.csv'
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, usecols=['YEAR', '491', '507'], index_col=['YEAR'])
                # print(neighbour_sig_mean_anom_df)
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour

            if region == 'GRL':  ## Get series from Meighen and Devon Ice Caps to normalize (ACN, fog_ids 16, 39)
                neighbour_anom_in = lookup_anomaly_dir / 'GRL_all_SEL_gla_anomalies.csv'
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, usecols=['YEAR', '16', '39'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = mean_anomaly_dir / f'ACN_spt_ERRORs.csv'
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, usecols=['YEAR', '102349', '104095'], index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / STD_neigbour

            if region in ['ANT', 'NZL', 'SA1', 'TRP']: ## Get series from Echaurren to normalize (SA2, fog_id 1344)
                neighbour_anom_in = lookup_anomaly_dir / 'SA2_all_reg_gla_anomalies.csv'
                neighbour_anom_df = pd.read_csv(neighbour_anom_in, usecols= ['YEAR','1344'], index_col=['YEAR'])
                neighbour_sig_mean_anom_in = mean_anomaly_dir / f'SA2_spt_ERRORs.csv'
                neighbour_sig_mean_anom_df = pd.read_csv(neighbour_sig_mean_anom_in, index_col=['YEAR'])
                max_neighbour_sig_mean_anom = neighbour_sig_mean_anom_df.max(axis=1)
                STD_neigbour = neighbour_anom_df.loc[neighbour_anom_df.index.isin(list(reference_period))].std()
                norm_neighbour = neighbour_anom_df / (STD_neigbour)

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
        reg_anom_fill_path = long_norm_anomaly_dir / f'{region}_spt_anoms_fill.csv'
        reg_anom_fill_df.to_csv(reg_anom_fill_path)

        reg_anom_sig_fill_df = pd.concat(spt_anom_sig_fill_lst, axis='columns')
        reg_anom_sig_fill_path = long_norm_anomaly_dir / f'{region}_spt_ERRORs_fill.csv'
        reg_anom_sig_fill_df.to_csv(reg_anom_sig_fill_path)


def calculate_consensus_estimate_and_error_global_glacier_regional_anomaly(
    begin_year: int,
    end_year: int,
    # start year of the period of interest, only geodetic from this year on will be considered, 0 if to use the start of the anomaly period
    min_year_geo_obs: int,
    # minimum length in years of the geodetic observations accounted for anomaly calibration
    min_length_geo: float,
    long_norm_anomaly_dir: Path,
    geodetic_change_file: Path,
    region_oce_dir: Path,
    regions: List[str]
) -> None:
    region_oce_dir.mkdir(parents=True, exist_ok=True)

    #  Make a list of the full period of interest
    yr_lst = list(range(begin_year, end_year + 1))

    # read global geodetic mass-balance data from csv into dataframe
    # TODO: Move to data formatting function
    geo_df = pd.read_csv(
        geodetic_change_file,
        usecols=['WGMS_ID', 'GLACIER_REGION_CODE', 'GLACIER_SUBREGION_CODE', 'ini_date', 'fin_date', 'mb_chg_rate', 'sigma_tot_mb_chg', 'sigma_elevation_chg', 'elevation_chg_rate']
    )
    # TODO: Is sorting needed?
    geo_df.sort_values('WGMS_ID', inplace=True)
    geo_df['POR'] = geo_df['fin_date'] - geo_df['ini_date']

    for region in regions:
        print(region)

        ## create regional OCE and sigma OCE empty dataframes, including 3 error sources dataframes and the full error
        year_index = pd.Series(yr_lst, name='YEAR')
        reg_oce_df = pd.DataFrame(index=year_index)
        reg_sig_dh_oce_df = pd.DataFrame(index=year_index)
        reg_sig_rho_oce_df = pd.DataFrame(index=year_index)
        reg_sig_anom_oce_df = pd.DataFrame(index=year_index)
        reg_sig_oce_df = pd.DataFrame(index=year_index)

        ## number crunching: select geodetic data for glacier region group
        # TODO: Move to parameters or constant
        subregion_mapping = {
            'SA1': 'SAN-01',
            'SA2': 'SAN-02'
        }
        if region in subregion_mapping:
            mask = geo_df['GLACIER_SUBREGION_CODE'].eq(subregion_mapping[region])
        else:
            mask = geo_df['GLACIER_REGION_CODE'].eq(region)
        geo_reg_df = geo_df.loc[mask].copy()
        # TODO: Perform filling in data formatting function (where already done)
        geo_reg_df['sigma_elevation_chg'] = geo_reg_df['sigma_elevation_chg'].fillna(
            geo_reg_df['sigma_elevation_chg'].mean()
        )

        ## create a list of wgms_ids belonging to the region group
        reg_wgms_id_lst = geo_reg_df['WGMS_ID'].unique().tolist()

        # # read regional anomaly data and uncertainties from csv files into dataframe
        reg_spt_anom_file = list(long_norm_anomaly_dir.glob(f'{region}_spt_anoms*.csv'))[0]
        reg_spt_anom_df = pd.read_csv(reg_spt_anom_file, index_col='YEAR')
        reg_spt_anom_df.columns = reg_spt_anom_df.columns.astype(int)
        reg_spt_anom_err_file = list(long_norm_anomaly_dir.glob(f'{region}_spt_ERRORs*.csv'))[0]
        reg_spt_anom_err_df = pd.read_csv(reg_spt_anom_err_file, index_col='YEAR')
        reg_spt_anom_err_df.columns = reg_spt_anom_err_df.columns.astype(int)

        ############################################################################################################################

        ###### CALCULATING OCE: Loop through all glaciers in the region with available geodetic estimates ######

        for fog_id in reg_wgms_id_lst:
            print(fog_id)

            # TODO: Convert to pd.Series
            id_spt_anom_df = reg_spt_anom_df[[fog_id]]
            id_spt_anom_err_df = reg_spt_anom_err_df[[fog_id]]

            # Define period of the complete anomaly series
            min_year = id_spt_anom_df.first_valid_index()
            # TODO: Move 2000 to parameters
            if min_year > 2000:
                min_year = 2000
            elif min_year_geo_obs > 0:
                min_year = min_year_geo_obs
            # TODO: Should this be id_spt_anom_df.last_valid_index()?
            max_year = id_spt_anom_df.index.max()

            # Create geodetic mass balance series and geodetic Dataframe for selected glacier
            # TODO: Pre-index on WGMS_ID
            geo_mb_gla_df = geo_reg_df.loc[geo_reg_df['WGMS_ID'] == fog_id]
            # TODO: Filter records in data formatting function
            geo_mb_gla_df = geo_mb_gla_df[geo_mb_gla_df['POR'] > min_length_geo]

            # Select geodetic estimates inside the period of interest and longer than min_length_geo
            # TODO: Move outside for loop using multi-indexing
            # TODO: What is the reasoning behind the -2 and +1 here?
            geo_ind_gla_sel_df = geo_mb_gla_df[(geo_mb_gla_df['ini_date'] >= min_year - 2) & (geo_mb_gla_df['fin_date'] <= max_year + 1)]

            # Create empty dataframes for calibrated series, calibrated series uncertainty, sigma geodetic uncertainty and distance to observation period
            # TODO: Convert to pd.Series
            cal_series_df = pd.DataFrame(index=year_index)
            sig_dh_cal_series_df = pd.DataFrame(index=year_index)
            sig_rho_cal_series_df = pd.DataFrame(index=year_index)
            sig_anom_cal_series_df = pd.DataFrame(index=year_index)
            sigma_geo_df = pd.DataFrame(index=year_index)
            dist_geo_df = pd.DataFrame(index=year_index)
            dist_geo_df = dist_geo_df.reset_index()


            ###### Calculate the calibrated series #####
            for index, row in geo_ind_gla_sel_df.iterrows():
                # TODO: Why filter further with integer-resolution period of record?
                if (int(row['fin_date']) - int(row['ini_date'])) <= min_length_geo:
                    continue
                # TODO: Was it intentional to leave out the last year?
                ref_period_geo_obs = range(int(row['ini_date']), int(row['fin_date']))
                ref_anom_period_df = id_spt_anom_df[id_spt_anom_df.index.isin(ref_period_geo_obs)]
                # -------here there is a problem with the dates !!!! need to correct geodetic values to hydrological years
                avg_ref_anom = ref_anom_period_df.mean()
                ref_anom = id_spt_anom_df - avg_ref_anom

                cal_val = row['mb_chg_rate'] + ref_anom
                cal_series_df[f'serie_{index}'] = cal_val[fog_id]

                # Three uncertainty sources: elevation change, density conversion, and anomaly

                # We save them in the same unit of mass change, multiplying dh by density
                # TODO: Move density to parameters or constant
                # 1. Error in mass change from dh error sources
                # TODO: Why not use sigma_tot_mb_chg here?
                sig_dh_cal_series_df[f'sig_serie_{index}'] = row['sigma_elevation_chg'] * 0.85
                # 2. Error in mass change from density error sources
                sig_rho_cal_series_df[f'sig_serie_{index}'] = np.abs(row['elevation_chg_rate']) * 0.06
                # 3. Error in mass change from anomalies
                sig_anom_cal_series_df[f'sig_serie_{index}'] = id_spt_anom_err_df

                # Create geodetic estimate uncertainty dataframe
                sigma_geo_df[f'serie_{index}'] = row['sigma_tot_mb_chg']
                i_date = int(row['ini_date'])
                f_date = int(row['fin_date'])

                # # Create Distance to geodetic observation period dataframe
                dist_geo_df[f'serie_{index}']= dist_geo_df['YEAR'].apply(lambda row: helpers.dis_fil(row, i_date, f_date))

            if cal_series_df.empty:
                continue

            ###### Apply weights to calculate the mean calibrated series #####

            ## Calculate weight related to geodetic estimate uncertainty
            fill_sigma = geo_ind_gla_sel_df['sigma_tot_mb_chg'].max()
            if math.isnan(fill_sigma):
                fill_sigma = 1.0

            sigma_geo_df.fillna(fill_sigma, inplace=True)
            sigma_ratio_df = 1 / sigma_geo_df
            wgt1_sigma_df = sigma_ratio_df.div(sigma_ratio_df.sum(axis=1), axis=0) # pass to percentage

            ## Calculate weight related to distance to the geodetic estimate survey period
            p = 2 ## P value for inverse distance weighting
            dist_geo_df.set_index('YEAR', inplace=True)
            inv_dist_df = (1 / dist_geo_df) ** p
            wgt2_dist_df = inv_dist_df.div(inv_dist_df.sum(axis=1), axis=0) # pass to percentage

            ## Calculate weight related to uncertaity and distance combined
            W1_W2_comb_df = wgt1_sigma_df.add(wgt2_dist_df)

            ##### Calculate MEANS of calibrated series: Artihmetic, Weight_1, Weight_2, Weight_combined #####

            # Apply the weights to the calibrated series
            # cal_series_W1_df = cal_series_df.mul(wgt1_sigma_df)
            # cal_series_W2_df = cal_series_df.mul(wgt2_dist_df)
            cal_series_W1_W2_df = cal_series_df.mul(W1_W2_comb_df) / 2

            # # calibrated series means
            # cal_mean_df = pd.DataFrame(index=year_index)
            # cal_mean_df['MEAN'] = cal_series_df.mean(axis=1)
            # cal_mean_df['MEAN_sigma_W'] = cal_series_W1_df.sum(axis=1, min_count=1)
            # cal_mean_df['MEAN_dist_W'] = cal_series_W2_df.sum(axis=1, min_count=1)
            # cal_mean_df['MEAN_combined_W'] = cal_series_W1_W2_df.sum(axis=1, min_count=1)

            # # Cumulative series of the different means
            # cal_mean_cum_df = pd.DataFrame(index=year_index)
            # cal_mean_cum_df['Cum_MEAN'] = cal_mean_df['MEAN'].cumsum()
            # cal_mean_cum_df['Cum_MEAN_sigma_W'] = cal_mean_df['MEAN_sigma_W'].cumsum(skipna=True)
            # cal_mean_cum_df['Cum_MEAN_dist_W'] = cal_mean_df['MEAN_dist_W'].cumsum(skipna=True)
            # cal_mean_cum_df['Cum_MEAN_combined_W'] = cal_mean_df['MEAN_combined_W'].cumsum(skipna=True)

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

            reg_oce_df[fog_id] = cal_series_W1_W2_df.sum(axis=1, min_count=1)

        ### Save regional OCEs
        reg_oce_df.to_csv(region_oce_dir / f'{region}_regional_CEs.csv')
        reg_sig_dh_oce_df.to_csv(region_oce_dir / f'{region}_regional_sigma_dh_CEs.csv')
        reg_sig_rho_oce_df.to_csv(region_oce_dir / f'{region}_regional_sigma_rho_CEs.csv')
        reg_sig_anom_oce_df.to_csv(region_oce_dir / f'{region}_regional_sigma_anom_CEs.csv')


def calculate_regional_mass_balance(
    region_oce_dir: Path,
    regional_balance_dir: Path,
    regions: List[str],
    rgi_reg: dict,
    rgi_code: dict,
    begin_year: int,
    rgi_area_file: Path,
    glims_attribute_file: Path,
    rgi_attribute_dir: Path
) -> None:
    """
    Calculate the regional mass loss.
    """
    regional_balance_dir.mkdir(parents=True, exist_ok=True)

    id_rgi_area_df = pd.read_csv(rgi_area_file, dtype={'WGMS_ID': 'Int64'})
    id_glims_coords_df = pd.read_csv(
        glims_attribute_file, usecols=['glac_id', 'db_area', 'CenLat', 'CenLon']
    )
    # TODO: Avoid renaming
    id_glims_coords_df.rename(columns={'glac_id': 'RGIId'}, inplace=True)
    id_glims_coords_df.set_index('RGIId', inplace=True)

    ###### Calculate specific glacier mass balance by region ######

    Reg_mb_df = pd.DataFrame(columns=regions)
    Reg_sig_mb_df = pd.DataFrame(columns=regions)

    for region in regions:
        print(region)

        in_oce_file = region_oce_dir / f'{region}_regional_CEs.csv'
        in_sig_dh_oce_file = region_oce_dir / f'{region}_regional_sigma_dh_CEs.csv'
        in_sig_rho_oce_file = region_oce_dir / f'{region}_regional_sigma_rho_CEs.csv'
        in_sig_anom_oce_file = region_oce_dir / f'{region}_regional_sigma_anom_CEs.csv'

        oce_df = pd.read_csv(in_oce_file, index_col='YEAR')
        sig_anom_oce_df = pd.read_csv(in_sig_anom_oce_file, index_col='YEAR')
        sig_dh_oce_df = pd.read_csv(in_sig_dh_oce_file, index_col='YEAR')
        sig_rho_oce_df = pd.read_csv(in_sig_rho_oce_file, index_col='YEAR')
        sig_oce_df = (sig_dh_oce_df**2 + sig_rho_oce_df**2 + sig_anom_oce_df**2)**0.5

        # TODO: Why anomaly file and not the main file?
        yr = sig_anom_oce_df.first_valid_index()
        year_mask = oce_df.index >= yr
        null_years = oce_df.index[~year_mask]
        null_glaciers = oce_df.columns[sig_oce_df[year_mask].isnull().any()]
        # TODO: Can dropping be replaced with later indexing?
        # TODO: Why are null years not dropped from oce_df?
        for df in [sig_anom_oce_df, sig_dh_oce_df, sig_rho_oce_df, sig_oce_df]:
            df.drop(labels=null_years, inplace=True)
        for df in [oce_df, sig_anom_oce_df, sig_dh_oce_df, sig_rho_oce_df, sig_oce_df]:
            df.drop(columns=null_glaciers, inplace=True)
            # Convert columns to int
            df.columns = df.columns.astype(int)

        rgi_file = rgi_attribute_dir / f'{rgi_code[region]}_rgi60_{rgi_reg[region]}.csv'
        rgi_df = pd.read_csv(rgi_file, usecols=['RGIId', 'CenLat', 'CenLon', 'Connect'], index_col='RGIId')
        if region == 'GRL':
            rgi_df = rgi_df[rgi_df['Connect'] != 2]
            l1l2 = rgi_df.index

        # Keep only glaciers in the region
        if region == 'SA1':
            rgi_area_df = id_rgi_area_df[id_rgi_area_df['GLACIER_SUBREGION_CODE'] == 'SAN-01']
        elif region == 'SA2':
            rgi_area_df = id_rgi_area_df[id_rgi_area_df['GLACIER_SUBREGION_CODE'] == 'SAN-02']
        elif region == 'GRL':
            rgi_area_df = id_rgi_area_df[id_rgi_area_df['RGIId'].isin(l1l2)]
        else:
            rgi_area_df = id_rgi_area_df[id_rgi_area_df['GLACIER_REGION_CODE'] == region]

        nb_gla_reg = rgi_area_df.shape[0]
        tot_area_rgi_reg = rgi_area_df['AREA'].sum()

        ## select wgms_ids belonging to the region group
        wgms_id_lst = oce_df.columns.to_list()

        ## Calculate total area of observed glaciers presenting an area value in FoG

        ## Remove glacier IDS with no Area, only for FoG areas
        rgi_area_df = rgi_area_df.set_index('WGMS_ID')
        id_lst = [i for i in wgms_id_lst if i in rgi_area_df.index]
        gla_obs_df = rgi_area_df.loc[id_lst]
        tot_area_obs = gla_obs_df['AREA'].sum()
        nb_gla_obs = gla_obs_df.shape[0]
        gla_obs_df = gla_obs_df.reset_index().set_index('RGIId')

        if region == 'CAU':
            tot_area_rgi_reg = id_glims_coords_df['db_area'].sum()
            gla_obs_area_coord_df = pd.merge(gla_obs_df, id_glims_coords_df, left_index=True, right_index=True).drop_duplicates()
        else:
            gla_obs_area_coord_df = pd.merge(gla_obs_df, rgi_df, left_index=True, right_index=True)

        gla_obs_area_coord_df = gla_obs_area_coord_df.reset_index().set_index('WGMS_ID')
        gla_obs_area_coord_df = gla_obs_area_coord_df[~gla_obs_area_coord_df.index.duplicated()]

        print('Area observed (km2):', f'{round(tot_area_obs)} / {round(tot_area_rgi_reg)}')
        print('Glaciers observed (#):', f'{nb_gla_obs} / {nb_gla_reg}')
        ####### Calculate all glaciers time series and uncertainties ##########

        ## 1. Calculate OCE series for unobserved glaciers as the Weigthed mean from the regional glacier sample with observations
        gla_obs_area_coord_df = gla_obs_area_coord_df.loc[id_lst]
        area_weight = gla_obs_area_coord_df['AREA'] / tot_area_obs
        rel_mb_df = oce_df.loc[:, id_lst] * area_weight
        rel_sig_dh_mb_df = sig_dh_oce_df.loc[:, id_lst] * area_weight
        rel_sig_rho_mb_df = sig_rho_oce_df.loc[:, id_lst] * area_weight
        rel_sig_anom_mb_df = sig_anom_oce_df.loc[:, id_lst] * area_weight
        rel_sig_mb_df = np.sqrt(rel_sig_dh_mb_df**2 + rel_sig_rho_mb_df**2 + rel_sig_anom_mb_df**2)
        list_lat = gla_obs_area_coord_df['CenLat'].to_list()
        list_lon = gla_obs_area_coord_df['CenLon'].to_list()

        # Area-weighted OCE for observed glaciers
        Aw_oce_obs = rel_mb_df.sum(axis=1, min_count=1)

        ## 2. Calculate OCE uncertainties for observed glaciers

        # Weighted mean Sigma OCE of observed glaciers (only to use later for unobserved glaciers)
        Sig_oce_obs_gla = rel_sig_mb_df.sum(axis=1, min_count=1)
        Sig_dh_mb_gla = rel_sig_dh_mb_df.sum(axis=1, min_count=1)
        Sig_rho_mb_gla = rel_sig_rho_mb_df.sum(axis=1, min_count=1)
        Sig_anom_mb_gla = rel_sig_anom_mb_df.sum(axis=1, min_count=1)

        ## 3. Add OCE series and uncertainties for unobserved glaciers

        # Id -9999 for unobserved glaciers, OCE is the area weighted average of the regional observed series

        out_oce = regional_balance_dir / 'spt_CEs_obs-unobs_per_region'
        out_oce.mkdir(parents=True, exist_ok=True)

        oce_df['unobs_gla'] = Aw_oce_obs
        oce_df.to_csv(out_oce / f'{region}_CEs_obs-unobs.csv')

        sig_oce_df['unobs_gla'] = Sig_oce_obs_gla
        sig_oce_df.to_csv(out_oce / f'{region}_sigma_tot_CEs_obs-unobs.csv')

        sig_rho_oce_df['unobs_gla'] = Sig_rho_mb_gla
        sig_rho_oce_df.to_csv(out_oce / f'{region}_sig_rho_CEs_obs-unobs.csv')
        sig_anom_oce_df['unobs_gla'] = Sig_anom_mb_gla
        sig_anom_oce_df.to_csv(out_oce / f'{region}_sigma_anom_CEs_obs-unobs.csv')
        sig_dh_oce_df['unobs_gla'] = Sig_dh_mb_gla
        sig_dh_oce_df.to_csv(out_oce / f'{region}_sigma_dh_CEs_obs-unobs.csv')

        ####### Calculate Regional specific mass balance time series ##########

        Reg_mb_df[region] = Aw_oce_obs

        # We can't apply to the whole YEAR/ID dataframe at once here, we need to loop for each YEAR of the dataframes
        # to compute the pairwise error propagation for dh and density across all glaciers of that year
        list_sig_dh_yearly = []
        list_sig_rho_yearly = []
        list_sig_anom_yearly = []

        # Spatial correlation for rho for a 1-year period
        def sig_rho_dv_spatialcorr_yearly(d):
            return propagation.sig_rho_dv_spatialcorr(d, dt=1)

        # !! Project coordinates outside loop
        # Get median latitude and longitude among all values
        med_lat = np.median(list_lat)
        med_lon = np.median(list_lon)
        # Find the metric (UTM) system centered on these coordinates
        utm_zone = propagation.latlon_to_utm(lat=med_lat, lon=med_lon)
        epsg = propagation.utm_to_epsg(utm_zone)
        # Reproject latitude and longitude to easting/northing
        easting, northing = propagation.reproject_from_latlon(
            [list_lat, list_lon], out_crs=pyproj.CRS.from_epsg(epsg)
        )
        coords = np.column_stack([easting, northing])

        for year in rel_sig_dh_mb_df.index:
            print(year)

            # Spatial correlation for dh
            yearly_dh_errors = rel_sig_dh_mb_df.loc[year].values
            sig_dh_obs = propagation.double_sum_covar(
                coords=coords,
                errors=yearly_dh_errors,
                spatialcorr_func=propagation.sig_dh_spatialcorr
            )
            # # Check final estimate is between fully correlated and independent
            # sig_dh_fullcorr = np.sum(yearly_dh_errors)
            # sig_dh_uncorr = np.sqrt(np.sum(yearly_dh_errors**2))
            # print(f"{sig_dh_uncorr}, {sig_dh_obs}, {sig_dh_fullcorr}")
            # assert sig_dh_uncorr <= sig_dh_obs <= sig_dh_fullcorr

            # Spatial correlation for rho
            yearly_rho_errors = rel_sig_rho_mb_df.loc[year].values
            sig_rho_obs = propagation.double_sum_covar(
                coords=coords,
                errors=yearly_rho_errors,
                spatialcorr_func=sig_rho_dv_spatialcorr_yearly
            )
            # # Check propagation works as intended: final estimate is between fully correlated and independent
            # sig_rho_fullcorr = np.sum(yearly_rho_errors)
            # sig_rho_uncorr = np.sqrt(np.sum(yearly_rho_errors ** 2))
            # print(f"{sig_rho_uncorr}, {sig_rho_obs}, {sig_rho_fullcorr}")
            # assert sig_rho_uncorr <= sig_rho_obs <= sig_rho_fullcorr

            # Spatial correlation for anom
            yearly_anom_errors = rel_sig_anom_mb_df.loc[year].values
            sig_anom_obs = propagation.double_sum_covar(
                coords=coords,
                errors=yearly_anom_errors,
                spatialcorr_func=propagation.ba_anom_spatialcorr
            )
            # # Check propagation works as intended: final estimate is between fully correlated and independent
            # sig_anom_fullcorr = np.sum(yearly_anom_errors)
            # sig_anom_uncorr = np.sqrt(np.sum(yearly_anom_errors**2))
            # print(f"{sig_anom_uncorr}, {sig_anom_obs}, {sig_anom_fullcorr}")
            # assert sig_anom_uncorr <= sig_anom_obs <= sig_anom_fullcorr

            # Append to list for each yearly period
            list_sig_dh_yearly.append(sig_dh_obs)
            list_sig_rho_yearly.append(sig_rho_obs)
            list_sig_anom_yearly.append(sig_anom_obs)

        # Format results as indexed series
        # TODO: Why sig_anom_oce_df and not oce_df for index?
        Aw_sig_dh_obs = pd.Series(list_sig_dh_yearly, index=sig_anom_oce_df.index)
        Aw_sig_rho_obs = pd.Series(list_sig_rho_yearly, index=sig_anom_oce_df.index)
        Aw_sig_anom_obs = pd.Series(list_sig_anom_yearly, index=sig_anom_oce_df.index)
        Sig_oce_obs_propag = (Aw_sig_dh_obs**2 + Aw_sig_rho_obs**2 + Aw_sig_anom_obs**2)**0.5

        # Defining area-weighted uncertainty of unobserved glaciers based on the mean uncertainty of observed glaciers
        area_unobs = round(tot_area_rgi_reg, 2) - round(tot_area_obs, 2)
        sig_W_unobs = Sig_oce_obs_gla * (area_unobs / tot_area_rgi_reg)

        # Area-weight the observed glaciers before combining in final uncertainty
        sig_W_obs = Sig_oce_obs_propag * (tot_area_obs / tot_area_rgi_reg)

        # Final regional uncertainty
        reg_sig = np.sqrt(sig_W_obs**2 + sig_W_unobs**2)
        Reg_sig_mb_df[region] = reg_sig

        pd.DataFrame({
            'Aw_B m w.e.': Aw_oce_obs / 1000,
            'sigma_B m w.e.': reg_sig / 1000,
            'sigma_propagated m w.e.': Sig_oce_obs_propag / 1000,
            'sigma_dh m w.e.': Aw_sig_dh_obs / 1000,
            'sigma_rho m w.e.': Aw_sig_rho_obs / 1000,
            'sigma_anom m w.e.': Aw_sig_anom_obs / 1000
        }).to_csv(regional_balance_dir / f'{region}_B_and_sigma.csv')

    Reg_mb_df = Reg_mb_df[Reg_mb_df.index >= begin_year] / 1000
    Reg_sig_mb_df = Reg_sig_mb_df[Reg_sig_mb_df.index >= begin_year] / 1000
    # Save regional Mass balance series
    Reg_mb_df.to_csv(regional_balance_dir / 'Regional_B_series_AreaWeighted_code.csv')
    Reg_sig_mb_df.to_csv(regional_balance_dir / 'Regional_B_series_uncertainty_code.csv')

    # Compile files into single csvs
    # !! Moved from start of function, before the necessary files exist
    # TODO: Unclear why this is necessary.
    reg_mb_lst = []
    reg_sig_mb_lst = []

    for region in regions:
        in_data = regional_balance_dir / f'{region}_B_and_sigma.csv'
        data_df = pd.read_csv(in_data, index_col='YEAR')
        mb_df = pd.DataFrame(data_df['Aw_B m w.e.'])
        mb_df = mb_df.rename(columns={'Aw_B m w.e.': region})
        sig_df = pd.DataFrame(data_df['sigma_B m w.e.'])
        sig_df = sig_df.rename(columns={'sigma_B m w.e.': region})
        reg_mb_lst.append(mb_df)
        reg_sig_mb_lst.append(sig_df)

    reg_mb_df = pd.concat(reg_mb_lst, axis=1)
    reg_sig_mb_df = pd.concat(reg_sig_mb_lst, axis=1)

    ### Save regional Mass balance series
    reg_mb_df.to_csv(regional_balance_dir / 'Regional_B_series_AreaWeighted.csv')
    reg_sig_mb_df.to_csv(regional_balance_dir / 'Regional_B_series_uncertainty.csv')


def calculate_regional_mass_balance_essd(
    regional_balance_dir: Path,
    rgi_code: Dict[str, str],
    rgi_region: Dict[str, str],
    glacier_id_lut_file: Path,
    glims_attribute_file: Path,
    rgi_attribute_dir: Path,
    regional_balance_essd_dir: Path,
    regions: List[str],
    runs: List[str] = ['cal_series', 'error_dh', 'error_anom', 'error_rho', 'error_tot']
) -> None:
    regional_balance_essd_dir.mkdir(parents=True, exist_ok=True)

    id_link_df = pd.read_csv(glacier_id_lut_file)
    id_glims_coords_df = pd.read_csv(glims_attribute_file, usecols= ['glac_id', 'CenLat', 'CenLon', 'Area'])
    id_glims_coords_df.rename(columns={'glac_id': 'GLIMS_ID'}, inplace=True)
    id_glims_coords_df.set_index('GLIMS_ID', inplace=True)

    ###### Calculate specific glacier mass balance by region ######

    oce_dir = regional_balance_dir / 'spt_CEs_obs-unobs_per_region'
    for region in regions:
        print(region)

        # Regional OCE series and three sources of uncertainty
        in_oce_file = oce_dir / f'{region}_CEs_obs-unobs.csv'
        in_sig_dh_oce_file = oce_dir / f'{region}_sigma_dh_CEs_obs-unobs.csv'
        in_sig_rho_oce_file = oce_dir / f'{region}_sig_rho_CEs_obs-unobs.csv'
        in_sig_anom_oce_file = oce_dir / f'{region}_sigma_anom_CEs_obs-unobs.csv'
        in_sig_tot_oce_file = oce_dir / f'{region}_sigma_tot_CEs_obs-unobs.csv'

        oce_df = pd.read_csv(in_oce_file, index_col='YEAR')
        sig_anom_oce_df = pd.read_csv(in_sig_anom_oce_file, index_col='YEAR')
        sig_dh_oce_df = pd.read_csv(in_sig_dh_oce_file, index_col='YEAR')
        sig_rho_oce_df = pd.read_csv(in_sig_rho_oce_file, index_col='YEAR')
        sig_tot_oce_df = pd.read_csv(in_sig_tot_oce_file, index_col='YEAR')

        yr = oce_df.first_valid_index()
        year_mask = oce_df.index >= yr
        null_years = oce_df.index[~year_mask]
        for df in [oce_df, sig_anom_oce_df, sig_dh_oce_df, sig_rho_oce_df, sig_tot_oce_df]:
            df.drop(labels=null_years.intersection(df.index), inplace=True)

        oce_unobs_df = oce_df[['unobs_gla']].rename(columns={'unobs_gla': 'UNOBSERVED_GLACIERS'})
        sig_anom_unobs_df = sig_anom_oce_df[['unobs_gla']].rename(columns={'unobs_gla': 'UNOBSERVED_GLACIERS'})
        sig_dh_unobs_df = sig_dh_oce_df[['unobs_gla']].rename(columns={'unobs_gla': 'UNOBSERVED_GLACIERS'})
        sig_rho_unobs_df = sig_rho_oce_df[['unobs_gla']].rename(columns={'unobs_gla': 'UNOBSERVED_GLACIERS'})
        sig_tot_unobs_df = sig_tot_oce_df[['unobs_gla']].rename(columns={'unobs_gla': 'UNOBSERVED_GLACIERS'})

        for df in [oce_df, sig_anom_oce_df, sig_dh_oce_df, sig_rho_oce_df, sig_tot_oce_df]:
            df.drop(columns=['unobs_gla'], inplace=True)

        nan_lst = sig_tot_oce_df.columns[sig_tot_oce_df.isna().any()].tolist()

        sig_tot_oce_df = sig_tot_oce_df.drop(columns = nan_lst)
        oce_df = oce_df.drop(columns = nan_lst)
        sig_rho_oce_df = sig_rho_oce_df.drop(columns = nan_lst)
        sig_anom_oce_df = sig_anom_oce_df.drop(columns = nan_lst)
        sig_dh_oce_df = sig_dh_oce_df.drop(columns = nan_lst)

        oce_T_df = oce_df.transpose()
        oce_T_df.index.name = 'WGMS_ID'
        oce_T_df = oce_T_df.reset_index()
        oce_T_df['WGMS_ID'] = oce_T_df['WGMS_ID'].astype(int)

        sig_anom_T_df = sig_anom_oce_df.transpose()
        sig_anom_T_df.index.name = 'WGMS_ID'
        sig_anom_T_df = sig_anom_T_df.reset_index()
        sig_anom_T_df['WGMS_ID'] = sig_anom_T_df['WGMS_ID'].astype(int)

        sig_dh_T_df = sig_dh_oce_df.transpose()
        sig_dh_T_df.index.name = 'WGMS_ID'
        sig_dh_T_df = sig_dh_T_df.reset_index()
        sig_dh_T_df['WGMS_ID'] = sig_dh_T_df['WGMS_ID'].astype(int)

        sig_rho_T_df = sig_rho_oce_df.transpose()
        sig_rho_T_df.index.name = 'WGMS_ID'
        sig_rho_T_df = sig_rho_T_df.reset_index()
        sig_rho_T_df['WGMS_ID'] = sig_rho_T_df['WGMS_ID'].astype(int)

        sig_tot_T_df = sig_tot_oce_df.transpose()
        sig_tot_T_df.index.name = 'WGMS_ID'
        sig_tot_T_df = sig_tot_T_df.reset_index()
        sig_tot_T_df['WGMS_ID'] = sig_tot_T_df['WGMS_ID'].astype(int)

        # Create a file with all RGI glaciers
        filename = rgi_attribute_dir / f'{rgi_code[region]}_rgi60_{rgi_region[region]}.csv'
        if region == 'GRL':
            rgi_df_all = pd.read_csv(filename, encoding='latin1', usecols=['RGIId', 'CenLat', 'CenLon', 'Area', 'Connect'], index_col='RGIId')
            rgi_df = rgi_df_all.loc[rgi_df_all['Connect'] != 2]
            rgi_df['REGION'] = region
            first_column = rgi_df.pop('REGION')
            rgi_df.insert(0, 'REGION', first_column)

        elif region == 'SA1':
            rgi_df = pd.read_csv(filename, encoding='latin1', usecols=['RGIId', 'CenLat', 'CenLon', 'Area', 'O2Region'], index_col='RGIId')
            rgi_df = rgi_df[rgi_df['O2Region'] == 1]
            rgi_df['REGION'] = 'SAN'
            rgi_df.insert(0, 'REGION', rgi_df.pop('REGION'))
            rgi_df.insert(1, 'O2Region', rgi_df.pop('O2Region'))

        elif region == 'SA2':
            rgi_df = pd.read_csv(filename, encoding='latin1', usecols=['RGIId', 'CenLat', 'CenLon', 'Area', 'O2Region'], index_col='RGIId')
            rgi_df = rgi_df[rgi_df['O2Region'] == 2]
            rgi_df['REGION'] = 'SAN'
            first_column = rgi_df.pop('REGION')
            rgi_df.insert(0, 'REGION', first_column)
            rgi_df.insert(1, 'O2Region', rgi_df.pop('O2Region'))

        else:
            rgi_df = pd.read_csv(filename, encoding='latin1', usecols=['RGIId', 'CenLat', 'CenLon', 'Area'], index_col='RGIId')
            rgi_df['REGION'] = region
            first_column = rgi_df.pop('REGION')
            rgi_df.insert(0, 'REGION', first_column)

        rgi_df.reset_index(inplace=True)
        id_glims_coords_df.reset_index(inplace=True)

        if 'cal_series' in runs:
            if region == 'CAU':
                reg_id_df = id_glims_coords_df.merge(id_link_df.drop_duplicates(subset=['GLIMS_ID']), how='left')
                reg_id_df = reg_id_df.drop(columns='RGIId')
                reg_oce_df = reg_id_df.merge(oce_T_df, on='WGMS_ID', how='left').set_index('GLIMS_ID')
                reg_oce_df['WGMS_ID'].fillna('no_WGMS_ID', inplace=True)
                reg_oce_df.fillna('no_obs', inplace=True)
            else:
                reg_id_df = rgi_df.merge(id_link_df.drop_duplicates(subset=['WGMS_ID']), how='left')
                reg_id_df = reg_id_df.drop(columns='GLIMS_ID')
                reg_oce_df = reg_id_df.merge(oce_T_df, on='WGMS_ID', how='left').set_index('RGIId')
                reg_oce_df['WGMS_ID'].fillna('no_WGMS_ID', inplace=True)
                reg_oce_df.fillna('no_obs', inplace=True)

            # Replace "no_obs" in target rows with unobserved data
            for year in range(oce_unobs_df.first_valid_index(), oce_unobs_df.last_valid_index() + 1):
                reg_oce_df.loc[(reg_oce_df[year] == 'no_obs'),year] = oce_unobs_df.loc[year, 'UNOBSERVED_GLACIERS']

            reg_oce_df.to_csv(regional_balance_essd_dir / f'{region}_gla_MEAN-CAL-mass-change-series_obs_unobs.csv')

        ### Error dh
        if 'error_dh' in runs:
            if region == 'CAU':
                reg_id_df = id_glims_coords_df.merge(id_link_df.drop_duplicates(subset=['GLIMS_ID']), how='left')
                reg_id_df = reg_id_df.drop(columns='RGIId')
                reg_sig_dh_df = reg_id_df.merge(sig_dh_T_df, on='WGMS_ID', how='left').set_index('GLIMS_ID')
                reg_sig_dh_df['WGMS_ID'].fillna('no_WGMS_ID', inplace=True)
                reg_sig_dh_df.fillna('no_obs', inplace=True)
            else:
                reg_id_df = rgi_df.merge(id_link_df.drop_duplicates(subset=['WGMS_ID']), how='left')
                reg_id_df = reg_id_df.drop(columns='GLIMS_ID')
                reg_sig_dh_df = reg_id_df.merge(sig_dh_T_df, on='WGMS_ID', how='left').set_index('RGIId')
                reg_sig_dh_df['WGMS_ID'].fillna('no_WGMS_ID', inplace=True)
                reg_sig_dh_df.fillna('no_obs', inplace=True)

            # Replace "no_obs" in target rows with unobserved data
            for year in range(sig_dh_unobs_df.first_valid_index(), sig_dh_unobs_df.last_valid_index() + 1):
                reg_sig_dh_df.loc[(reg_sig_dh_df[year] == 'no_obs'),
                    year] = sig_dh_unobs_df.loc[year, 'UNOBSERVED_GLACIERS']

            reg_sig_dh_df.to_csv(regional_balance_essd_dir / f'{region}_gla_mean-cal-mass-change_DH-ERROR_obs_unobs.csv')

        ### Error anom
        if 'error_anom' in runs:
            if region == 'CAU':
                reg_id_df = id_glims_coords_df.merge(id_link_df.drop_duplicates(subset=['GLIMS_ID']), how='left')
                reg_id_df = reg_id_df.drop(columns='RGIId')
                reg_sig_anom_df = reg_id_df.merge(sig_anom_T_df, on='WGMS_ID', how='left').set_index('GLIMS_ID')
                reg_sig_anom_df['WGMS_ID'].fillna('no_WGMS_ID', inplace=True)
                reg_sig_anom_df.fillna('no_obs', inplace=True)
            else:
                reg_id_df = rgi_df.merge(id_link_df.drop_duplicates(subset=['WGMS_ID']), how='left')
                reg_id_df = reg_id_df.drop(columns='GLIMS_ID')
                reg_sig_anom_df = reg_id_df.merge(sig_anom_T_df, on='WGMS_ID', how='left').set_index('RGIId')
                reg_sig_anom_df['WGMS_ID'].fillna('no_WGMS_ID', inplace=True)
                reg_sig_anom_df.fillna('no_obs', inplace=True)

            # Replace "no_obs" in target rows with unobserved data
            for year in range(sig_anom_unobs_df.first_valid_index(), sig_anom_unobs_df.last_valid_index() + 1):
                reg_sig_anom_df.loc[(reg_sig_anom_df[year] == 'no_obs'), year] = sig_anom_unobs_df.loc[year, 'UNOBSERVED_GLACIERS']

            reg_sig_anom_df.to_csv(regional_balance_essd_dir / f'{region}_gla_mean-cal-mass-change_ANOM-ERROR_obs_unobs.csv')

        ### Error rho
        if 'error_rho' in runs:
            if region == 'CAU':
                reg_id_df = id_glims_coords_df.merge(id_link_df.drop_duplicates(subset=['GLIMS_ID']), how='left')
                reg_id_df = reg_id_df.drop(columns='RGIId')
                reg_sig_rho_df = reg_id_df.merge(sig_rho_T_df, on='WGMS_ID', how='left').set_index('GLIMS_ID')
                reg_sig_rho_df['WGMS_ID'].fillna('no_WGMS_ID', inplace=True)
                reg_sig_rho_df.fillna('no_obs', inplace=True)
            else:
                reg_id_df = rgi_df.merge(id_link_df.drop_duplicates(subset=['WGMS_ID']), how='left')
                reg_id_df = reg_id_df.drop(columns='GLIMS_ID')
                reg_sig_rho_df = reg_id_df.merge(sig_rho_T_df, on='WGMS_ID', how='left').set_index('RGIId')
                reg_sig_rho_df['WGMS_ID'].fillna('no_WGMS_ID', inplace=True)
                reg_sig_rho_df.fillna('no_obs', inplace=True)

            # Replace "no_obs" in target rows with unobserved data
            for year in range(sig_rho_unobs_df.first_valid_index(), sig_rho_unobs_df.last_valid_index() + 1):
                reg_sig_rho_df.loc[(reg_sig_rho_df[year] == 'no_obs'),
                    year] = sig_rho_unobs_df.loc[year, 'UNOBSERVED_GLACIERS']

            reg_sig_rho_df.to_csv(regional_balance_essd_dir / f'{region}_gla_mean-cal-mass-change_RHO-ERROR_obs_unobs.csv')

        ### Error tot
        if 'error_tot' in runs:
            if region == 'CAU':
                reg_id_df = id_glims_coords_df.merge(id_link_df.drop_duplicates(subset=['GLIMS_ID']), how='left')
                reg_id_df = reg_id_df.drop(columns='RGIId')
                reg_sig_tot_df = reg_id_df.merge(sig_tot_T_df, on='WGMS_ID', how='left').set_index('GLIMS_ID')
                reg_sig_tot_df['WGMS_ID'].fillna('no_WGMS_ID', inplace=True)
                reg_sig_tot_df.fillna('no_obs', inplace=True)
            else:
                reg_id_df = rgi_df.merge(id_link_df.drop_duplicates(subset=['WGMS_ID']), how='left')
                reg_id_df = reg_id_df.drop(columns='GLIMS_ID')
                reg_sig_tot_df = reg_id_df.merge(sig_tot_T_df, on='WGMS_ID', how='left').set_index('RGIId')
                reg_sig_tot_df['WGMS_ID'].fillna('no_WGMS_ID', inplace=True)
                reg_sig_tot_df.fillna('no_obs', inplace=True)

            # Replace "no_obs" in target rows with unobserved data
            for year in range(sig_tot_unobs_df.first_valid_index(), sig_tot_unobs_df.last_valid_index() + 1):  # Loop through years 1946-1953
                reg_sig_tot_df.loc[(reg_sig_tot_df[year] == 'no_obs'), year] = sig_tot_unobs_df.loc[year, 'UNOBSERVED_GLACIERS']

            reg_sig_tot_df.to_csv(regional_balance_essd_dir / f'{region}_gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs.csv')


def calculate_regional_mass_loss(
    regional_balance_dir: Path,
    region_oce_dir: Path,
    regional_area_change_file: Path,
    rgi_area_file: Path,
    zemp_regional_series_dir: Path,
    ini_yr: int,
    fin_yr: int,
    regions: List[str],
    rgi_code: dict,
    rgi_reg: dict,
    mass_loss_dir: Path
) -> None:
    """
    Calculate the regional mass loss.
    """
    mass_loss_dir.mkdir(parents=True, exist_ok=True)

    # period to calculate the cumulative mass loss
    ini_yr_full_obs = ini_yr
    fin_yr_obs = fin_yr

    # FULL period
    PoR = list(range(ini_yr_full_obs, fin_yr_obs + 1))
    PoR_full = list(range(ini_yr_full_obs, fin_yr_obs + 1))

    # TODO: Move to parameters
    S_ocean = 362.5 * 10**6 # Cogley et al 2012
    sig_area = 0.05 # Paul et al 2015


    #################################################################################################
    ##    READ input files
    #################################################################################################

    area_weighted_file = regional_balance_dir / 'Regional_B_series_AreaWeighted.csv'
    uncertainty_file = regional_balance_dir / 'Regional_B_series_uncertainty.csv'
    ba_df = pd.read_csv(area_weighted_file, index_col='YEAR')
    sig_ba_df = pd.read_csv(uncertainty_file, index_col='YEAR')

    reg_area_zemp_df = pd.read_csv(regional_area_change_file, index_col='YEAR')
    id_area_df = pd.read_csv(rgi_area_file)

    ############################################################################################################################

    ###### Calculate total glacier mass loss by region ######

    cols = ['region', 'area_mean_' + str(ini_yr_full_obs) +'-' + str(fin_yr_obs) + ' [km2]', 'area_mean_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [km2]',
            'percentage_area_obs' ,'DM [Gt yr-1]', 'sigma_DM [Gt yr-1]', 'CUM_DM_'+str(min(PoR))+'_'+str(max(PoR))+' [Gt]', 'sigma_CUM_DM_' + str(min(PoR)) + '_' + str(max(PoR)) + ' [Gt]',
            'B [mwe yr-1]', 'sigma_B [mwe yr-1]',  'SLE [mm yr-1]', 'sigma_SLE [mm yr-1]', 'zemp_DM [Gt yr-1]', 'zemp_sigma_DM [Gt yr-1]',
            'zemp_CUM_DM_'+str(min(PoR))+'_'+str(max(PoR))+' [Gt]']


    glob_cum_df = pd.DataFrame(index=regions, columns=cols)

    Reg_DM_df = pd.DataFrame()
    Reg_sig_DM_df = pd.DataFrame()

    for region in regions:
        print(region)

        out_DM_series = mass_loss_dir / 'regional_mass_loss_series'
        out_DM_series.mkdir(parents=True, exist_ok=True)

        in_oce_file = region_oce_dir / f"{region}_regional_CEs.csv"
        oce_df = pd.read_csv(in_oce_file, index_col='YEAR')

        if region == 'SA1':
            rgi_area_df = id_area_df.loc[(id_area_df['GLACIER_REGION_CODE'] == 'SAN')].set_index('WGMS_ID')
            rgi_area_df = rgi_area_df.loc[rgi_area_df['GLACIER_SUBREGION_CODE']== 'SAN-01']
        elif region == 'SA2':
            rgi_area_df = id_area_df.loc[(id_area_df['GLACIER_REGION_CODE'] == 'SAN')].set_index('WGMS_ID')
            rgi_area_df = rgi_area_df.loc[rgi_area_df['GLACIER_SUBREGION_CODE']== 'SAN-02']
        else:
            rgi_area_df = id_area_df.loc[id_area_df['GLACIER_REGION_CODE'] == region].set_index('WGMS_ID')

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

        reg_file.to_csv(Path(out_DM_series, 'results_region_' + rgi_code[region] + '_' + region + '_' + rgi_reg[region] + '.csv'))

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

        if region in ('SA1', 'SA2'):
            file_region = 'SAN'
        else:
            file_region = region
        in_regional_series_df = zemp_regional_series_dir / f'Zemp_etal_results_region_{str(rgi_code[region]).lstrip("0")}_{file_region}.csv'


        zemp_df = pd.read_csv(in_regional_series_df, header=26, index_col='Year')
        zemp_2019_df = zemp_df[[' INT_Gt', ' sig_Int_Gt', ' sig_Total_Gt']]
        zemp_2019_df.index.name = 'Year'
        zemp_2019_df.columns = ['MB_Gt', 'sig_Int_Gt', 'MB_sigma_Gt']

        zemp_1976_2016 = zemp_2019_df['MB_Gt'].loc[(zemp_2019_df.index.isin(PoR))].sum() / len(PoR)
        zemp_sig_1976_2016 = np.sqrt(zemp_2019_df['MB_sigma_Gt'].loc[zemp_2019_df.index.isin(PoR)].pow(2).sum() / len(PoR))
        zemp_cum_1976_2016 = zemp_2019_df['MB_Gt'].loc[(zemp_2019_df.index.isin(PoR))].sum()


        for index, row in glob_cum_df.iterrows():
            if index == region:
                row['region']= rgi_reg[region]
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
    Reg_B_cum_df.to_csv(mass_loss_dir / 'Cumulative_Regional_Bmwe_series.csv')

    Reg_DM_df.to_csv(mass_loss_dir / 'Regional_DM_series.csv')
    Reg_DM_cum_df.to_csv(mass_loss_dir / 'Cumulative_Regional_DM_series.csv')
    Reg_sig_DM_df.to_csv(mass_loss_dir / 'Regional_DM_series_uncertainty.csv')

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

    glob_df.to_csv(mass_loss_dir / f"Global_DM_series_year_{ini_yr_full_obs}-{fin_yr_obs}.csv")
    glob_cum_df.to_csv(mass_loss_dir / f"Cum_DM_Gt_per_region_PoR_{min(PoR)}_{max(PoR)}.csv", index=False)
