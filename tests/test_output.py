from typing import Tuple, Union

import pandas as pd


def assert_dataframes_equal(
    old: Union[pd.DataFrame, str],
    new: Union[pd.DataFrame, str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(old, str):
        old = pd.read_csv(old)
    if isinstance(new, str):
        new = pd.read_csv(new)
    for df in old, new:
        if df.index.name is None:
            df.reset_index(inplace=True, drop=True)
        df.sort_index(axis=0, inplace=True)
        df.sort_index(axis=1, inplace=True)
        df.dropna(axis='rows', how='all', inplace=True)
        df.dropna(axis='columns', how='all', inplace=True)
    try:
        pd.testing.assert_frame_equal(
            left=old,
            right=new,
            check_dtype=False,
            check_index_type=False,
            check_exact=False,
            atol=1e-5
        )
    except Exception as error:
        print(str(error))
    return old, new


# ---- 2 (in), 1 (out) ----

# ba [equal]
old, new = assert_dataframes_equal(
    'data/input/2_Kriging_spatial_anomalies/fog_2025-01_ba.csv',
    'data/_output/ba.csv'
)

# ba_unc [equal]
old, new = assert_dataframes_equal(
    'data/input/2_Kriging_spatial_anomalies/fog_2025-01_ba_unc.csv',
    'data/_output/ba_unc.csv'
)

# bs [equal]
old, new = assert_dataframes_equal(
    'data/input/2_Kriging_spatial_anomalies/fog_2025-01_bs.csv',
    'data/_output/bs.csv'
)

# bw [equal]
old, new = assert_dataframes_equal(
    'data/input/2_Kriging_spatial_anomalies/fog_2025-01_bw.csv',
    'data/_output/bw.csv'
)

# FOG_coord [equal]
old, new = assert_dataframes_equal(
    'data/input/2_Kriging_spatial_anomalies/FOG_coord_2025-01.csv',
    'data/_output/FOG_coord_2025-01.csv'
)

# _FOG_GEO_MASS_BALANCE_DATA: 2 (in), 1 (out) [equal except for rows with null ID]
old, new = assert_dataframes_equal(
    'data/input/2_Kriging_spatial_anomalies/_FOG_GEO_MASS_BALANCE_DATA_2025-01.csv',
    'data/_output/_FOG_GEO_MASS_BALANCE_DATA_2025-01.csv'
)
_ = assert_dataframes_equal(
    old[old['ID'].notnull()].sort_values(['WGMS_ID', 'ID', 'ini_date', 'fin_date', 'elevation_chg_rate']),
    new.sort_values(['WGMS_ID', 'ID', 'ini_date', 'fin_date', 'elevation_chg_rate'])
)

# ---- 2 (out), 2 (out) ----

region = 'ISL'
name = 'Iceland'
code = '06'

# LONG-NORM: spt_anoms [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/2_Kriging_spatial_anomalies/out_data_2025-01/LONG-NORM_spatial_gla_anom_ref_2011-2020/{region}_spt_anoms_ref_2011-2020_2025-01.csv',
    f'data/_output/LONG-NORM_spatial_gla_anom/{region}_spt_anoms.csv'
)

# LONG-NORM: spt_ERRORs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/2_Kriging_spatial_anomalies/out_data_2025-01/LONG-NORM_spatial_gla_anom_ref_2011-2020/{region}_spt_ERRORs_ref_2011-2020_2025-01.csv',
    f'data/_output/LONG-NORM_spatial_gla_anom/{region}_spt_ERRORs.csv'
)

# LONG-NORM: spt_anoms_fill [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/2_Kriging_spatial_anomalies/out_data_2025-01/LONG-NORM_spatial_gla_anom_ref_2011-2020/{region}_spt_anoms_fill_ref_2011-2020_2025-01.csv',
    f'data/_output/LONG-NORM_spatial_gla_anom/{region}_spt_anoms_fill.csv'
)

# LONG-NORM: spt_ERRORs_fill [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/2_Kriging_spatial_anomalies/out_data_2025-01/LONG-NORM_spatial_gla_anom_ref_2011-2020/{region}_spt_ERRORs_fill_ref_2011-2020_2025-01.csv',
    f'data/_output/LONG-NORM_spatial_gla_anom/{region}_spt_ERRORs_fill.csv'
)

# LOOKUP: reg_gla [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/2_Kriging_spatial_anomalies/out_data_2025-01/LOOKUP_spatial_and_reg_ids_ref_2011-2020/{region}_all_reg_gla_anomalies.csv',
    f'data/_output/LOOKUP_spatial_and_reg_ids/{region}_all_reg_gla_anomalies.csv'
)

# LOOKUP: SEL_gla_UNC [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/2_Kriging_spatial_anomalies/out_data_2025-01/LOOKUP_spatial_and_reg_ids_ref_2011-2020/{region}_all_SEL_gla_anomalies_UNC.csv',
    f'data/_output/LOOKUP_spatial_and_reg_ids/{region}_all_SEL_gla_anomalies_UNC.csv'
)

# MEAN: spt_anoms [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/2_Kriging_spatial_anomalies/out_data_2025-01/MEAN_spatial_gla_anom_ref_2011-2020/{region}_spt_anoms_ref_2011-2020_2025-01.csv',
    f'data/_output/MEAN_spatial_gla_anom/{region}_spt_anoms.csv'
)

# MEAN: spt_ERRORs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/2_Kriging_spatial_anomalies/out_data_2025-01/MEAN_spatial_gla_anom_ref_2011-2020/{region}_spt_ERRORs_ref_2011-2020_2025-01.csv',
    f'data/_output/MEAN_spatial_gla_anom/{region}_spt_ERRORs.csv'
)

#  ---- 3 (out), 3 (out) ----

# regional [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/3_Kriging_global_CE_spatial_anomaly/out_data_2025-01/OCE_files_by_region/{region}_regional_CEs.csv',
    f'data/_output/OCE_files_by_region/{region}_regional_CEs.csv'
)

# sigma_anom [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/3_Kriging_global_CE_spatial_anomaly/out_data_2025-01/OCE_files_by_region/{region}_regional_sigma_anom_CEs.csv',
    f'data/_output/OCE_files_by_region/{region}_regional_sigma_anom_CEs.csv'
)

# sigma_dh [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/3_Kriging_global_CE_spatial_anomaly/out_data_2025-01/OCE_files_by_region/{region}_regional_sigma_dh_CEs.csv',
    f'data/_output/OCE_files_by_region/{region}_regional_sigma_dh_CEs.csv'
)

# sigma_rho [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/3_Kriging_global_CE_spatial_anomaly/out_data_2025-01/OCE_files_by_region/{region}_regional_sigma_rho_CEs.csv',
    f'data/_output/OCE_files_by_region/{region}_regional_sigma_rho_CEs.csv'
)

# ---- 4 (out) -> 4 (out) ----

# B_and_sigma [some columns different values, different years index]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01/{region}_B_and_sigma.csv',
    f'data/_output/regional_balance/{region}_B_and_sigma.csv'
)
columns = ['sigma_B m w.e.', 'sigma_rho m w.e.', 'sigma_propagated m w.e.']
_ = assert_dataframes_equal(
    old.drop(columns=columns).copy(),
    new.drop(columns=columns).copy()
)

# CEs_obs-unobs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01/spt_CEs_obs-unobs_per_region/{region}_CEs_obs-unobs.csv',
    f'data/_output/regional_balance/spt_CEs_obs-unobs_per_region/{region}_CEs_obs-unobs.csv'
)

# sig_rho_CEs_obs-unobs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01/spt_CEs_obs-unobs_per_region/{region}_sig_rho_CEs_obs-unobs.csv',
    f'data/_output/regional_balance/spt_CEs_obs-unobs_per_region/{region}_sig_rho_CEs_obs-unobs.csv'
)

# sigma_anom_CEs_obs-unobs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01/spt_CEs_obs-unobs_per_region/{region}_sigma_anom_CEs_obs-unobs.csv',
    f'data/_output/regional_balance/spt_CEs_obs-unobs_per_region/{region}_sigma_anom_CEs_obs-unobs.csv'
)

# sigma_dh_CEs_obs-unobs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01/spt_CEs_obs-unobs_per_region/{region}_sigma_dh_CEs_obs-unobs.csv',
    f'data/_output/regional_balance/spt_CEs_obs-unobs_per_region/{region}_sigma_dh_CEs_obs-unobs.csv'
)

# sigma_tot_CEs_obs-unobs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01/spt_CEs_obs-unobs_per_region/{region}_sigma_tot_CEs_obs-unobs.csv',
    f'data/_output/regional_balance/spt_CEs_obs-unobs_per_region/{region}_sigma_tot_CEs_obs-unobs.csv'
)

# Regional_B_series_AreaWeighted [missing regions]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01/Regional_B_series_AreaWeighted.csv',
    f'data/_output/regional_balance/Regional_B_series_AreaWeighted.csv'
)
_ = assert_dataframes_equal(
    old[['YEAR', region]].copy(),
    new[['YEAR', region]].copy()
)

# Regional_B_series_AreaWeighted_code [missing regions]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01/Regional_B_series_AreaWeighted_code.csv',
    f'data/_output/regional_balance/Regional_B_series_AreaWeighted_code.csv'
)
_ = assert_dataframes_equal(
    old[['YEAR', region]].copy(),
    new[['YEAR', region]].copy()
)

# Regional_B_series_uncertainty [missing regions]
old, new = assert_dataframes_equal(
    'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01/Regional_B_series_uncertainty.csv',
    'data/_output/regional_balance/Regional_B_series_uncertainty.csv'
)
old, new = assert_dataframes_equal(
    old[['YEAR', region]].copy(),
    new[['YEAR', region]].copy()
)

# Regional_B_series_uncertainty_code [missing regions]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01/Regional_B_series_uncertainty_code.csv',
    f'data/_output/regional_balance/Regional_B_series_uncertainty_code.csv'
)
old, new = assert_dataframes_equal(
    old[['YEAR', region]].copy(),
    new[['YEAR', region]].copy()
)

# ---- 4 (out), 4 (out) ESSD ----

# {region}_gla_mean-cal-mass-change_ANOM-ERROR_obs_unobs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01_Dussaillant_etal_format/{region}_gla_mean-cal-mass-change_ANOM-ERROR_obs_unobs.csv',
    f'data/_output/regional_balance_essd/{region}_gla_mean-cal-mass-change_ANOM-ERROR_obs_unobs.csv'
)

# {region}_gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01_Dussaillant_etal_format/{region}_gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs.csv',
    f'data/_output/regional_balance_essd/{region}_gla_mean-cal-mass-change_TOTAL-ERROR_obs_unobs.csv'
)

# {region}_gla_mean-cal-mass-change_ANOM-ERROR_obs_unobs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01_Dussaillant_etal_format/{region}_gla_mean-cal-mass-change_ANOM-ERROR_obs_unobs.csv',
    f'data/_output/regional_balance_essd/{region}_gla_mean-cal-mass-change_ANOM-ERROR_obs_unobs.csv'
)

# {region}_gla_mean-cal-mass-change_RHO-ERROR_obs_unobs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01_Dussaillant_etal_format/{region}_gla_mean-cal-mass-change_RHO-ERROR_obs_unobs.csv',
    f'data/_output/regional_balance_essd/{region}_gla_mean-cal-mass-change_RHO-ERROR_obs_unobs.csv'
)

# {region}_gla_MEAN-CAL-mass-change-series_obs_unobs [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/4_Kriging_regional_mass_balance/out_data_2025-01_Dussaillant_etal_format/{region}_gla_MEAN-CAL-mass-change-series_obs_unobs.csv',
    f'data/_output/regional_balance_essd/{region}_gla_MEAN-CAL-mass-change-series_obs_unobs.csv'
)

# ---- 5 (out), 5 (out) ----

# mass_loss/regional_mass_loss_series/results_region_06_ISL_Iceland.csv [equal]
old, new = assert_dataframes_equal(
    f'data/mb_data_crunching_local/5_Kriging_regional_mass_loss/out_data_2025-01/regional_mass_loss_series/results_region_{int(code)}_{region}_{name}.csv',
    f'data/_output/mass_loss/regional_mass_loss_series/results_region_{code}_{region}_{name}.csv'
)

# mass_loss/Cum_DM_Gt_per_region_PoR_1976_2024.csv [equal]
old, new = assert_dataframes_equal(
    'data/mb_data_crunching_local/5_Kriging_regional_mass_loss/out_data_2025-01/Cum_DM_Gt_per_region_PoR_1976_2024.csv',
    'data/_output/mass_loss/Cum_DM_Gt_per_region_PoR_1976_2024.csv'
)
_ = assert_dataframes_equal(
    old[old['region'].eq(name)].drop(columns=['Unnamed: 16', 'Unnamed: 17']),
    new[new['region'].eq(name)].copy()
)

# mass_loss/Cumulative_Regional_Bmwe_series.csv [equal]
old, new = assert_dataframes_equal(
    'data/mb_data_crunching_local/5_Kriging_regional_mass_loss/out_data_2025-01/Cumulative_Regional_Bmwe_series.csv',
    'data/_output/mass_loss/Cumulative_Regional_Bmwe_series.csv'
)
_ = assert_dataframes_equal(
    old[['YEAR', region]].copy(),
    new[['YEAR', region]].copy()
)

# mass_loss/Cumulative_Regional_DM_series.csv [equal]
old, new = assert_dataframes_equal(
    'data/mb_data_crunching_local/5_Kriging_regional_mass_loss/out_data_2025-01/Cumulative_Regional_DM_series.csv',
    'data/_output/mass_loss/Cumulative_Regional_DM_series.csv'
)
_ = assert_dataframes_equal(
    old[['YEAR', region]].copy(),
    new[['YEAR', region]].copy()
)

# mass_loss/Regional_DM_series_uncertainty.csv [equal]
old, new = assert_dataframes_equal(
    'data/mb_data_crunching_local/5_Kriging_regional_mass_loss/out_data_2025-01/Regional_DM_series_uncertainty.csv',
    'data/_output/mass_loss/Regional_DM_series_uncertainty.csv'
)
_ = assert_dataframes_equal(
    old[['YEAR', region]].copy(),
    new[['YEAR', region]].copy()
)

# mass_loss/Regional_DM_series.csv [equal]
old, new = assert_dataframes_equal(
    'data/mb_data_crunching_local/5_Kriging_regional_mass_loss/out_data_2025-01/Regional_DM_series.csv',
    'data/_output/mass_loss/Regional_DM_series.csv'
)
_ = assert_dataframes_equal(
    old[['YEAR', region]].copy(),
    new[['YEAR', region]].copy()
)

# mass_loss/Global_DM_series_year_1976-2024.csv
# old, new = assert_dataframes_equal(
#     'data/mb_data_crunching_local/5_Kriging_regional_mass_loss/out_data_2025-01/Global_DM_series_year_1976-2024.csv',
#     'data/_output/mass_loss/Global_DM_series_year_1976-2024.csv'
# )
