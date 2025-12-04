"""helpers"""
import numpy as np
import pandas as pd
from typing import List

def date_format(month: int,
                year: int):
    """
    A simple function to make decimal year.

    Parameters
    ----------
    month: int
    The integer value of an inputted month (i.e., values ranging from 1 to 12)
    year : int
    The integer value for the year

    Returns
    -------
    float
        A floating point decimal year number.

    """
    default_month = 6
    month=default_month if month > 12.0 else month
    offset = year + (1.0 / 12 * (month - 1))
    return offset


def cum_to_rate(elev_chg_cum: float,
                sur_date: float,
                ref_date: float):
    """
    A simple function to transform cumulative elevation changes to rate

    Parameters
    ----------
    elev_chg_cum: float
    Numeric value for the elevation change from the dataset (column "ELEVATION_CHANGE" in "FOG_ELEVATION_CHANGE_DATA_2025-01.csv")
    sur_date: float
    Floating point decimal year for the "SURVEY_DATE" column value of "FOG_ELEVATION_CHANGE_DATA_2025-01.csv" (formatted via the date_format() function)
    ref_date: float
    Floating point decimal year for the "REFERENCE_DATE" column value of "FOG_ELEVATION_CHANGE_DATA_2025-01.csv" (formatted via the date_format() function)

    Returns
    -------
    float
        A floating point decimal value for the elevation rate of change

    """
    if (sur_date-ref_date) != 0:
        elev_chg_rate = elev_chg_cum/(sur_date-ref_date) if elev_chg_cum != 0 else elev_chg_cum
        return elev_chg_rate
    else:
        return elev_chg_cum

def create_mb_dataframe(in_df: pd.DataFrame,
                        id_lst: List[str],
                        yr_lst: List[int],
                        mb_field: str) -> pd.DataFrame:
    """
    This function selects mass-balance data from the input file into a dataframe.

    Parameters
    ----------
    in_df: pd.DataFrame
        The input data frame
    id_lst: List[str]
        A list of WGMS id strings from the input dataset (i.e., from 'fog_bw-bs-ba_2025-01.csv')
    yr_lst: List[int]
        A list of integer years from the input dataset (i.e., from 'fog_bw-bs-ba_2025-01.csv')
    mb_field: str
        One string column name (from 'fog_bw-bs-ba_2025-01.csv') of:
        'ANNUAL_BALANCE'
        'SUMMER_BALANCE'
        'WINTER_BALANCE'
        'ANNUAL_BALANCE_UNC'

    Returns
    -------
    pd.DataFrame
        A formatted Pandas Dataframe

    """
    print('Creating dataframe of {}...'.format(mb_field))

    mb_dict = {}

    for id in id_lst:
        mb_lst = []
        for yr in yr_lst:
            if yr in in_df.loc[in_df['WGMS_ID'] == id]['YEAR'].values:
                mb_lst.append(
                    in_df[(in_df['WGMS_ID'] == id) & (in_df['YEAR'] == yr)].iloc[0][mb_field])
            else:
                mb_lst.append(np.nan)
        mb_dict.update({id: mb_lst})

    mb_df = pd.DataFrame(mb_dict, index=yr_lst)

    print('..done.')
    return mb_df


def calc_anomalies(in_df, ref_period, region):
    """This function calculates the anomalies of glacier mass balances over a defined reference period."""
    print('Calculating anomalies for reference period from {} to {}...'.format(min(ref_period), max(ref_period)))

    # create subset over reference period
    in_ref_df = in_df.loc[in_df.index.isin(ref_period)]

    # create subset with minimal data
    ref_period_df = in_df.loc[in_df.index.isin(ref_period)]
    if region == 'ASN':
        ref_period_ids = ref_period_df.count() > 3
    else:
        ref_period_ids = ref_period_df.count() > 7
    ref_period_id_lst = list(ref_period_ids[ref_period_ids].index)

    # create subset of glacier ids with good data over reference period
    good_ids_in_ref_df = in_ref_df[ref_period_id_lst]
    good_ids_in_df=in_df[ref_period_id_lst]

    # calculate anomaly (x_i-x_avg) for data over reference period
    avg_ref_df = good_ids_in_ref_df.mean()
    anomaly_ref_df = round(good_ids_in_df - avg_ref_df, 0)
    # print(anomaly_ref_df)
    print('done.')
    return anomaly_ref_df

def calc_anomalies_unc(in_df, in_unc_df, ref_period, region):
    """This function calculates the uncertainties of glacier mass balances series"""

    # create subset with minimal data
    ref_period_df = in_df.loc[in_df.index.isin(ref_period)]
    if region == 'ASN':
        ref_period_ids = ref_period_df.count() > 3
    else:
        ref_period_ids = ref_period_df.count() > 7
    ref_period_id_lst = list(ref_period_ids[ref_period_ids].index)

    # calculate mb uncertainty of glacier ids with good data over reference period
    unc_ref_df = in_unc_df[ref_period_id_lst]
    reg_unc_mean= np.nanmean(unc_ref_df)

    print(unc_ref_df)
    print(reg_unc_mean)

    for id in ref_period_id_lst:
        # id=0
        year_min = in_df[id].first_valid_index()
        yrs= list(range(1915,year_min))

        if unc_ref_df[id].isnull().all():
            unc_ref_df[id][id].fillna(reg_unc_mean, inplace=True)
        else:
            unc_ref_df[id].fillna(unc_ref_df[id].mean(), inplace=True)
        # unc_ref_df.loc[unc_ref_df.index.isin(yrs), [id]] = np.nan
        unc_ref_df[id].mask(unc_ref_df.index.isin(yrs), np.nan, inplace=True)

        print(unc_ref_df[id])
        # exit()
    print('done.')
    return unc_ref_df

def calc_spt_anomalies_unc(in_df, in_unc_df, id_lst):
    """This function calculates the uncertainties of glacier mass balances series"""

    # calculate mb uncertainty of glacier ids with good data over reference period
    unc_ref_df = in_unc_df[id_lst]
    reg_unc_mean= np.nanmean(unc_ref_df)

    for id in unc_ref_df.columns:
        year_min = in_df[id].first_valid_index()
        yrs= list(range(1915,year_min))

        if unc_ref_df[id].isnull().all():
            unc_ref_df[id].fillna(reg_unc_mean, inplace=True)
        else:
            unc_ref_df[id].fillna(unc_ref_df[id].mean(), inplace=True)

        unc_ref_df[id].mask(unc_ref_df.index.isin(yrs), np.nan, inplace=True)

    print('done.')
    return unc_ref_df

def dis_fil(row, ini_date, fin_date):
    if row > fin_date:
        return row - fin_date + 1
    if row <= ini_date:
        return ini_date + 2 - row
    else:
        return 1.0
