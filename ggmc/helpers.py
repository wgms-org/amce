"""helpers"""
from matplotlib import dates
import numpy as np
import pandas as pd
from typing import List


def wgms_date_to_decimal_year(date: pd.Series) -> pd.Series:
    """
    Convert WGMS date (yyyymmdd) to decimal year, ignoring day.

    Unknown month are represented as 99 and set to June (6).

    Example
    -------
    >>> date = pd.Series([20000115, 20010620, 20029999])
    >>> wgms_date_to_decimal_year(date)
    0    2000.000000
    1    2001.416667
    2    2002.416667
    dtype: float64
    """
    date = date.astype('string')
    years = date.str.slice(0, 4).astype(int)
    months = date.str.slice(4, 6).replace('99', '06').astype(int)
    return years + (1 / 12 * (months.fillna(6) - 1))


def change_to_rate(
    change: pd.Series,
    begin_date: pd.Series,
    end_date: pd.Series
) -> pd.Series:
    """
    Convert change to change rate.

    Parameters
    ----------
    change: float
        Change value.
    begin_date: float
        Begin date in decimal format.
    end_date: float
        End date in decimal format.

    Example
    -------
    >>> change = pd.Series([-10, 5, 1])
    >>> begin_date = pd.Series([2000.0, 2001.0, 2002.0])
    >>> end_date = pd.Series([2001.0, 2003.0, 2002.0])
    >>> change_to_rate(change, begin_date, end_date)
    0   -10.0
    1     2.5
    2     1.0
    dtype: float64
    """
    duration = end_date - begin_date
    return (change / duration).where(duration != 0, change)


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
    good_ids_in_df = in_df[ref_period_id_lst]

    # calculate anomaly (x_i-x_avg) for data over reference period
    avg_ref_df = good_ids_in_ref_df.mean()
    anomaly_ref_df = round(good_ids_in_df - avg_ref_df, 0)
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
    reg_unc_mean = np.nanmean(unc_ref_df)

    print(unc_ref_df)
    print(reg_unc_mean)

    for id in ref_period_id_lst:
        year_min = in_df[id].first_valid_index()
        yrs = list(range(1915, year_min))

        if unc_ref_df[id].isnull().all():
            unc_ref_df[id][id].fillna(reg_unc_mean, inplace=True)
        else:
            unc_ref_df[id].fillna(unc_ref_df[id].mean(), inplace=True)
        # unc_ref_df.loc[unc_ref_df.index.isin(yrs), [id]] = np.nan
        unc_ref_df[id].mask(unc_ref_df.index.isin(yrs), np.nan, inplace=True)

        print(unc_ref_df[id])

    return unc_ref_df

def calc_spt_anomalies_unc(in_df, in_unc_df, id_lst):
    """This function calculates the uncertainties of glacier mass balances series"""

    # calculate mb uncertainty of glacier ids with good data over reference period
    unc_ref_df = in_unc_df[id_lst].copy()
    reg_unc_mean= np.nanmean(unc_ref_df)

    for id in unc_ref_df.columns:
        year_min = in_df[id].first_valid_index()
        # TODO: Move 1915 to parameter or constant
        yrs= list(range(1915,year_min))

        if unc_ref_df[id].isnull().all():
            unc_ref_df[id].fillna(reg_unc_mean, inplace=True)
        else:
            unc_ref_df[id].fillna(unc_ref_df[id].mean(), inplace=True)

        unc_ref_df[id].mask(unc_ref_df.index.isin(yrs), np.nan, inplace=True)

    return unc_ref_df

def dis_fil(row, ini_date, fin_date):
    if row > fin_date:
        return row - fin_date + 1
    if row <= ini_date:
        return ini_date + 2 - row
    return 1.0
