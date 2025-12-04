import pytest
from ..ggmc.functions import *

# !! Paths to the validate data structured as from the parent directory
# !! given `pytest tests` will be run from that location.

@pytest.fixture(scope="module")
def data_prep_elevation_change_validate():
    coord_path = './data/validate/1_glacier_change_data/FOG_coord_2025-01.csv'
    coord_validate_data = pd.read_csv(coord_path, delimiter=',', header=0, index_col=0)
    mbd_path = './data/validate/1_glacier_change_data/_FOG_GEO_MASS_BALANCE_DATA_2025-01.csv'
    mbd_validate_data = pd.read_csv(mbd_path, delimiter=',', header=0)
    return coord_validate_data,mbd_validate_data

@pytest.fixture(scope="module")
def data_prep_spt_anom_validate():
    """
    The mb_data_crunching_local/1_glacier_change_data/out_data_edited/fog-2024-01 versions of the outputs
    appeared to be the output that matches her original data_prep_spt_anom function. On first investigation,
    these do not match the input files in "2_Kriging_spatial_anomalies", from the in_data directory within
    the fog-2025-01 sub-directory (i.e., mb_data_crunching_local/2_Kriging_spatial_anomalies/in_data/fog-2025-01).
    Hence the naming of the (sub)directory `from_2nd_step`
    
    """
    ba_path = './data/validate/1_glacier_change_data/from_2nd_step/fog_2025-01_ba.csv'
    ba_valid = pd.read_csv(ba_path, delimiter=',', header=0, index_col=0)
    bs_path = './data/validate/1_glacier_change_data/from_2nd_step/fog_2025-01_bs.csv'
    bs_valid = pd.read_csv(bs_path, delimiter=',', header=0, index_col=0)
    bw_path = './data/validate/1_glacier_change_data/from_2nd_step/fog_2025-01_bw.csv'
    bw_valid = pd.read_csv(bw_path, delimiter=',', header=0, index_col=0)
    ba_unc_path = './data/validate/1_glacier_change_data/from_2nd_step/fog_2025-01_ba_unc.csv'
    ba_unc_valid = pd.read_csv(ba_unc_path, delimiter=',', header=0, index_col=0)
    return ba_valid, bs_valid, bw_valid, ba_unc_valid

