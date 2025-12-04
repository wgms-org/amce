import pytest
from ..ggmc.functions import *

# Section 1 Function Tests

# Input the fog version
FOG_VERSION = '2025-01'

# Input the path (as a string) of a directory where you'd like the outputs to be delivered
OUTPUT_DATA_PATH = './data/output/1_glacier_change_data/'

# Input files for the elevation change function (as relative paths)
ELEV_CHANGE_INPUT = './data/input/1_glacier_change_data/FOG_ELEVATION_CHANGE_DATA_2025-01.csv'
GLAC_SERIES_INPUT = './data/input/1_glacier_change_data/FOG_GLACIER_SERIES_2025-01.csv'

# Input a list of providers to clean with the function
PROVIDERS_TO_DROP = ['Robert McNabb','Thorsten Seehaus']

# Input the path (as a string) to the WGMS Fog input data (relative path only)
WGMS_INPUT = 'data/input/1_glacier_change_data/fog_bw-bs-ba_2025-01.csv'

# Test the validity of the outputs
def test_data_prep_spt_anom(data_prep_spt_anom_validate):
    # Call the original function
    ba_workflow, bs_workflow, bw_workflow, ba_unc_workflow = data_prep_spt_anom(FOG_VERSION,
                                                                                WGMS_INPUT,
                                                                                OUTPUT_DATA_PATH)
    
    # Retrieve the validated original output
    ba_valid, bs_valid, bw_valid, ba_unc_valid = data_prep_spt_anom_validate

    # pd.testing.assert_frame_equal(ba_workflow,ba_valid)
    pd.testing.assert_frame_equal(bs_workflow,bs_valid)
    pd.testing.assert_frame_equal(bw_workflow,bw_valid)
    pd.testing.assert_frame_equal(ba_unc_workflow,ba_unc_valid)

def test_data_prep_elevation_change(data_prep_elevation_change_validate):
    # Call the original function
    coord_workflow, mdb_workflow = data_prep_elevation_change(FOG_VERSION,
                                                              ELEV_CHANGE_INPUT,
                                                              GLAC_SERIES_INPUT,
                                                              PROVIDERS_TO_DROP,
                                                              OUTPUT_DATA_PATH)
    # Retrieve the validated original output
    coord_valid, mdb_valid = data_prep_elevation_change_validate

    pd.testing.assert_frame_equal(coord_workflow,coord_valid)
    pd.testing.assert_frame_equal(mdb_workflow,mdb_valid)

    