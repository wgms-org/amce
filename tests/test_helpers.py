import pytest
from ..ggmc.helpers import *

def test_date_format():
    """
    The test function for date_format
    """
    
    assert date_format(7,2020) == 2020.5