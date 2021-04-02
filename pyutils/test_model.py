# ======================================================================== #
# 1D firm innovation models
# Author : Eddie Lee, edlee@csh.ac.at
# ======================================================================== #
from .model import *



def test_segment_by_bound_vel():
    leftright = np.zeros((100,2))
    leftright[:10,1] = 1 

    windows, vsign, v = segment_by_bound_vel(leftright, 0, moving_window=1)
    assert len(windows)==len(vsign)==len(v)
    assert [i[1]-i[0] for i in windows]==[9,1,89]

    windows, vsign, v = segment_by_bound_vel(leftright, .2, moving_window=2)
    assert [i[1]-i[0] for i in windows]==[1, 8, 2, 88]
