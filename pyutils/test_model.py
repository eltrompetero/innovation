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

def test_reconstruct_lattice():
    firm_snapshot = [[LiteFirm((0,2), 0., 0., 0., 0, "test"),
                      LiteFirm((1,3), 0., 0., 0., 0, "test")]]
    lattice_snapshot = reconstruct_lattice(firm_snapshot, [(0, 5)])
    assert np.array_equal(lattice_snapshot[0].occupancy, np.array([1,2,2,1,0,0]))
