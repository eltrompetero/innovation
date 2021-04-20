# ====================================================================================== #
# Testing module for CPP module extension.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import numpy as np
import os, pickle
from pyutils.model_ext import *



def test_TopicLattice():
    # check initialization
    lattice = TopicLattice()
    assert lattice.len()==1
    assert lattice.get_occ(0)==0

    # check adding of elements
    lattice.add(0, 2)
    assert lattice.len()==1
    assert lattice.get_occ(0)==2

    lattice.add(0, 10)
    assert lattice.len()==1
    assert lattice.get_occ(0)==12

    # check removal
    lattice.remove(0, 10)
    assert lattice.len()==1
    assert lattice.get_occ(0)==2

    # check extension
    lattice.extend_right(3)
    assert lattice.len()==4
    assert lattice.get_occ(0)==2

    # check viewing functions
    lattice.add(2, 2)
    assert np.array_equal(lattice.get_occ(np.array([0,1,2], dtype=np.int32)),
                          [lattice.get_occ(i) for i in range(3)])
    assert np.array_equal([lattice.get_occ(i) for i in range(3)],
                          [2, 0, 2])

    assert np.array_equal(lattice.view(), [2, 0, 2, 0])

    # check shrinking
    lattice.remove(0, 2)
    lattice.remove(2, 2)
    assert np.array_equal(lattice.view(), [0, 0, 0, 0]), lattice.view()

    lattice.shrink_left(1)
    assert np.array_equal(lattice.view(), [0, 0, 0])

    lattice.add(2, 3)
    assert np.array_equal(lattice.view(), [0, 3, 0]), lattice.view()
    
    # check pickling
    try:
        with open('test.p', 'wb') as f:
            pickle.dump({'lattice':lattice}, f);
        with open('test.p', 'rb') as f:
            lattice = pickle.load(f)['lattice']
        assert np.array_equal(lattice.view(), [0, 3, 0]), lattice.view()
    finally:
        os.remove('test.p')

def test_LiteFirm():
    # check initialization
    firm = LiteFirm((1,2), .1, .2, .3, 1, 'test')
    assert firm.sites==(1,2)
    assert firm.innov==.1
    assert firm.wealth==.2
    assert firm.connection_cost==.3
    assert firm.age==1
    assert firm.id=='test'

    # check pickling
    try:
        assert not os.path.isfile('test.p')
        pickle.dump({'firm':firm}, open('test.p','wb'))
        firm = pickle.load(open('test.p','rb'))['firm']
        assert firm.sites==(1,2)
        assert firm.innov==.1
        assert firm.wealth==.2
        assert firm.connection_cost==.3
        assert firm.age==1
        assert firm.id=='test'
    finally:
        os.remove('test.p')
