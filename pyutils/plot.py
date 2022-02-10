# ====================================================================================== #
# Plotting helper functions.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #


def jangili_params(ix=0):
    if ix==0:
        return {'G':30,
                'ro':.53,
                're':.43,
                'rd':.414,
                'I':.1,
                'dt':.1}
    elif ix==1:
        return {'G':20,
                'ro':.53,
                're':.43,
                'rd':.414,
                'I':.1,
                'dt':.1}
    else:
        raise Exception

def prb_params(ix=0):
    if ix==0:
        return {'G':90,
                'ro':.68,
                're':.43,
                'rd':.42,
                'I':1.8,
                'dt':.1}
    elif ix==1:
        return {'G':40,
                'ro':.6,
                're':.43,
                'rd':.423,
                'I':.7,
                'dt':.1}
    else:
        raise Exception

def covid_params(ix=0):
    if ix==0:
        return {'G':60,
                'ro':.5,
                're':.78,
                'rd':.47,
                'I':.1,
                'dt':.1,
                'Q':2.3271742356782428}
    elif ix==1:
        return {'G':45,
                'ro':.5,
                're':.78,
                'rd':.47,
                'I':.12,
                'dt':.01,
                'Q':2.3174342105263157}
    else:
        raise Exception

def patent_params(ix=0):
    if ix==0:
        return {'G':23,
                'ro':.54,
                're':.43,
                'rd':.416,
                'I':.48,
                'dt':.1}
    else:
        raise Exception
