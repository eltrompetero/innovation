# ====================================================================================== #
# For analyzing genome data for innovation project.
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from Bio import Phylo
from multiprocess import Pool
from warnings import warn

from .utils import *



def par_distance(tree):
    """Get total distance from root to each terminal.

    Parameters
    ----------
    tree : Phylo.Tree

    Returns
    -------
    list
    """

    terminals = tree.get_terminals()
    def loop_wrapper(t):
        return tree.distance(t, terminals[0])
        
    with Pool() as pool:
        return list(pool.map(loop_wrapper, terminals, 100))

def branching_ratio(clades):
    """Recursive thru tree and get branching ratio per unit distance.

    Parameters
    ----------
    clades : list of Clade
    
    Returns
    -------
    list of int
        Branching ratio per unit branch length per branch in the entire tree. The
        average of this should give the tendency for the tree to branch per unit
        length, which is Q-1 for our model.
    """
    
    all_dist = []
    
    # count branching ratio by looking at where each new clade splits off in terms
    # of the hamming distance, this is really trying to use the structure of the tree as
    # inferred from the observed lineages instead of treating each lineage as the point
    # of splitting...either way could be appropriate, the latter would be the most model-free
    # way of imputting branching ratio
    n = np.bincount([c.branch_length for c in clades])
    ntot = n.sum()
    # there should be no cases of where two children appear without a parent, but this happens
    # near the root
    warn(f'n[0] not 0, it is {n[0]}')
    for i, n_ in enumerate(n):
        if i>0:
            all_dist.append(n_+1)
    
    # consider all non-terminal clades to be a new starting point for computing branching
    # ratios
    newclades = []
    for c in clades:
        if not c.is_terminal():
            newclades.append(c.clades)
        
    # recurse over each new child clade from previous parent clade
    counter = 0
    while newclades:# and counter<100:
        clades = newclades.pop(0)
        
        # see above
        n = np.bincount([c.branch_length for c in clades])
        ntot = n.sum()
        warn(f'n[0] not 0, it is {n[0]}')
        for i, n_ in enumerate(n):
            if i>0:
                all_dist.append(n_+1)
        
        for c in clades:
            if not c.is_terminal():
                newclades.append(c.clades)
        counter += 1
        
    return all_dist
