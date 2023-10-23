import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from scipy import sparse
from jax import random
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

from .utils import *


class KTree():
    def __init__(self, n0, n1, K, gamma=0, rng=None, sparse_adj=True):
        """
        Parameters
        ----------
        n0 : int
            Length of initial single branch.
        n1 : int
            Length of the branches diverging from initial branch.
        K: int
            number of chains in the network
        gamma : float, 0.
            Probability of connection between parallel branches.
        rng : np.random.RandomState, None
        sparse_adj : bool, True
            If False, adj array will not be sparse_adj.
        """
        self.gamma = gamma
        if sparse_adj:
            self.adj = sparse.lil_array((n0+n1*K, n0+n1*K), dtype=np.bool_)
        else:
            self.adj = np.zeros((n0+n1*K, n0+n1*K), dtype=np.bool_)
        self.rng = rng if not rng is None else np.random

        # create first, shared root branch
        ix = zip(*((i,i+1) for i in range(n0-1)))
        self.adj[tuple(ix)] = True
        # create interleaved representation of K branches
        ix = zip(*((i,i+K) for i in range(n0, n0+K*n1-K)))
        self.adj[tuple(ix)] = True

        # create connections between branches
        for i in range(K):
            self.adj[n0-1,n0+i] = True        
        if gamma==1:
            # take every branch pair and randomly connect sequential generations with probability gamma
            # must consider both ordered directions of connections
            for i, j in combinations(range(K), 2):
                # from i->j
                ix = zip(*((n0+i+K*el,n0+j+K*(el+1)) for el in range(n1-1)))
                self.adj[tuple(ix)] = True

                # from j->i
                ix = zip(*((n0+j+K*el,n0+i+K*(el+1)) for el in range(n1-1)))
                self.adj[tuple(ix)] = True
        elif gamma:
            # random connections
            for i, j in combinations(range(K), 2):
                # from i->j
                seq = self.rng.rand(n1-1) < gamma
                ix = zip(*((n0+i+K*el,n0+j+K*(el+1)) for el in range(n1-1)))
                self.adj[tuple(ix)] = seq

                # from j->i
                seq = self.rng.rand(n1-1) < gamma
                ix = zip(*((n0+j+K*el,n0+i+K*(el+1)) for el in range(n1-1)))
                self.adj[tuple(ix)] = seq

    def as_graph(self):
        return nx.DiGraph(self.adj)
#end KTree

class JaxBiTree():
    def __init__(self, n0, n1, gamma=0, seed=0):
        """
        Parameters
        ----------
        n0 : int
            Length of initial single branch.
        n1 : int
            Length of the two branches to diverge from initial branch.
        gamma : float, 0.
            Probability of connection between parallel branches.
        seed : int, 0
        """
        self.key = random.PRNGKey(seed)
        
        # create skeleton of tree
        ix = jnp.vstack([(i,i+1) for i in range(n0)])
        ix = jnp.vstack((ix, jnp.vstack([(i,i+2) for i in range(n0, 2*n1+n0-2)])))
        ix = jnp.append(ix, jnp.array([[n0-1,n0+1]]), axis=0)
        data = jnp.ones(n0 + 2*n1-2 + 1, dtype=jnp.int8)
        
        # add interleaving edges
        self.gamma = gamma
        if gamma:
            self.key, subkey = random.split(self.key)
            newdata = random.uniform(subkey, (2*n1-3,)) < gamma
            newix = jnp.vstack([(i,i+3) for i in range(n0, 2*n1+n0-3)])

            newix = newix[newdata,:]
            newdata = newdata[newdata].astype(jnp.int8)

            data = jnp.concatenate((data, newdata))
            ix = jnp.vstack((ix, newix))

        self.adj = jsparse.BCOO((data, ix), shape=(n0+2*n1,n0+2*n1))
        
    def as_graph(self):
        return nx.DiGraph(np.array(self.adj.todense()))
#end BiTree


class Tree():
    def __init__(self, r, n):
        """Create Bethe lattice with n nodes.

        Parameters
        ----------
        r : int
            Branching ratio.
        n : int
            Number of nodes.
        """
        self.r = r
        self.n = n

        # Create a full r-ary tree of height 3 and branching ratio 3
        G = nx.full_rary_tree(r, n)

        self.DG = nx.DiGraph()

        # Add nodes
        self.DG.add_nodes_from(G.nodes())

        # Manually add directed edges from parent to children
        for u, v in G.edges():
            if u<v:
                self.DG.add_edge(u, v)
            else:
                self.DG.add_edge(v, u)
    
        self.adj = nx.to_scipy_sparse_array(self.DG, format='coo')

    def as_graph(self):
        return self.DG
#end Tree



def create_directed_tree(N, k, density):
    if N <= 0 or k <= 0 or k >= N:
        raise ValueError("Invalid values for N and k. N must be greater than 0, and k must be between 1 and N-1.")
    if density < 0 or density > 1:
        raise ValueError("Invalid value for density. It must be between 0 and 1.")

    G = nx.DiGraph()

    # Add the root node
    G.add_node(0)

    # Create the tree by adding nodes with degree k
    node_counter = 1
    stack = [0]  # Use a stack to keep track of the parent nodes
    while node_counter < N:
        parent = stack.pop()
        for i in range(min(k, N - node_counter)):
            child = node_counter
            G.add_edge(parent, child)
            stack.append(child)
            node_counter += 1

    # Connect nodes of consecutive generations with probability 'density'
    for parent, child in G.edges():
        if child - parent == 1 and np.random.rand() <= density:
            G.add_edge(child, parent)

    return G

def draw_directed_tree(G, pos=None, ax=None,
                       node=True, edge=True, label=False):
    if ax is None:
        fig, ax = plt.subplots()
    if not pos is None:
        pos = graphviz_layout(G, prog='dot')

    if edge:
        nx.draw_networkx_edges(G, pos,
                            ax=ax,
                            arrows=True, alpha=.5)
    if node:
        nx.draw_networkx_nodes(G, pos,
                            ax=ax,
                            node_size=20,
                            node_color='skyblue')
    if label:
        nx.draw_networkx_labels(G, pos,
                                ax=ax,
                                labels=dict(zip(list(G.nodes), list(G.nodes))),
                                font_size=10,
                                font_weight='bold');
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    return pos

def select_largest_component(adj):
    """Take the adjacency matrix of the largest connected component in the graph.
    
    Parameters
    ----------
    adj : ndarray
    
    Returns
    -------
    ndarray
    """
    G = nx.Graph(adj)
    connected_components = nx.connected_components(G)
    largest_connected_component = max(connected_components, key=len)
    largest_connected_subgraph = G.subgraph(largest_connected_component)
    return nx.adjacency_matrix(largest_connected_subgraph).toarray()

def graph_as_shells(starting_nodes, adj):
    """Group nodes into shells growing out from starting nodes.
    
    Parameters
    ----------
    starting_nodes : list
        Starting nodes for numbering.
    adj : ndarray
        Adjacency matrix.
    
    Returns
    -------
    list of lists
        Each consecutive list is a list of nodes that belong to the next shell.
    """
    assert len(starting_nodes)>0 and all([0<=n<adj.shape[0] for n in starting_nodes])
    grouped_nodes = set()
    remaining_nodes = set(range(n))

    shells = [starting_nodes]
    grouped_nodes = set(shells[0])
    while len(grouped_nodes)<n_:
        #print(len(grouped_nodes))
        # consider all the children of the previous shell
        children = set()
        for p in shells[-1]:
            child_ix = set(np.where(adj[p])[0].tolist())
            children |= child_ix

        # remove any children that have already been grouped
        children -= grouped_nodes
        shells.append(list(children))
        grouped_nodes |= children
    return shells

def reindex_graph_by_shell(shells):
    """Give a new indexing of nodes in graph to ensure that index always increases
    shell by shell. Numbering within each shell is arbitrary.
    
    Parameters
    ----------
    shells : list of lists
        Node numbers grouped by consecutive shell.
    
    Returns
    -------
    np.ndarray
        New indices for nodes.
    """
    return list(chain.from_iterable(shells))

def get_pair_combinations(array1, array2):
    pairs = []
    for item1 in array1:
        for item2 in array2:
            pairs.append((item1, item2))
    return pairs

def add_random_edge_between_generations(network, pos_links, links, γ):
    to_add = np.random.choice(len(pos_links), int(γ*len(pos_links)), replace= False)
    #print(to_add, [i for i in to_add])
    network.add_edges_from(np.array(pos_links)[[i for i in to_add]])
    return network
