# ====================================================================================== #
# Organization module for simulation results.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from time import sleep
from glob import glob

from .utils import *



class SimLedger():
    """Class for keeping track of firm simulations.

    Parameters
    ----------
    ledger_path : str, 'cache/sim_ledger.p'
    cache_dr : str, 'cache/sims'
    """
    def __init__(self,
                 ledger_path='./cache/sim_ledger.p',
                 cache_dr='./cache/sims'):
        self.ledger_path = ledger_path
        self.cache_dr = cache_dr

        self.reload_ledger()

    def reload_ledger(self):
        if os.path.isfile(self.ledger_path):
            self.ledger = pd.read_pickle(self.ledger_path)
        else:
            self.ledger = pd.DataFrame(columns=['L0',
                                                'g0',
                                                'obs_rate',
                                                'expand_rate',
                                                'innov_rate',
                                                'exploit_rate',
                                                'growf',
                                                'depressed_frac',
                                                'connect_cost',
                                                'n_sims',
                                                'T',
                                                'dt'])
            self.ledger['L0'] = self.ledger['L0'].astype(int)
            self.ledger['n_sims'] = self.ledger['n_sims'].astype(int)

    def save_ledger(self, mx_tries=3):
        """Try to overwritre existing pickle except when it is locked.

        Parameters
        ----------
        mx_tries : int, 3
            Number of times to try saving ledger when lock file is present.
        """

        if os.path.isfile(self.ledger_path):
            print("Overwriting existing ledger cache.")
        
        counter = 0
        while os.path.isfile(self.ledger_path+'.lock') and counter<mx_tries:
            print("File locked...")
            sleep(1)
            counter += 1
        
        if counter<mx_tries:
            with open(self.ledger_path+'.lock', 'w') as f:
                f.write('')

            self.ledger.to_pickle(self.ledger_path)
            os.remove(self.ledger_path+'.lock')
        else:
            print("Failed to save ledger.")

    def add(self, name, props,
            save_to_file=True,
            force_overwrite=False,
            ignore_missing_file=False):
        """Add a simulation to the ledger.

        Parameters
        ----------
        name : str
            Folder name in cache_dr.
        props : dict
        save_to_file : bool, True
        force_overwrite : bool, False
            Overwrite existing entry if True.
        ignore_missing_file : bool, False
        """
        
        if not ignore_missing_file:
            assert os.path.isdir(self.cache_dr), "Specified cache not found."
        
        if force_overwrite and name in self.ledger.index:
            self.ledger[name] = pd.Series(props, index=props.keys(), name=name)
        elif not force_overwrite and name in self.ledger.index:
            raise Exception("Ledger entry already exists.")
        else:
            self.ledger = self.ledger.append(pd.Series(props, index=props.keys(), name=name))

        if save_to_file:
            self.save_ledger()

    def remove(self, name,
               save_to_file=False,
               delete_cache_files=False):
        """Delete a simulation entry from the ledger.

        Parameters
        ----------
        name : str or int
        save_to_file : bool, False
            Save ledger to file.
        delete_cache_files : bool, False
            Delete pickles and cache dir.
        """
        
        from shutil import rmtree

        if isinstance(name, int):
            name = self.ledger.index[name]
        else: 
            assert name in self.ledger.index, "Ledger entry does not exist."

        self.ledger = self.ledger.drop(index=name)

        if save_to_file:
            self.save_ledger()

        if delete_cache_files is True:
            rmtree(f'{self.cache_dr}/{name}')
        elif delete_cache_files=='density':
            for f in glob(f'{self.cache_dr}/{name}/*_density.parquet'):
                os.remove(f)
        else:
            print("Cached files not removed.")

    def load(self, name):
        """Load simulation with given name.

        Parameters
        ----------
        name : str or int
            If str, tries to load ledger with this index. If int, tries to load the ith entry.

        Returns
        -------
        Simulator
            Storage data member contains sim results.
        """
        
        if isinstance(name, int):
            name = self.ledger.index[name]
        elif isinstance(name, np.int64):
            name = self.ledger.index[name]

        with open(f'{self.cache_dr}/{name}/top.p', 'rb') as f:
            simulator = pickle.load(f)['simulator']
        return simulator

    def info(self, cols):
        """
        Parameters
        ----------
        cols : list of str

        Returns
        -------
        pd.DataFrame
            Return specified columns and count col, which always counts from 0
            sequentially up to the last entry.
        """
        
        if type(cols) is str:
            cols = [cols]

        dfcopy = pd.concat((pd.Series(range(len(self.ledger)),
                                      name='count',
                                      index=self.ledger.index),
                                      self.ledger[cols]),
                                      ignore_index=False,
                                      axis=1)
        return dfcopy
#end SimLedger
