# ====================================================================================== #
# Functions for manipulating data.
# Author: Eddie Lee, edlee@santafe.edu
# ====================================================================================== #
from .utils import *

# get map from topic id to topic name
df = pd.read_csv('../starter_packet/topic taxonomy.csv')
TOPICIDMAP = dict(df[['topic_name','topic_id']].values[:,::-1].tolist())



class PQAccess():
    """Class for accessing multiple PQ files at once."""
    # topic col default names
    TCOLS = ['topic_%d'%i for i in range(1,11)]
    TCOLS += ['topic_%d_score'%i for i in range(1,11)]

    def __init__(self, fname_list):
        if type(fname_list) is str:
            fname_list = [fname_list]

        self.pq = []
        for f in fname_list:
            self.pq.append(ParquetFile(f))

    def col(self, col_names, ix=0, **sample_kw):
        """Extract col of data from PQ file concatenated into pandas Series.

        Parameters
        ----------
        col_names : str or list of str
        ix : int, 0
        random_state : int or numpy.random.RandomState
        **sample_kw
            Other keyword args for random sampling.
        
        Returns
        -------
        pd.DataFrame
        """

        if type(col_names) is str:
            col_names = [col_names]
        
        return pd.concat([i.sample(**sample_kw) for i in self.pq[ix].iter_row_groups(col_names)])

    def _topic_weights(self, ix=0):
        """For temp use with public firm database that Alan sent.
        
        Topic weights by record given columns as unique topics.

        Parameters
        ----------
        ix : int, 0

        Returns
        -------
        pd.DataFrame
        """
        
        df = self.col(['domain', 'topic', 'topic_rlcy_sum'], ix=ix)
        # replace topic ids with names
        df['topic'] = [TOPICIDMAP.get(i,str(i)) for i in df['topic']]

        df.set_index('domain', inplace=True)
        return df
    
    def topic_weights(self, ix=0, **sample_kw):
        """Topic weights by record given columns as unique topics.

        Parameters
        ----------
        col_names : str or list of str
        ix : int, 0
        **sample_kw

        Returns
        -------
        pd.DataFrame
            Domain by topic.
        """

        df = self.col(['domain']+self.TCOLS, ix=ix, **sample_kw)
        df.set_index('domain', inplace=True)
        return df
#end PQAccess



class Firehose(PQAccess):
    """Class for manipulating firehose data."""
#end Firehose



# ============== #
# Useful methods #
# ============== #
def csv_to_pq(flist, outfile,
              iprint=False,
              **write_kw):
    """Convert list of csv files into pq file.

    These are ostensibly firehose csvs sent by Alan on 2020/05/31.

    Parameters
    ----------
    flist : list of str
        CSV files to read in. Format is set to Alan's settings.
    outfile : str
        Save pq file to here, overwriting anything already at this location.
    iprint : bool, False
    **write_kw
        compression : str, 'snappy'
        row_group_offsets : int, 5_000_000
            Make sure row groups are small enough that they can be loaded in
            ram. This might depend on the system being used
    """
    
    from fastparquet import write
    import datetime

    # set up column names for reading from csv files
    cols = []
    dtype = {}
    for i in range(1, 11):
        cols.append('topic_%d'%i)
        dtype['topic_%d'%i] = str
        cols.append('topic_%d_score'%i)
        dtype['topic_%d_score'%i] = float
    cols.append('domain')
    dtype['domain'] = str
    cols.append('date')
    dtype['date'] = str
    cols.append('time')
    dtype['time'] = str

    if not 'compression' in write_kw.keys():
        write_kw['compression'] = 'snappy'
    if not 'row_group_offsets' in write_kw.keys():
        write_kw['row_group_offsets'] = 5_000_000
    
    # create file with first data set
    f = flist[0]
    df = pd.read_csv(f, sep='\t', header=None, names=cols, na_values=r'\N', dtype=dtype)
    write(outfile, df, append=False, **write_kw)
    if iprint: print("Done with %s."%f)

    # append onto existing file
    for f in flist[1:]:
        df = pd.read_csv(f, sep='\t', header=None, names=cols, na_values=r'\N', dtype=dtype)
        write(outfile, df, append=True, **write_kw)
        if iprint: print("Done with %s."%f)
