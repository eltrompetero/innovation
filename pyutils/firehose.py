# ====================================================================================== #
# Module for accessing and setting up firehose data.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import duckdb as db
import fastparquet as fp
from uuid import uuid4

from .utils import *

FIREHOSE_PATH='/home/eddie/Dropbox/Research/corporations/starter_packet/firehose'


def setup(thisday,
          force=False,
          firehose_path='/home/eddie/Dropbox/Research/corporations/starter_packet/firehose',
          percent=10):
    """
    1. Downsample day to percent.
    2. Calculate 5% percentile of average relevancy and truncate
    3. Create temp parquet 
    
    Parameters
    ----------
    day : str
    force : bool, False
    firehose_path : str, '/home/eddie/Dropbox/Research/corporations/starter_packet/firehose'
    """
    
    conn = db.connect(database=':memory:', read_only=False)

    if force or not os.path.isfile(f'{firehose_path}/{thisday}/subsample.pq'):
        # create subsample DF
        if os.path.isfile(f'{firehose_path}/{thisday}/all.pq'):
            q = f'''
                    SELECT *
                    FROM parquet_scan('{firehose_path}/{thisday}/all.pq')
                    TABLESAMPLE({percent} PERCENT)
                 '''
        else:
            q = f'''
                    SELECT *
                    FROM parquet_scan('{firehose_path}/{thisday}/*_Firehose*.parquet')
                    TABLESAMPLE({percent} PERCENT)
                 '''

        df = conn.execute(q).fetchdf()
        fp.write(f'{firehose_path}/{thisday}/subsample.pq', df, 100_000,
                 compression='SNAPPY')
        del df
    
    pqfile = f'{firehose_path}/{thisday}/subsample.pq'
    if force or not os.path.isfile(f'{firehose_path}/{thisday}/percentile_cutoff.p'):
        # get distribution of relevancy to determine cutoff for individ domains
        q = f'''PRAGMA threads=16;

                SELECT avg_rlvcy
                FROM (SELECT domain, AVG(topic_1_score) AS avg_rlvcy
                    FROM (SELECT domain, topic_1_score FROM parquet_scan('{pqfile}')
                          WHERE topic_1_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_2_score FROM parquet_scan('{pqfile}')
                          WHERE topic_2_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_3_score FROM parquet_scan('{pqfile}')
                          WHERE topic_3_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_4_score FROM parquet_scan('{pqfile}')
                          WHERE topic_4_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_6_score FROM parquet_scan('{pqfile}')
                          WHERE topic_5_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_6_score FROM parquet_scan('{pqfile}')
                          WHERE topic_6_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_7_score FROM parquet_scan('{pqfile}')
                          WHERE topic_7_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_8_score FROM parquet_scan('{pqfile}')
                          WHERE topic_8_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_9_score FROM parquet_scan('{pqfile}')
                          WHERE topic_9_score IS NOT NULL
                          UNION ALL 
                          SELECT domain, topic_10_score FROM parquet_scan('{pqfile}')
                          WHERE topic_10_score IS NOT NULL) udomains
                     GROUP BY domain) rlvnt_domains
                INNER JOIN parquet_scan('{pqfile}') odata
                    ON rlvnt_domains.domain = odata.domain
             '''
        df = conn.execute(q).fetchdf()
        percentile_cutoff = np.percentile(df['avg_rlvcy'], 5)
        del df

        with open(f'{firehose_path}/{thisday}/percentile_cutoff.p', 'wb') as f:
            pickle.dump({'percentile_cutoff':percentile_cutoff}, f)
    else:
        with open(f'{firehose_path}/{thisday}/percentile_cutoff.p', 'rb') as f:
            percentile_cutoff = pickle.load(f)['percentile_cutoff']
    print(f'{percentile_cutoff=:.5f}')
    
    q = f'''PRAGMA threads=16;

            SELECT odata.*
            FROM (SELECT domain, AVG(topic_1_score) AS avg_rlvcy
                FROM (SELECT domain, topic_1_score FROM parquet_scan('{pqfile}')
                      WHERE topic_1_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_2_score FROM parquet_scan('{pqfile}')
                      WHERE topic_2_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_3_score FROM parquet_scan('{pqfile}')
                      WHERE topic_3_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_4_score FROM parquet_scan('{pqfile}')
                      WHERE topic_4_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_5_score FROM parquet_scan('{pqfile}')
                      WHERE topic_5_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_6_score FROM parquet_scan('{pqfile}')
                      WHERE topic_6_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_7_score FROM parquet_scan('{pqfile}')
                      WHERE topic_7_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_8_score FROM parquet_scan('{pqfile}')
                      WHERE topic_8_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_9_score FROM parquet_scan('{pqfile}')
                      WHERE topic_9_score IS NOT NULL
                      UNION ALL 
                      SELECT domain, topic_10_score FROM parquet_scan('{pqfile}')
                      WHERE topic_10_score IS NOT NULL) udomains
                 GROUP BY domain) rlvnt_domains
            INNER JOIN parquet_scan('{pqfile}') odata
                ON rlvnt_domains.domain = odata.domain
            WHERE avg_rlvcy > {percentile_cutoff}
         '''
    df = conn.execute(q).fetchdf()
    fp.write(f'{firehose_path}/{thisday}/filtered_subsample.pq', df,
             compression='SNAPPY')

    print(f'Done with {thisday=}.')

def firm_topics(thisday=None, regex=None, force=False):
    """Topics spanned by a firm, i.e. unique domain topic pairs.

    Parameters
    ----------
    thisday : None
    regex : None
        For accessing many different udomains simultaneously.
    force : bool, False

    Returns
    -------
    pd.DataFrame
    """
    
    if not regex is None:
        pqfile = f'{FIREHOSE_PATH}/{regex}'
    else:
        pqfile = f'{FIREHOSE_PATH}/{thisday}/filtered_subsample.pq'
    
    conn = db.connect(database=':memory:', read_only=False)

    if thisday and (force or not os.path.isfile(f'{FIREHOSE_PATH}/{thisday}/udomains.pq')):
        q = f'''PRAGMA threads=16;

                COPY (SELECT DISTINCT *
                      FROM (SELECT domain, topic_1 FROM parquet_scan('{pqfile}')
                            WHERE topic_1_score IS NOT NULL
                            UNION ALL 
                            SELECT domain, topic_2 FROM parquet_scan('{pqfile}')
                            WHERE topic_2_score IS NOT NULL
                            UNION ALL 
                            SELECT domain, topic_3 FROM parquet_scan('{pqfile}')
                            WHERE topic_3_score IS NOT NULL
                            UNION ALL 
                            SELECT domain, topic_4 FROM parquet_scan('{pqfile}')
                            WHERE topic_4_score IS NOT NULL
                            UNION ALL 
                            SELECT domain, topic_5 FROM parquet_scan('{pqfile}')
                            WHERE topic_5_score IS NOT NULL
                            UNION ALL 
                            SELECT domain, topic_6 FROM parquet_scan('{pqfile}')
                            WHERE topic_6_score IS NOT NULL
                            UNION ALL 
                            SELECT domain, topic_7 FROM parquet_scan('{pqfile}')
                            WHERE topic_7_score IS NOT NULL
                            UNION ALL 
                            SELECT domain, topic_8 FROM parquet_scan('{pqfile}')
                            WHERE topic_8_score IS NOT NULL
                            UNION ALL 
                            SELECT domain, topic_9 FROM parquet_scan('{pqfile}')
                            WHERE topic_9_score IS NOT NULL
                            UNION ALL
                            SELECT domain, topic_10 FROM parquet_scan('{pqfile}')
                            WHERE topic_10_score IS NOT NULL) udomains)
                TO '{FIREHOSE_PATH}/{thisday}/udomains.pq' (FORMAT 'parquet')
             '''
        conn.execute(q)
    elif regex:
        q = f'''PRAGMA threads=16;
                SELECT DISTINCT *
                FROM parquet_scan('{pqfile}')
             '''

    if thisday:
        q = f'''SELECT *
                FROM parquet_scan('{FIREHOSE_PATH}/{thisday}/udomains.pq')
             '''

    df = conn.execute(q).fetchdf().groupby('domain', sort=False)
    return df

def firm_source(thisday=None, regex=None, force=False):
    """Unique source ids considered by a firm (i.e. domain).

    Parameters
    ----------
    thisday : None
    regex : None
        For accessing many different sources simulatneously.
    force : bool, False

    Returns
    -------
    pd.DataFrame
    """
    
    if not regex is None:
        pqfile = f'{FIREHOSE_PATH}/{regex}'
    else:
        pqfile = f'{FIREHOSE_PATH}/{thisday}/filtered_subsample.pq'
    
    conn = db.connect(database=':memory:', read_only=False)

    if thisday and (force or not os.path.isfile(f'{FIREHOSE_PATH}/{thisday}/usource.pq')):
        q = f'''PRAGMA threads=16;

                COPY (SELECT DISTINCT *
                      FROM (SELECT domain, source_id FROM parquet_scan('{pqfile}')))
                TO '{FIREHOSE_PATH}/{thisday}/usource.pq' (FORMAT 'parquet')
             '''
        conn.execute(q)
    elif regex:
        tempfile = os.path.isfile(f'cache/usource_{str(uuid4())}')
        while os.path.isfile(f'cache/{tempfile}'):
            tempfile = os.path.isfile(f'cache/usource_{str(uuid4())}')

        q = f'''COPY (SELECT *
                      FROM parquet_scan('{pqfile}'))
                TO 'cache/{tempfile}.pq' (FORMAT 'parquet')
             '''
        conn.execute(q)

    if thisday:
        q = f'''SELECT *
                FROM parquet_scan('{FIREHOSE_PATH}/{thisday}/usource.pq')
             '''
    elif regex:
        q = f'''SELECT *
                FROM parquet_scan('cache/{tempfile}.pq')
             '''

    df = conn.execute(q).fetchdf().groupby('domain', sort=False)
    return df

def hist_firm_records(bins, fname):
    """Histogram of firm records for given bins.

    Parameters
    ----------
    bins : ndarray
    fname : str
        Can include regular expressions b/c it's passed to parquet_scan.

    Returns
    -------
    ndarray
        Counts in each bin.
    """
    
    conn = db_conn()

    # use query to construct counts over multiple days
    # create bins as case conditions
    s = ''
    for i in range(len(bins)-1):
        s += f"count(CASE WHEN counts>={bins[i]} AND counts < {bins[i+1]} THEN 1 END) AS 'bin {i}',"
    s = s[:-1]

    q = f'''PRAGMA threads=16;
            SELECT {s}
            FROM (SELECT domain, COUNT(*) as counts
                  FROM parquet_scan('{fname}')
                  GROUP BY domain)
         '''
    return conn.execute(q).fetchdf().values.ravel()

def hist_firm_topics(bins, fname):
    """Topics spanned by a firm.

    Parameters
    ----------
    bins : ndarray
    fname : str
        Can include regular expressions b/c it's passed to parquet_scan.

    Returns
    -------
    ndarray
        Counts in each bin.
    """
    
    conn = db_conn()

    # use query to construct counts over multiple days
    # create bins as case conditions
    s = ''
    for i in range(len(bins)-1):
        s += f"count(CASE WHEN counts>={bins[i]} AND counts < {bins[i+1]} THEN 1 END) AS 'bin {i}',"
    s = s[:-1]

    q = f'''PRAGMA threads=16;
            SELECT {s}
            FROM (SELECT APPROX_COUNT_DISTINCT(topic_1) as counts
                  FROM parquet_scan('{fname}')
                  GROUP BY domain)
         '''

    return conn.execute(q).fetchdf().values.ravel()

def hist_firm_source(bins, fname):
    """Topics spanned by a firm.

    Parameters
    ----------
    bins : ndarray
    fname : str
        Can include regular expressions b/c it's passed to parquet_scan.

    Returns
    -------
    ndarray
        Counts in each bin.
    """
    
    conn = db_conn()

    # use query to construct counts over multiple days
    # create bins as case conditions
    s = ''
    for i in range(len(bins)-1):
        s += f"count(CASE WHEN counts>={bins[i]} AND counts < {bins[i+1]} THEN 1 END) AS 'bin {i}',"
    s = s[:-1]

    q = f'''PRAGMA threads=16;
            SELECT {s}
            FROM (SELECT COUNT(*) as counts
                  FROM parquet_scan('{fname}')
                  GROUP BY domain)
         '''

    return conn.execute(q).fetchdf().values.ravel()

def setup_cooc(thisday=None, cache=False, subsample=True):
    """Get co-occurrence and single occurrence counts for all topics and save to
    parquet file.
    
    There is a problem with this because co-occ counts a single topic multiple times
    depending on how many other topics are included in the record. This somehow
    overrepresents topics that appear in complex articles. We could avoid this by
    only considering articles with at least 10 labels, but this also may filter out
    some important pieces.

    I checked manually with a simple DataFrame that this does what is expected in the
    counting pairs section.
    
    Parameters
    ----------
    thisday : str, None
    cache : bool, False
        If True, load from cache or save to cache.
    subsample : bool, True
        If True, use filtered subsample instead of entire data set.

    Returns
    -------
    pd.DataFrame
    """
    
    if cache and os.path.isfile('cache/pairs_cooc.pq'):
        q = '''SELECT *
               FROM parquet_scan('cache/pairs_cooc.pq')
            '''
        return db_conn().execute(q).fetchdf()

    q = '''PRAGMA threads=16;
        PRAGMA memory_limit='128GB';

        CREATE TABLE topics (
          source_id varchar(255),
          topic_1 varchar(255),
          PRIMARY KEY (source_id, topic_1)
        );
        
        INSERT INTO topics
        SELECT DISTINCT source_id, topic_1
        FROM (
        '''
    for i in range(1, 11):
        if not subsample:
            q += f'''SELECT DISTINCT source_id, topic_{i} AS topic_1
                     FROM parquet_scan('{FIREHOSE_PATH}/{thisday}/Redacted*parquet')
                     WHERE topic_{i} IS NOT NULL AND topic_{i} <> ''
                     UNION ALL''' + '\n'
        else:
            q += f'''SELECT DISTINCT source_id, topic_{i} AS topic_1
                     FROM parquet_scan('{FIREHOSE_PATH}/{thisday}/filtered_subsample.pq')
                     WHERE topic_{i} IS NOT NULL AND topic_{i} <> ''
                     UNION ALL''' + '\n'
    q = q.rstrip()[:-9]
    q += ''');\n'''
    
    q += f'''CREATE TABLE results AS
                SELECT topics1.topic_1 AS topic_1, topics2.topic_1 AS topic_2, COUNT(*) AS counts
                FROM topics topics1
                INNER JOIN topics topics2
                   ON topics1.source_id = topics2.source_id AND topics1.topic_1 < topics2.topic_1
                GROUP BY topics1.topic_1, topics2.topic_1;
         '''
    if cache:
        q += '''
             COPY results
             TO 'cache/pairs_cooc.pq' (FORMAT 'parquet');
             '''
    q += '''
         SELECT * FROM results
         '''
    return db_conn().execute(q).fetchdf()
 
def _setup_cooc(thisday=None, regex=None):
    """Get co-occurrence and single occurrence counts for all topics and save to
    parquet file.
    
    There is a problem with this because co-occ counts a single topic multiple times
    depending on how many other topics are included in the record. This somehow
    overrepresents topics that appear in complex articles. We could avoid this by
    only considering articles with at least 10 labels, but this also may filter out
    some important pieces.
    
    Parameters
    ----------
    thisday : str
    """
    
    assert (not thisday is None) or (not regex is None)
    thisday = thisday or regex
    
    def loop_wrapper(args):
        topic1, topic2 = args
        q = f'''
             SELECT topic_1, topic_2, COUNT(*) as counts
             FROM (SELECT DISTINCT source_id,
                          LEAST(topic_{topic1}, topic_{topic2}) AS topic_1,
                          GREATEST(topic_{topic1}, topic_{topic2}) AS topic_2
                   FROM parquet_scan('{FIREHOSE_PATH}/{thisday}/filtered_subsample.pq')
                   WHERE topic_{topic1} IS NOT NULL AND topic_{topic1} <> ''
                       AND topic_{topic2} IS NOT NULL AND topic_{topic2} <> '')
             GROUP BY topic_1, topic_2
             '''
        thisdf = db_conn().execute(q).fetchdf()
        return thisdf
        
    with Pool() as pool:
        df = pool.map(loop_wrapper, combinations(range(1,11), 2))

    pairsdf = pd.concat(list(df))
    del df  # clear memory
    pairsdf = pairsdf.groupby(['topic_1','topic_2'], sort=False).sum()
    pairsdf = pairsdf.reset_index()
    if regex:
        fp.write(f'cache/pairs_cooc.pq', pairsdf)
    else:
        fp.write(f'{FIREHOSE_PATH}/{thisday}/pairs_cooc.pq', pairsdf)

def topic_frequency(thisday):
    """Calculate frequencies of each topic.
    """

    q = f'''PRAGMA threads=16;
            
            SELECT topic_1, COUNT(*) as counts
            FROM ('''
    for i in range(1, 11):
        q += f'''SELECT domain, topic_{i} FROM parquet_scan('{FIREHOSE_PATH}/{thisday}/filtered_subsample.pq')
                 WHERE topic_{i} IS NOT NULL AND topic_{i} <> ''
                 UNION ALL''' + '\n'
    q = q[:-10]
    q += ')\nGROUP BY topic_1'
    
    frequencydf = db_conn().execute(q).fetchdf()
    return frequencydf 

def compustat():
    """Load compustat info.

    Returns
    -------
    pd.DataFrame
    """
    
    q = f'''SELECT *
            FROM parquet_scan('../data/compustat_quarterly_size.parquet')
         '''

    return db_conn().execute(q).fetchdf()
