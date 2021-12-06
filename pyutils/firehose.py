# ====================================================================================== #
# Module for accessing and setting up firehose data.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import duckdb as db
import fastparquet as fp
from uuid import uuid4
from workspace.utils import increment_name

from .utils import *

FIREHOSE_PATH='/home/eddie/Dropbox/Research/corporations/starter_packet/firehose'



def setup(thisday,
          force=False,
          percent=10):
    """
    1. Downsample day to percent.
    2. Calculate 5% percentile of average relevancy and truncate
    3. Create temp parquet 
    
    Parameters
    ----------
    day : str
    force : bool, False
    """
    
    conn = db_conn()

    if force or not os.path.isfile(f'{FIREHOSE_PATH}/{thisday}/subsample.pq'):
        # create subsample DF
        q = f'''PRAGMA threads=32;
                
             COPY
                 (SELECT domain||topic_1_score||topic_2_score||topic_3_score||topic_4_score||
                         topic_5_score||topic_6_score||topic_7_score||topic_8_score||topic_9_score||
                         topic_10_score AS article_id,
                        hashed_email__id_hex,
                        hashed_email__id_base64,
                        domain,
                        interaction_type,
                        topic_1,
                        topic_1_score,
                        topic_2,
                        topic_2_score,
                        topic_3,
                        topic_3_score,
                        topic_4,
                        topic_4_score,
                        topic_5,
                        topic_5_score,
                        topic_6,
                        topic_6_score,
                        topic_7,
                        topic_7_score,
                        topic_8,
                        topic_8_score,
                        topic_9,
                        topic_9_score,
                        topic_10,
                        topic_10_score,
                        universal_datetime,
                        country,
                        stateregion,
                        postal_code,
                        localized_datetime,
                        custom_id,
                        source_id,
                        date_localized,
                        time_localized,
                        time_utc,
                        date_utc
                 FROM parquet_scan('{FIREHOSE_PATH}/{thisday}/*_Firehose*.parquet')
                 TABLESAMPLE({percent} PERCENT)
                 WHERE topic_1_score IS NOT NULL AND
                    topic_2_score IS NOT NULL AND
                    topic_3_score IS NOT NULL AND
                    topic_4_score IS NOT NULL AND
                    topic_5_score IS NOT NULL AND
                    topic_6_score IS NOT NULL AND
                    topic_7_score IS NOT NULL AND
                    topic_8_score IS NOT NULL AND
                    topic_9_score IS NOT NULL AND
                    topic_10_score IS NOT NULL)
             TO '{FIREHOSE_PATH}/{thisday}/filtered_subsample.pq' (FORMAT PARQUET)
             '''
    conn.execute(q)

    #pqfile = f'{FIREHOSE_PATH}/{thisday}/subsample.pq'
    #if force or not os.path.isfile(f'{FIREHOSE_PATH}/{thisday}/percentile_cutoff.p'):
    #    # get distribution of relevancy to determine cutoff for individ domains
    #    q = f'''PRAGMA threads=32;

    #            SELECT avg_rlvcy
    #            FROM (SELECT domain, AVG(topic_1_score) AS avg_rlvcy
    #                FROM (SELECT domain, topic_1_score FROM parquet_scan('{pqfile}')
    #                      WHERE topic_1_score IS NOT NULL
    #                      UNION ALL
    #                      SELECT domain, topic_2_score FROM parquet_scan('{pqfile}')
    #                      WHERE topic_2_score IS NOT NULL
    #                      UNION ALL
    #                      SELECT domain, topic_3_score FROM parquet_scan('{pqfile}')
    #                      WHERE topic_3_score IS NOT NULL
    #                      UNION ALL
    #                      SELECT domain, topic_4_score FROM parquet_scan('{pqfile}')
    #                      WHERE topic_4_score IS NOT NULL
    #                      UNION ALL
    #                      SELECT domain, topic_6_score FROM parquet_scan('{pqfile}')
    #                      WHERE topic_5_score IS NOT NULL
    #                      UNION ALL
    #                      SELECT domain, topic_6_score FROM parquet_scan('{pqfile}')
    #                      WHERE topic_6_score IS NOT NULL
    #                      UNION ALL
    #                      SELECT domain, topic_7_score FROM parquet_scan('{pqfile}')
    #                      WHERE topic_7_score IS NOT NULL
    #                      UNION ALL
    #                      SELECT domain, topic_8_score FROM parquet_scan('{pqfile}')
    #                      WHERE topic_8_score IS NOT NULL
    #                      UNION ALL
    #                      SELECT domain, topic_9_score FROM parquet_scan('{pqfile}')
    #                      WHERE topic_9_score IS NOT NULL
    #                      UNION ALL 
    #                      SELECT domain, topic_10_score FROM parquet_scan('{pqfile}')
    #                      WHERE topic_10_score IS NOT NULL) udomains
    #                 GROUP BY domain) rlvnt_domains
    #            INNER JOIN parquet_scan('{pqfile}') odata
    #                ON rlvnt_domains.domain = odata.domain
    #         '''
    #    df = conn.execute(q).fetchdf()
    #    percentile_cutoff = np.percentile(df['avg_rlvcy'], 5)
    #    del df
    #
    #    with open(f'{FIREHOSE_PATH}/{thisday}/percentile_cutoff.p', 'wb') as f:
    #        pickle.dump({'percentile_cutoff':percentile_cutoff}, f)
    #else:
    #    with open(f'{FIREHOSE_PATH}/{thisday}/percentile_cutoff.p', 'rb') as f:
    #        percentile_cutoff = pickle.load(f)['percentile_cutoff']
    #print(f'{percentile_cutoff=:.5f}')
    
    #q = f'''PRAGMA threads=32;

    #        SELECT odata.*
    #        FROM (SELECT domain, AVG(topic_1_score) AS avg_rlvcy
    #            FROM (SELECT domain, topic_1_score FROM parquet_scan('{pqfile}')
    #                  WHERE topic_1_score IS NOT NULL
    #                  UNION ALL
    #                  SELECT domain, topic_2_score FROM parquet_scan('{pqfile}')
    #                  WHERE topic_2_score IS NOT NULL
    #                  UNION ALL
    #                  SELECT domain, topic_3_score FROM parquet_scan('{pqfile}')
    #                  WHERE topic_3_score IS NOT NULL
    #                  UNION ALL
    #                  SELECT domain, topic_4_score FROM parquet_scan('{pqfile}')
    #                  WHERE topic_4_score IS NOT NULL
    #                  UNION ALL
    #                  SELECT domain, topic_5_score FROM parquet_scan('{pqfile}')
    #                  WHERE topic_5_score IS NOT NULL
    #                  UNION ALL
    #                  SELECT domain, topic_6_score FROM parquet_scan('{pqfile}')
    #                  WHERE topic_6_score IS NOT NULL
    #                  UNION ALL
    #                  SELECT domain, topic_7_score FROM parquet_scan('{pqfile}')
    #                  WHERE topic_7_score IS NOT NULL
    #                  UNION ALL
    #                  SELECT domain, topic_8_score FROM parquet_scan('{pqfile}')
    #                  WHERE topic_8_score IS NOT NULL
    #                  UNION ALL
    #                  SELECT domain, topic_9_score FROM parquet_scan('{pqfile}')
    #                  WHERE topic_9_score IS NOT NULL
    #                  UNION ALL 
    #                  SELECT domain, topic_10_score FROM parquet_scan('{pqfile}')
    #                  WHERE topic_10_score IS NOT NULL) udomains
    #             GROUP BY domain) rlvnt_domains
    #        INNER JOIN parquet_scan('{pqfile}') odata
    #            ON rlvnt_domains.domain = odata.domain
    #        WHERE avg_rlvcy > {percentile_cutoff}
    #     '''
    #df = conn.execute(q).fetchdf()
    #fp.write(f'{FIREHOSE_PATH}/{thisday}/filtered_subsample.pq', df,
    #         compression='SNAPPY')

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
    """Unique source ids with topic labels considered by a firm (i.e. as specified by domain).

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
        q = f'''PRAGMA threads=32;

                COPY (SELECT DISTINCT *
                      FROM (SELECT domain, source_id,
                                topic_1, topic_1_score,
                                topic_2, topic_2_score,
                                topic_3, topic_3_score,
                                topic_4, topic_4_score,
                                topic_5, topic_5_score,
                                topic_6, topic_6_score,
                                topic_7, topic_7_score,
                                topic_8, topic_8_score,
                                topic_9, topic_9_score,
                                topic_10, topic_10_score
                            FROM parquet_scan('{pqfile}')))
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

def setup_cooc(thisday=None, cache=True, subsample=True):
    """Get co-occurrence all pairs of topics and cache to parquet file. This should
    somehow reflect the fact that some connections between topics are overrepresented
    either because employees are reading more about them or because such links appear
    more often in the universe of articles.

    Here, we are just looking at the universe of articles, so we are not counting how
    often such articles are read.
    
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
    cache : bool, True
        If True, save to cache.
    subsample : bool, True
        If True, use filtered subsample instead of entire data set.

    Returns
    -------
    pd.DataFrame
    """

    # otherwise we must do calculation
    # for each unique article, identify the topic labels
    q = '''PRAGMA threads=32;

        CREATE TABLE topics (
          article_id varchar(255),
          topic_1 varchar(255),
          PRIMARY KEY (article_id, topic_1)
        );
        
        INSERT INTO topics
        SELECT DISTINCT article_id, topic_1
        FROM (
        '''
    for i in range(1, 11):
        if not subsample:
            q += f'''SELECT DISTINCT article_id, topic_{i} AS topic_1
                     FROM parquet_scan('{FIREHOSE_PATH}/{thisday}/Redacted*parquet')
                     WHERE topic_{i} IS NOT NULL AND topic_{i} <> ''
                     UNION ALL''' + '\n'
        else:
            q += f'''SELECT DISTINCT article_id, topic_{i} AS topic_1
                     FROM parquet_scan('{FIREHOSE_PATH}/{thisday}/filtered_subsample.pq')
                     UNION ALL''' + '\n'
    q = q.rstrip()[:-9]
    q += ''');\n'''
    
    q += f'''CREATE TABLE results AS
                SELECT topics1.topic_1 AS topic_1, topics2.topic_1 AS topic_2, COUNT(*) AS counts
                FROM topics topics1
                INNER JOIN topics topics2
                   ON topics1.article_id = topics2.article_id AND topics1.topic_1 < topics2.topic_1
                GROUP BY topics1.topic_1, topics2.topic_1;
         '''
    if cache:
        fname = increment_name('cache/pairs_cooc', 'pq')
        q += f'''
              COPY results
              TO '{fname}' (FORMAT 'parquet');
              '''
    db_conn().execute(q)
 
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
