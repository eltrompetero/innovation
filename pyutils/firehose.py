# ====================================================================================== #
# Module for accessing firehose data.
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import duckdb as db
import fastparquet as fp

from .utils import *


def setup(day,
          force=False,
          firehose_path='/home/eddie/Dropbox/Research/corporations/starter_packet/firehose'):
    """
    1. downsample full day.
    2. calculate 5% avg relevancy and truncate
    3. create temp parquet 
    
    Parameters
    ----------
    day : str
    force : bool, False
    firehose_path : str, '/home/eddie/Dropbox/Research/corporations/starter_packet/firehose'
    """
    
    if force or not os.path.isfile(f'{firehose_path}/{thisday}/subsample.pq'):
        # create subsample DF
        q = f'''SELECT *
                FROM parquet_scan('{firehose_path}/{thisday}/all.pq')
                TABLESAMPLE(25 PERCENT)
             '''

        df = conn.execute(q).fetchdf()
        fp.write(f'{firehose_path}/{thisday}/subsample.pq', df, 100_000,
                 compression='SNAPPY')
        del df
    
    pqfile = f'{firehose_path}/{thisday}/subsample.pq'
    if force or not os.path.isfile(f'{firehose_path}/{thisday}/percentile_cutoff.p'):
        # get distribution of relevancy to determine cutoff
        q = f'''SELECT avg_rlvcy
                FROM (SELECT domain, AVG(topic_1_score) AS avg_rlvcy
                    FROM (SELECT domain, topic_1_score FROM parquet_scan('{pqfile}')
                          WHERE topic_1_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_2_score FROM parquet_scan('{pqfile}')
                          WHERE topic_1_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_3_score FROM parquet_scan('{pqfile}')
                          WHERE topic_1_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_4_score FROM parquet_scan('{pqfile}')
                          WHERE topic_1_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_5_score FROM parquet_scan('{pqfile}')
                          WHERE topic_1_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_6_score FROM parquet_scan('{pqfile}')
                          WHERE topic_1_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_7_score FROM parquet_scan('{pqfile}')
                          WHERE topic_1_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_8_score FROM parquet_scan('{pqfile}')
                          WHERE topic_1_score IS NOT NULL
                          UNION ALL
                          SELECT domain, topic_9_score FROM parquet_scan('{pqfile}')
                          WHERE topic_1_score IS NOT NULL
                          UNION ALL 
                          SELECT domain, topic_10_score FROM parquet_scan('{pqfile}')) udomains
                          WHERE topic_1_score IS NOT NULL
                     GROUP BY domain) rlvnt_domains
                INNER JOIN parquet_scan('{pqfile}') odata
                    ON rlvnt_domains.domain = odata.domain
             '''
        df = conn.execute(q).fetchdf()
        percentile_cutoff = np.percentile(df['avg_rlvcy'], 5)
        del df
        save_pickle(['percentile_cutoff'], f'{firehose_path}/{thisday}/percentile_cutoff.p')
    else:
        with open(f'{firehose_path}/{thisday}/percentile_cutoff.p', 'rb') as f:
            percentile_cutoff = pickle.load(f)['percentile_cutoff']
    print(f'{percentile_cutoff=:.5f}')
    
    q = f'''SELECT odata.*
            FROM (SELECT domain, AVG(topic_1_score) AS avg_rlvcy
                FROM (SELECT domain, topic_1_score FROM parquet_scan('{pqfile}')
                      WHERE topic_1_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_2_score FROM parquet_scan('{pqfile}')
                      WHERE topic_1_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_3_score FROM parquet_scan('{pqfile}')
                      WHERE topic_1_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_4_score FROM parquet_scan('{pqfile}')
                      WHERE topic_1_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_5_score FROM parquet_scan('{pqfile}')
                      WHERE topic_1_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_6_score FROM parquet_scan('{pqfile}')
                      WHERE topic_1_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_7_score FROM parquet_scan('{pqfile}')
                      WHERE topic_1_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_8_score FROM parquet_scan('{pqfile}')
                      WHERE topic_1_score IS NOT NULL
                      UNION ALL
                      SELECT domain, topic_9_score FROM parquet_scan('{pqfile}')
                      WHERE topic_1_score IS NOT NULL
                      UNION ALL 
                      SELECT domain, topic_10_score FROM parquet_scan('{pqfile}')) udomains
                      WHERE topic_1_score IS NOT NULL
                 GROUP BY domain) rlvnt_domains
            INNER JOIN parquet_scan('{pqfile}') odata
                ON rlvnt_domains.domain = odata.domain
            WHERE avg_rlvcy > {percentile_cutoff}
         '''
    
    df = conn.execute(q).fetchdf()
    fp.write(f'{firehose_path}/{thisday}/filtered_subsample.pq', df,
             compression='SNAPPY')
