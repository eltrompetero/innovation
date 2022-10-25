# ====================================================================================== #
# USPTO data handling.
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import pandas as pd
from .utils import db_conn


def uspc_codes(cat):
    """Including categories 1-5.
    
    Parameters
    ----------
    cat : int
    
    Returns
    -------
    list of int
        USPC main classes that belong to the specified category.
    """
    
    assert 1<=cat<=5
    
    if cat==1:
        c = [8, 19, 71, 127, 442, 504, 106, 118, 401, 427, 48, 55, 95, 96, 534, 536, 540, 544, 546, 548, 549,
             552, 554, 556, 558, 560, 562, 564, 568, 570, 520, 521, 522, 523, 524, 525, 526, 527, 528, 530,
             23, 34, 44, 102, 117, 149, 156, 159, 162, 196, 201, 202, 203, 204, 205, 208, 210, 216, 222,
             252, 260, 261, 349, 366, 416, 422, 423, 430, 436, 494, 501, 502, 510, 512, 516, 518, 585, 588]
    elif cat==2:
        c = [178, 333, 340, 342, 343, 358, 367, 370, 375, 379, 385, 455, 341, 380, 382, 395, 700, 701, 702,
                704, 705, 706, 707, 708, 709, 710, 712, 713, 714, 345, 347, 360, 365, 369, 711]
    elif cat==3:
        c = [424, 514, 128, 600, 601, 602, 604, 606, 607, 435, 800, 351, 433, 623]
    elif cat==4:
        c = [174, 200, 327, 329, 330, 331, 332, 334, 335, 336, 337, 338, 392, 439, 313, 314, 315, 362, 372,
             445, 73, 324, 356, 374, 250, 376, 378, 60, 136, 290, 310, 318, 320, 322, 323, 361, 363, 388,
             429, 257, 326, 438, 505, 191, 218, 219, 307, 346, 348, 377, 381, 386]
    elif cat==5:
        c = [65, 82, 83, 125, 141, 142, 144, 173, 209, 221, 225, 226, 234, 241, 242, 264, 271, 407, 408, 409,
                414, 425, 451, 493, 29, 72, 75, 76, 140, 147, 148, 163, 164, 228, 266, 270, 413, 419, 420, 91,
                92, 123, 185, 188, 192, 251, 303, 415, 417, 418, 464, 474, 475, 476, 477, 352, 353, 355, 359,
                396, 399, 104, 105, 114, 152, 180, 187, 213, 238, 244, 246, 258, 280, 293, 295, 296, 298, 301,
                305, 410, 440, 7, 16, 42, 49, 51, 74, 81, 86, 89, 100, 124, 157, 184, 193, 194, 198, 212, 227,
                235, 239, 254, 267, 291, 294, 384, 400, 402, 406, 411, 453, 454, 470, 482, 483, 492, 508]
    else:
        raise NotImplementedError
    df = pd.DataFrame(c, columns=['uspc_class'])
    return df

def setup_cites(select_code, select_year):
    """Set up database connection for extracting citation curves.

    Created data tables include filtered_patents, app_year, grant_year, cites,
    avg_cites_per_year, apps_per_year.

    Parameters
    ----------
    select_code : int, 4
    select_year : int, 1990

    Returns
    -------
    duckdb.DuckDBPyConnection
    """

    codes = uspc_codes(select_code)

    # set up dataframes needed to calculate application citation trajectories for a given tech
    # class with normalization based on the no. of app's per year
    q = f'''PRAGMA threads=32;
         /* records of all citations to a select set of patents within the given technology
            class following Higham et al.
            some patents may have multiple pertinent classes */
            
         /* get all patent id's of patents that fall within the chosen tech category, but remember that 
            patents may appear multiple times if they belong to multiple tech classes */
         CREATE TABLE filtered_patents AS
             SELECT uspc.patent_id, uspc.mainclass_id
             /* ignore patents that fall outside of the normal integer classes since we are
                only considering patents that have been put into the six main categories 
                defined by Hall et al. */
             FROM (SELECT patent_id, CAST(mainclass_id AS INTEGER) AS mainclass_id
                   FROM parquet_scan('../data/uspto/uspc_20220904.pq')
                   WHERE mainclass_id ~ '^[0-9]+') uspc
             INNER JOIN codes
                 ON codes.uspc_class = uspc.mainclass_id;
                 
         /* year of patent application */
         CREATE TABLE app_year AS
             SELECT CAST(app.patent_id AS VARCHAR) AS patent_id,
                 CAST(SUBSTRING(date,1,4) AS INTEGER) AS year
                 /*CAST(REPLACE(SUBSTRING(date,1,7),'-','.') AS DOUBLE) - .01 AS year*/
             FROM parquet_scan('../data/uspto/application_20220904.pq') app
             WHERE year>=1900 and year<=2021;
         
         /* year when patent granted */
         CREATE TABLE grant_year AS
             SELECT CAST(patent_id AS VARCHAR) AS patent_id,
                    year AS year
             FROM parquet_scan('../data/uspto/patent_20220904.pq')
             WHERE year>=1900 and year<=2021;

         /* citation connection between citing and cited patents
            assuming that citation year is equal to when the citing patent's application
            year, but also recorded the year when patent was granted */
         CREATE TABLE cites AS
             SELECT DISTINCT cites.cited_patent_id AS cited,
                             cites.citing_patent_id AS citing,
                             grant_year.year AS citing_grant_year,
                             app_year.year AS citing_app_year,
                             grant_year_cited.year AS cited_grant_year,
                             app_year_cited.year AS cited_app_year,
                             cites.category AS category
             FROM parquet_scan('../data/uspto/uspatentcitation_20220904.pq') cites
             /* only consider cited patents w/in chosen tech category */
             INNER JOIN (SELECT DISTINCT patent_id FROM filtered_patents) AS filtered_patents_cited
                 ON filtered_patents_cited.patent_id = cites.cited_patent_id
             /* only consider citing patents w/in chosen tech category */
             INNER JOIN (SELECT DISTINCT patent_id FROM filtered_patents) AS filtered_patents_citing
                 ON filtered_patents_citing.patent_id = cites.citing_patent_id
             /* grant year of citing patents */
             INNER JOIN grant_year
                 ON cites.citing_patent_id = grant_year.patent_id
             /* app year of citing patents */
             INNER JOIN app_year
                 ON cites.citing_patent_id = app_year.patent_id
             /* grant year of cited patents */
             INNER JOIN grant_year AS grant_year_cited
                 ON cites.cited_patent_id = grant_year_cited.patent_id
             INNER JOIN app_year AS app_year_cited
                 ON cites.cited_patent_id = app_year_cited.patent_id;

         /* avg. no. of citations per application by year
           this is problematic because we can't see the true number of citations
           back into the past b/c many patents cite decades back 
           you can see this by plotting the typical number of citations per app
           against the no. of apps per year */
         CREATE TABLE avg_cites_per_year AS
             SELECT year,
                SUM(count)::FLOAT / COUNT(*) AS norm
             FROM (SELECT cites.cited, COUNT(*) AS count, MODE(cites.cited_app_year) AS year
                   FROM cites
                   GROUP BY cites.cited) thiscites
             GROUP BY year;
             
         /* normalize by the no. of patent applications per year
            this counts all apps across all tech categories unless filtered */
         CREATE TABLE apps_per_year AS
             SELECT year,
                    COUNT(*) AS norm
             FROM app_year
             /* only consider unique entries from filtered_patents, which can have repetition
                from the fact that some patents have multiple classes */
             RIGHT JOIN (SELECT DISTINCT patent_id 
                         FROM filtered_patents) AS filtered_patents
                 ON app_year.patent_id = filtered_patents.patent_id
             WHERE year IS NOT NULL
             GROUP BY year;
         '''
    conn = db_conn()
    conn.execute(q)
    return conn
