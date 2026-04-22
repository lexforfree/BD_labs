-- Create Hive external table over the preprocessed NFL CSV in HDFS.
-- The CSV has a single header row; skip.header.line.count handles it.

CREATE DATABASE IF NOT EXISTS nfl;
USE nfl;

DROP TABLE IF EXISTS nfl_pbp;

CREATE EXTERNAL TABLE nfl_pbp (
    qtr                INT     COMMENT 'Quarter (1-4)',
    down               INT     COMMENT 'Down (1-4)',
    ydstogo            FLOAT   COMMENT 'Yards to first down',
    yardline_100       INT     COMMENT 'Yards from own end zone (1-99)',
    score_differential INT     COMMENT 'Possession team score minus opponent',
    win                INT     COMMENT '1 if possession team won, 0 otherwise'
)
ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION 'hdfs://namenode:9000/data/nfl/processed'
TBLPROPERTIES (
    'skip.header.line.count' = '1'
);

-- Quick sanity check
SELECT COUNT(*) AS total_plays FROM nfl_pbp;
