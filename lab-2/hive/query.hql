-- Win Probability by game situation.
-- Groups plays into (qtr, down, score_diff_bucket, field_position_bucket)
-- and computes empirical win probability = wins / total plays.

USE nfl;

SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.execution.engine=mr;

INSERT OVERWRITE DIRECTORY 'hdfs://namenode:9000/output/hive_win_prob'
ROW FORMAT DELIMITED FIELDS TERMINATED BY '\t'
SELECT
    qtr,
    down,
    CASE
        WHEN score_differential <= -14 THEN 'le-14'
        WHEN score_differential <= -7  THEN '-13to-7'
        WHEN score_differential <= -1  THEN '-6to-1'
        WHEN score_differential  = 0   THEN '0'
        WHEN score_differential <= 6   THEN '1to6'
        WHEN score_differential <= 13  THEN '7to13'
        ELSE 'ge14'
    END                                         AS score_diff_bucket,
    CASE
        WHEN yardline_100 <= 25 THEN '0-25'
        WHEN yardline_100 <= 50 THEN '26-50'
        WHEN yardline_100 <= 75 THEN '51-75'
        ELSE '76-100'
    END                                         AS field_bucket,
    SUM(win)                                    AS wins,
    COUNT(*)                                    AS total,
    ROUND(SUM(win) / COUNT(*), 4)               AS win_probability
FROM nfl_pbp
WHERE qtr  BETWEEN 1 AND 4
  AND down BETWEEN 1 AND 4
GROUP BY
    qtr,
    down,
    CASE
        WHEN score_differential <= -14 THEN 'le-14'
        WHEN score_differential <= -7  THEN '-13to-7'
        WHEN score_differential <= -1  THEN '-6to-1'
        WHEN score_differential  = 0   THEN '0'
        WHEN score_differential <= 6   THEN '1to6'
        WHEN score_differential <= 13  THEN '7to13'
        ELSE 'ge14'
    END,
    CASE
        WHEN yardline_100 <= 25 THEN '0-25'
        WHEN yardline_100 <= 50 THEN '26-50'
        WHEN yardline_100 <= 75 THEN '51-75'
        ELSE '76-100'
    END
HAVING COUNT(*) >= 10
ORDER BY qtr, down, score_diff_bucket, field_bucket;
