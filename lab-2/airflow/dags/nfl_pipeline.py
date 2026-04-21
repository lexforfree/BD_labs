"""
Airflow DAG: NFL Win Probability Pipeline.

Sequentially runs:
  1. upload_to_hdfs    — preprocess & upload data into HDFS
  2. run_mapreduce     — Hadoop Streaming MapReduce job
  3. run_hive          — Hive HQL aggregation query
  4. run_spark         — PySpark aggregation job
  5. generate_report   — produce timing comparison JSON

Each step is a BashOperator that calls `docker exec` into the relevant
cluster container.  The Airflow container needs /var/run/docker.sock mounted
and docker CLI installed (see airflow/Dockerfile).

Schedule: weekly, so the pipeline re-runs automatically on updated data.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "student",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="nfl_win_probability",
    default_args=default_args,
    description="NFL Win Probability: MapReduce → Hive → Spark → Report",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["nfl", "hadoop", "spark", "bigdata"],
) as dag:

    # Trailing space prevents Airflow from treating the command as a Jinja template file
    # (BashOperator has template_ext=('.sh', '.bash'), so commands ending in .sh get
    # loaded as files unless a non-template character follows)
    upload_to_hdfs = BashOperator(
        task_id="upload_to_hdfs",
        bash_command="docker exec namenode bash /scripts/upload_to_hdfs.sh ",
        doc="Preprocess raw NFL CSVs and put merged file on HDFS.",
    )

    run_mapreduce = BashOperator(
        task_id="run_mapreduce",
        bash_command="docker exec namenode bash /scripts/run_mr.sh ",
        doc="Hadoop Streaming MapReduce: compute win probability by bucket.",
    )

    run_hive = BashOperator(
        task_id="run_hive",
        bash_command="docker exec hiveserver2 bash /scripts/run_hive.sh ",
        doc="Hive HQL query: same aggregation using MapReduce engine.",
    )

    run_spark = BashOperator(
        task_id="run_spark",
        bash_command="docker exec spark bash /scripts/run_spark.sh ",
        doc="PySpark: same aggregation using in-memory Spark execution.",
    )

    generate_report = BashOperator(
        task_id="generate_report",
        bash_command="docker exec visualization python3 /app/plot_results.py",
        doc="Regenerate timing comparison chart and win probability heatmap.",
    )

    # Sequential pipeline: each step depends on the previous
    upload_to_hdfs >> run_mapreduce >> run_hive >> run_spark >> generate_report
