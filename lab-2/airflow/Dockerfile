FROM apache/airflow:2.9.1
USER root
# Docker CLI is needed so BashOperator can run `docker exec` into cluster containers
RUN apt-get update && \
    apt-get install -y docker.io && \
    rm -rf /var/lib/apt/lists/* && \
    usermod -aG docker airflow
USER airflow
RUN pip install --no-cache-dir apache-airflow-providers-docker==3.9.2
