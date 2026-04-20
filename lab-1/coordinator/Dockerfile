FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir nats-py numpy

COPY mapreduce.py matmul.py linreg.py ./
# Pass the script as the docker compose run argument, e.g.:
#   docker compose run --rm coordinator python matmul.py
#   docker compose run --rm coordinator python linreg.py
