FROM python:3.12-slim

WORKDIR /app
RUN pip install --no-cache-dir nats-py numpy

COPY main.py mapreduce.py ./
CMD ["python", "main.py"]
