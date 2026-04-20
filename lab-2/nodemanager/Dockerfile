FROM bde2020/hadoop-nodemanager:2.0.0-hadoop3.2.1-java8
# Python3 is required by YARN task containers when running Python streaming jobs
RUN apt-get update && apt-get install -y python3 && rm -rf /var/lib/apt/lists/*
