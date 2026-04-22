FROM bde2020/hive:2.3.2-postgresql-metastore
# Debian Stretch is EOL — use archive repos
RUN echo "deb http://archive.debian.org/debian stretch main contrib non-free" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security stretch/updates main" >> /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y --allow-unauthenticated python3 && \
    rm -rf /var/lib/apt/lists/*
