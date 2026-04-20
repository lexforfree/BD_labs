FROM r-base:4.3.3

RUN Rscript -e "install.packages('jsonlite', repos='https://cran.r-project.org')"

COPY worker.R /app/worker.R
WORKDIR /app
CMD ["Rscript", "worker.R"]
