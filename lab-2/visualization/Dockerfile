FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir flask matplotlib
COPY . .
EXPOSE 5050
CMD ["python3", "app.py"]
