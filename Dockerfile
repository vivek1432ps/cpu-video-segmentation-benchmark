FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmark.py test_benchmark.py ./
RUN python -m unittest -v
ENTRYPOINT ["python", "benchmark.py"]
