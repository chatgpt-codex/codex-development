FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY ../data /data
ENV DATA_PATH=/data/products.csv
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
