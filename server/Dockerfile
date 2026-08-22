FROM python:3.11-slim

WORKDIR /app

# CPU-only torch: this container only encodes one question at a time, GPU
# passthrough would add nvidia-container-toolkit complexity for no benefit.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# s3_reproduction is copied in (not pip-installed) so joblib.load can resolve
# s3_reproduction.checkpoint.ModelCheckpoint -- the class path is baked into
# the pickled .joblib file at save time.
COPY s3_reproduction /app/s3_reproduction
COPY server/main.py /app/server_main.py

ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "server_main:app", "--host", "0.0.0.0", "--port", "8000"]
