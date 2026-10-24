# Backend image: API, scoring workers, graph worker, simulator, trainer, retrainer
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY core core
COPY data_gen data_gen
COPY pipeline pipeline
COPY models models
COPY graph graph
COPY api api
COPY simulator simulator
COPY scripts scripts
COPY tests tests
COPY pyproject.toml .
RUN mkdir -p data models/artifacts
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
