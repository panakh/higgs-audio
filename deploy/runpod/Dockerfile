FROM runpod/pytorch:0.7.0-cu1241-torch240-ubuntu2004

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

RUN apt-get update \

    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \

    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
COPY requirements-runpod.txt ./


RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements-runpod.txt

COPY . .

RUN python -m pip install --no-cache-dir -e .


ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1


CMD ["python", "-u", "-m", "serverless.runpod_worker"]
