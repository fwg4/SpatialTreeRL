FROM python:3.10-slim

WORKDIR /workspace

# ENV DEBIAN_FRONTEND=noninteractive

# 安裝系統依賴（如果有需要可加）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    fontconfig \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# copy requirements
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt