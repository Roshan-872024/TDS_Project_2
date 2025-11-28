# ------------------------------
# Dockerfile for Hugging Face
# FastAPI + Playwright + Chromium
# ------------------------------

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by Playwright Chromium
RUN apt-get update && apt-get install -y \
    curl wget git ffmpeg build-essential \
    libglib2.0-0 libnss3 libatk1.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libasound2 libpangocairo-1.0-0 libpango-1.0-0 \
    libxshmfence1 libgbm1 libgtk-3-0 \
    fonts-unifont \
    fonts-dejavu-core fonts-dejavu-extra \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Install Playwright browsers (NO system deps)
RUN playwright install chromium

# Copy app
COPY receive_requests.py /app/receive_requests.py
COPY README.md /app/README.md

WORKDIR /app

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "receive_requests:app", "--host", "0.0.0.0", "--port", "7860"]
