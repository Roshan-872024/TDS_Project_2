# ===================================================================
# Hugging Face Space – FastAPI + Playwright (Chromium) + OpenAI
# Fully compatible with CPU Basic hardware
# ===================================================================

FROM python:3.11-slim

# Prevent Debian interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Chromium
RUN apt-get update && apt-get install -y \
    wget curl unzip ffmpeg git \
    libnss3 libnspr4 \
    libx11-6 libx11-xcb1 libxcomposite1 libxdamage1 libxfixes3 \
    libxcb1 libxrandr2 libxkbcommon0 libxshmfence1 \
    libdrm2 libgbm1 libasound2 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libpango-1.0-0 libpangocairo-1.0-0 \
    libgtk-3-0 libgdk-pixbuf-2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Install Playwright Chromium only (no WebKit/Firefox)
RUN playwright install chromium

WORKDIR /app

COPY receive_requests.py /app/receive_requests.py

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "receive_requests:app", "--host", "0.0.0.0", "--port", "7860"]
