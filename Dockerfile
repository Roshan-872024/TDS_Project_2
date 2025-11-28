# ------------------------------
# Dockerfile for Hugging Face
# FastAPI + Playwright + LLM Quiz Solver
# ------------------------------

FROM python:3.11-slim

# Prevent interactive Debian prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    ffmpeg \
    build-essential \
    libglib2.0-0 \
    libnss3 \
    libgdk-pixbuf2.0-0 \
    libgtk-3-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libxshmfence1 \
    libgbm1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Install Playwright + Chromium
RUN playwright install --with-deps chromium

# Copy app
COPY receive_requests.py /app/receive_requests.py

WORKDIR /app

# Hugging Face provides $PORT env var
ENV PORT=7860

EXPOSE 7860

# Start FastAPI
CMD uvicorn receive_requests:app --host 0.0.0.0 --port ${PORT}
