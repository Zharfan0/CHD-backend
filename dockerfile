FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (kalau perlu)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dulu (supaya di-cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua file project
COPY . .

# Railway inject PORT environment variable secara otomatis
EXPOSE 8000

# Start server
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]