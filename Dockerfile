FROM python:3.10-slim

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and defaults
COPY . .

# Expose default port
EXPOSE 7860

# Run application
ENV PYTHONUNBUFFERED=1
# Use env PORT if provided, otherwise 7860
CMD ["python", "app.py"]
