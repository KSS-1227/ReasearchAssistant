FROM python:3.11-slim

# Install system dependencies needed by some packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh \
    && chmod 755 /app/entrypoint.sh \
    && ls -la /app/entrypoint.sh

# Copy and install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose both Streamlit and FastAPI ports
EXPOSE 8501 8000

# Health check (check Streamlit)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run entrypoint script that starts both services
ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]