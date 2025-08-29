# ---- Base image ----
    FROM python:3.13-slim

    # Prevent Python from writing .pyc files & buffer logs
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    
    # (Optional) set a timezone for logs
    ENV TZ=UTC
    
    # Create app directory
    WORKDIR /app
    
    # System deps (compile wheels faster, cleanup after)
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl && \
        rm -rf /var/lib/apt/lists/*
    
    # ---- Python deps (layer cached) ----
    COPY requirements.txt /app/requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ---- Copy your code & artifacts ----
    # Only copy what the API needs to run
    COPY app /app/app
    COPY src /app/src
    COPY models /app/models
    COPY data/processed /app/data/processed
    
    # Expose FastAPI port
    EXPOSE 8000
    
    # Healthcheck (simple ping)
    HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1
    
    # ---- Run the API ----
    CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    