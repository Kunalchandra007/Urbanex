FROM python:3.11-slim-bookworm

WORKDIR /app

# Optimize pip for speed
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --no-compile -r requirements.txt && \
    rm -rf /root/.cache/pip/* /root/.cache/bytecode* && \
    find /usr/local/lib -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy application code
COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 7860

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860", "--log-level", "info"]
