FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Make sure the project root is in PYTHONPATH so all imports resolve
ENV PYTHONPATH=/app

EXPOSE 7860

# Use uvicorn with workers for production-level stability
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
