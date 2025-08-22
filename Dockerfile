# Base image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY flask_app/ /app/

# Copy data and models (they are already pulled by CI)
COPY data/ /app/data/
COPY models/ /app/models/

# Expose port
EXPOSE 5001

# Run Flask app (development)
CMD ["python3", "app.py"]

# Uncomment for production (Gunicorn)
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "flask_app.app:app"]
