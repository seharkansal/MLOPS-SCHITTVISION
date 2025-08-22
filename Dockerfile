FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir dvc[s3]

# Copy app code
COPY flask_app/ /app/

# Copy DVC outputs
COPY data/label_encoder.pkl /app/data/label_encoder.pkl
COPY models/ /app/models/

# Expose port
EXPOSE 5001

# Run Flask (adjust if needed)
CMD ["python3", "flask_app/app.py"]

# For production (uncomment if using Gunicorn)
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "flask_app.app:app"]
