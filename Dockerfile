# Base image
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install Python dependencies + DVC
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir dvc[s3]

# Copy application code
COPY flask_app/ /app/

# Copy DVC pipeline config files
COPY *.dvc dvc.lock /app/

# Pass secrets for DVC remote access
ARG CAPSTONE_TEST
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV CAPSTONE_TEST=$CAPSTONE_TEST
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

# Pull all DVC-tracked files from remote
RUN dvc pull

# Expose port
EXPOSE 5001

# Run Flask app (development)
CMD ["python3", "flask_app/app.py"]

# Uncomment for production (Gunicorn)
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "flask_app.app:app"]
