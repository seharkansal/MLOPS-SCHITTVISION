FROM python:3.12-slim

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc[s3]

# Copy DVC-pulled data (must exist before build)
COPY ./data ./data

# # Set AWS credentials as build args
# ARG AWS_ACCESS_KEY_ID
# ARG AWS_SECRET_ACCESS_KEY
# ARG AWS_DEFAULT_REGION

# ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
# ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
# ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION

# # Pull DVC data from S3
# RUN dvc pull --force

# Expose port for Flask
EXPOSE 5001

# Run Flask app
CMD ["python3", "-m", "flask_app.app"]

# Production
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
