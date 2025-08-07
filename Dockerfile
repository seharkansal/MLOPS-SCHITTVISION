FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5001

# Command to run your Flask app
CMD ["python3", "flask_app/app.py"]

#Prod
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]