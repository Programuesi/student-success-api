# Use a small official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system deps (optional but helps for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container
COPY . .

# Expose port (for documentation; mapping is done with -p)
EXPOSE 8000

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
