# Use official Python image as base
FROM python:3.9-slim

# Add certificates
RUN apt-get update && apt-get install -y ca-certificates

# Set working directory
WORKDIR /stock_app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "stock_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
