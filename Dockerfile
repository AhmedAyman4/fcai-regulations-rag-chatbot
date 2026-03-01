# Use a stable Python base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
# Make sure to include the PDF file
COPY app.py .
COPY ["data/USC Faculty of Computer and Artificial Intelligence Internal Regulations (October 2019).pdf", "data/"]

# Set the port environment variable (Hugging Face Spaces default)
ENV PORT=7860

# Expose the port
EXPOSE 7860

# Set the command to run the application using Uvicorn
# This is a production-ready ASGI web server for FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]