# Use official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port Gradio will use
EXPOSE 7860

ENV GRADIO_SERVER_NAME="0.0.0.0"


# Command to run the application
CMD ["python", "test.py"]
