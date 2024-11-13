# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir python-graphblas[default] numpy==1.26.4 ipython

# Copy the current directory contents into the container at /app
COPY . /app

# Run Python when the container launches
CMD ["/usr/local/bin/python", "sssp.py"]
