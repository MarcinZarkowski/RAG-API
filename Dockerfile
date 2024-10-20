# Use the official Python image from the Python 3 version
FROM python:3.12.3

# Set the working directory
WORKDIR /app

# Install build dependencies and Rust (only if required)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (to leverage caching)
COPY ./requirements.txt /app/requirements.txt

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Create the directory for NLTK data
RUN mkdir -p /usr/local/share/nltk_data

# Download all required NLTK data (punkt, stopwords, etc.)
RUN python -m nltk.downloader -d /usr/local/share/nltk_data all

# Set NLTK data environment variable
ENV NLTK_DATA=/usr/local/share/nltk_data

# Copy the rest of the application code to the container
COPY . /app

# Expose port 8888
EXPOSE 8888

# Run the FastAPI application using Uvicorn
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8888"]
