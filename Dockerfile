# Use the Debian-based slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    SUMO_HOME=/usr/share/sumo \
    # Set the Lambda task root to /app to match the path in your traceback
    LAMBDA_TASK_ROOT=/app 

# Set the working directory in the container
WORKDIR ${LAMBDA_TASK_ROOT} # <-- This is now /app

# --- SYSTEM LAYER: Install RIC Dependencies and SUMO from Official Repo ---
# Install build tools, libcurl, git, and the SUMO packages directly from Debian's repos.
RUN apt-get update && apt-get install -y \
    # Dependencies for RIC and general use
    build-essential \
    libcurl4-openssl-dev \
    git \
    # Install SUMO from the default repository (it is available in Debian Bookworm/Trixie)
    sumo \
    sumo-tools \
    sumo-doc \
    # Install Python's RIC
    && pip install --no-cache-dir awslambdaric \
    # Clean up to keep image small
    && rm -rf /var/lib/apt/lists/*

# --- APP LAYER: Install Python Dependencies ---
# Copy requirements.txt first for build cache optimization
COPY requirements.txt .

# Install dependencies (This is where mangum must be installed)
RUN pip install --no-cache-dir -r requirements.txt

# --- COPY CODE ---
# Copy the rest of your application code
COPY . .

# This line is correct as the WORKDIR is now /app
RUN mkdir -p /app/docs

# --- LAMBDA CONFIGURATION ---
# Set the ENTRYPOINT to the Python Runtime Interface Client (RIC)
ENTRYPOINT ["/usr/local/bin/python", "-m", "awslambdaric"]

# Set the CMD to your handler: [file name].[handler variable]
# The RIC will use LAMBDA_TASK_ROOT (now /app) to find main.handler
CMD ["main.handler"]