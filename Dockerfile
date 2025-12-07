# python:3.12-slim is based on Debian Bookworm, which has SUMO in its repos
FROM python:3.12-slim

# Set environment variables
# 1. Prevent Python from buffering stdout/stderr (so logs show up immediately)
# 2. Set SUMO_HOME (required for SUMO tools to work properly)
ENV PYTHONUNBUFFERED=1 \
    SUMO_HOME=/usr/share/sumo

# Set the working directory in the container
WORKDIR /app

# --- SYSTEM LAYER: Install SUMO ---
# We update apt, install sumo, sumo-tools, and clean up to keep image small
RUN apt-get update && apt-get install -y \
    sumo \
    sumo-tools \
    sumo-doc \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- APP LAYER: Install Python Dependencies ---
# Copy only the requirements first to cache the pip install step
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- COPY CODE ---
# Copy the rest of your application code
COPY . .

# Create directory for outputs (optional, good for mapping volumes)
# RUN mkdir -p /app/docs

# --- RUN ---
# We use 'sumo' (headless) by default in Docker, not 'sumo-gui'
CMD ["python", "TIA_Agent.py"]