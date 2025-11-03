FROM python:3.12-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml poetry.lock* ./
RUN pip install poetry && poetry config virtualenvs.create false && poetry install --no-interaction

COPY . .

# Default entrypoint
CMD ["bash"]
