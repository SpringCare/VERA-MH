FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files and install dependencies
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . .

CMD ["python", "generate.py"]