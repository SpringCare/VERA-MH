FROM python:3.14-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Dependencies first, so this layer is cached until uv.lock changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Then the project itself, which installs the `vera` command
COPY . .
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"
CMD ["vera", "--help"]
