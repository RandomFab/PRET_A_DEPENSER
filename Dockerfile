# Stage 1: Builder
FROM python:3.13-slim AS builder
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: Runtime
FROM python:3.13-slim
WORKDIR /app

# On recopie l'exécutable uv pour qu'il soit disponible dans le container final
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
COPY . .
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh
CMD ["./start.sh"]