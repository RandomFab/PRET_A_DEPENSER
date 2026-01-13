FROM python:3.13-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY . .
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh
CMD ["./start.sh"]