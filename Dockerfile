FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

USER 65534:65534
EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "speech_analyzer.main:app", "--host", "0.0.0.0", "--port", "8000"]
