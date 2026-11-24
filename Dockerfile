# AI helped generate this code

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"

COPY app.py ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]