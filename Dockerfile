# GRPO-Guard CPU demo image: verify the evidence chain + Streamlit panel.
# Build:  docker build -t grpo-guard .
# Verify: docker compose run --rm verify
# Panel:  docker compose up panel
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY tests ./tests
COPY configs ./configs

RUN pip install --no-cache-dir . && \
    pip install --no-cache-dir streamlit pandas

COPY examples ./examples

ENTRYPOINT ["grpo-guard"]
CMD ["--help"]
