# Reproducible runs; supports Python 3.10–3.12 images (3.13 when base image available)
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py chainlit.md ./
COPY public ./public
COPY src ./src
COPY .chainlit ./.chainlit

EXPOSE 7860
ENV CHAINLIT_HOST=0.0.0.0
ENV PORT=7860

CMD ["sh", "-c", "chainlit run app.py --host 0.0.0.0 --port ${PORT:-7860}"]
