FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    DEEPFACE_HOME=/home/user \
    PORT=7860

WORKDIR /home/user/app

COPY --chown=user requirements-web.txt .
RUN pip install --no-cache-dir --user --timeout 300 --retries 5 -r requirements-web.txt

COPY --chown=user . .

RUN mkdir -p dataset && python warmup_models.py

EXPOSE 7860

CMD ["gunicorn", "web_app:app", "--bind", "0.0.0.0:7860", "--timeout", "300", "--workers", "1", "--max-requests", "300", "--max-requests-jitter", "30"]
