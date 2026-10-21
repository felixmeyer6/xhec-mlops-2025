FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /web_service

COPY requirements_app.txt /web_service/requirements_app.txt
RUN pip install --no-cache-dir -r requirements_app.txt \
    && useradd --create-home app

COPY --chown=app:app web_service/ /web_service/
# Fail the build early if the required, git-ignored artifacts were not exported.
RUN test -s local_models/pipeline.joblib && test -s local_models/metadata.json

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
