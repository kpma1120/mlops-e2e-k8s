FROM python:3.12-slim

COPY src /src
COPY fastapi_api .
COPY artifacts /artifacts

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
