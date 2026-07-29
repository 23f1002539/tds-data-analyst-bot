FROM python:3.11-slim

WORKDIR /app

# system deps for lxml/pandas/openpyxl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libxml2 libxslt1-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# .env is loaded by python-dotenv at runtime; on Cloud Run/CE, prefer env vars
CMD ["python", "bot.py"]