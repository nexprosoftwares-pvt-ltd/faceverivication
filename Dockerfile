FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dlib-bin BEFORE everything else
RUN pip install --no-cache-dir dlib-bin==19.24.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . .

ENV PORT=8080
CMD ["python", "app.py"]
