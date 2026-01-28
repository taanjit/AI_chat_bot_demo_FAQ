# Dockerfile at project root

FROM python:3.10-slim

WORKDIR /src

# Needed for HEALTHCHECK and some common wheels/build steps
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them first
COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

# Now copy the app code
COPY app/ ./app/

EXPOSE 8088

HEALTHCHECK CMD curl --fail http://localhost:8088/docs || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8088", "--workers=2", "--timeout-keep-alive=60"]

