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

# Copy the app code
COPY app/ ./app/

# Copy the .env file (contains GROQ_API_KEY and other secrets)
COPY app/.env ./app/.env

# IMPORTANT: Copy the vector store into the Docker image.
# The vector store must be pre-generated from the data_document/ folder
# using: python -m app.embedding_generation
# The FASTApi app reads it from ./app/vectorstores/smart_iot_vector_store/
COPY app/vectorstores/ ./app/vectorstores/

EXPOSE 8088

HEALTHCHECK CMD curl --fail http://localhost:8088/docs || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8088", "--workers=2", "--timeout-keep-alive=60"]
