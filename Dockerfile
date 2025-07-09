FROM python:3.11.9

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libsndfile1 \
    libgl1 \
    flac \
    cmake \
    ffmpeg \
    --fix-missing \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --default-timeout=100 --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app

ARG PORT=8003
ENV PORT=${PORT}
EXPOSE ${PORT}

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", "--timeout-keep-alive", "300"]