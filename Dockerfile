FROM python:3.8-slim

WORKDIR /home
COPY . .
ENV PYTHONPATH=/home

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libatlas-base-dev \
    libcupti-dev \
    curl \
    unzip \
    nano

RUN apt install curl -y

RUN apt-get install -y cuda


RUN curl -fsSL https://get.deta.dev/cli.sh | sh

RUN pip3 install -r requirements.txt

RUN python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"


CMD ["python3", "app/ai_solution_ml_train.py"]


