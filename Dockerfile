FROM continuumio/miniconda3

WORKDIR /home
COPY . .
ENV PYTHONPATH=/home

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

RUN curl -fsSL https://get.deta.dev/cli.sh | sh

RUN pip3 install -r requirements.txt

CMD ["python", "app/air_quality_ml_train.py"]   