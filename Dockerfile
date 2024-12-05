FROM continuumio/miniconda3

WORKDIR /home
COPY . .
ENV PYTHONPATH=/home

RUN apt-get update
RUN apt-get install nano unzip
RUN apt install curl -y

RUN curl -fsSL https://get.deta.dev/cli.sh | sh

RUN pip3 install -r requirements.txt

RUN pip3 install tensorflow==2.17.1 mlflow

CMD ["python", "app/ai_solution_ml_train.py"]


