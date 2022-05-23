# FROM ubuntu:20.04
FROM python:3.8

# RUN apt-get update -y
# RUN apt-get install python3.9-venv -y

WORKDIR /usr/Erdre

# RUN mkdir venv
# RUN python3 -m venv venv
# RUN source venv/bin/activate
RUN pip3 install dvc pandas pandas-profiling sklearn xgboost tensorflow tensorflow-probability edward2 plotly nonconformist
# RUN pip3 install numpy
RUN pip3 install flask flask-restful

RUN mkdir -p assets/data/raw

COPY src ./src
COPY dvc.yaml ./dvc.yaml
COPY params_default.yaml ./params.yaml
COPY params_default.yaml ./params_default.yaml

RUN dvc init --no-scm

CMD ["python3", "src/api.py"]

# RUN python3 src/api.py

# RUN dvc repro