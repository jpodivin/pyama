FROM ubuntu:jammy

COPY . pyama

RUN apt update && apt install pip git make libcurl4-openssl-dev libssl-dev -y
RUN pip install wheel Flask

RUN pip install ./pyama

ENV FLASK_APP=./pyama/pyama/__init__.py

EXPOSE 5000

CMD flask run --host=0.0.0.0 --debug
