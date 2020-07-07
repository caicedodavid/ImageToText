FROM python:3.8.3-slim-buster

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt-get update

COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

COPY ./src /usr/src/app

CMD ["gunicorn", "--bind=0.0.0.0:5000", "--workers=1", "manage:app"]
