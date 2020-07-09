FROM python:3.8.3-slim-buster

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

COPY ./src /usr/src/app

CMD ["gunicorn", "--bind=0.0.0.0:5000", "--workers=1", "manage:app"]
