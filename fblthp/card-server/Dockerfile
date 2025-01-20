FROM python:latest

RUN mkdir /server/
COPY ./requirements.txt /server/requirements.txt
RUN pip install -r /server/requirements.txt

EXPOSE 5001

WORKDIR /server/


