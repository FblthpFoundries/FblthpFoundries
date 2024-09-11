FROM python:latest

RUN mkdir /server/
COPY ./requirements.txt /server/requirements.txt
RUN apt-get update && apt-get install -y wkhtmltopdf

EXPOSE 5001

WORKDIR /server/


