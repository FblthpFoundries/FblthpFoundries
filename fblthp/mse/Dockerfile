FROM ubuntu:latest

RUN apt-get update -y && \
    apt-get install -y \
	wine \
	python3 \
	python3-venv\
	python3-pip


WORKDIR /
COPY MSE/ /MSE
COPY app.py /

RUN python3 -m venv fblthp

RUN fblthp/bin/pip install --upgrade pip 
RUN fblthp/bin/pip --no-cache-dir install flask

RUN apt-get clean

EXPOSE 6969

CMD ["fblthp/bin/python3", "app.py"]
