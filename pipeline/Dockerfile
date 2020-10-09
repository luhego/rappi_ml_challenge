FROM python:3.6-jessie
ENV PYTHONUNBUFFERED 1

RUN mkdir /pip_requirements
ADD requirements.txt /pip_requirements/requirements.txt
RUN pip install -r /pip_requirements/requirements.txt

WORKDIR /code
ADD . /code/
RUN ls /code

EXPOSE 3000
