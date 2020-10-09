FROM python:3.6-jessie
ENV PYTHONUNBUFFERED 1

RUN mkdir /pip_requirements
ADD api/requirements.txt /pip_requirements/requirements.txt
RUN pip install -r /pip_requirements/requirements.txt

WORKDIR /code/
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV PYTHONPATH="$PYTHONPATH:/code/pipeline/"

ADD api /code/
ADD pipeline /code/pipeline
ADD data /code/data
ADD artifacts /code/artifacts

RUN ls /code


EXPOSE 5000
CMD ["flask", "run"]
