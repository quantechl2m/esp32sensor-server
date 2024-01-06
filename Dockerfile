FROM python:3.9

# Path: /sensorApp
RUN mkdir /sensorApp
WORKDIR /sensorApp

COPY . /sensorApp/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["flask", "run"]