# https://github.com/docker/awesome-compose/blob/master/flask/app/Dockerfile
# https://hub.docker.com/_/python
FROM python:3.12.9
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["python"]
CMD ["app.py"]
