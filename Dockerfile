# FROM python:3.10

# COPY ./closetx /closetx
# COPY ./celery_app /celery_app
# WORKDIR /closetx

# RUN apt-get update && apt-get install -y libpq-dev build-essential
# RUN apt-get install -y default-libmysqlclient-dev
# RUN apt-get install -y default-mysql-client
# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# WORKDIR /

# CMD ["python3", "-m", "closetx.app.main", "run", "--host", "0.0.0.0", "--port", "8000"]


FROM python:3.10

COPY ./celery_app /celery_app
COPY ./closetx /closetx
COPY ./closetx/.env /celery_app/.env
WORKDIR /celery_app

RUN apt-get update && apt-get install -y libpq-dev build-essential
RUN apt-get install -y default-libmysqlclient-dev
RUN apt-get install -y default-mysql-client
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR /

CMD ["python3", "-m", "celery", "-A", "celery_app.app", "worker"]

