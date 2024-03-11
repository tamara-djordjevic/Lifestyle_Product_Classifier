FROM python:3.9

WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./app /code/app

CMD ["sh", "-c", "uvicorn app.main:app --port=8003 --host=0.0.0.0 --workers=1"]