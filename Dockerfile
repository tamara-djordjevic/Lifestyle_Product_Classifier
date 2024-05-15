FROM python:3.11

WORKDIR /code


COPY ./requirements.txt /code/requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

CMD ["sh", "-c", "uvicorn app.main:app --port=8003 --host=0.0.0.0 --workers=1"]