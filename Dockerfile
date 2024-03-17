# Используйте официальный образ Python
FROM python:latest

WORKDIR /app

ADD . /app

RUN pip install poetry

RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

CMD ["python", "flask_app.py"]