FROM python:3.12-slim

ENV PORT 8000

# opencv and pillow image dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 gcc python3-dev -y

# install pytorch CPU
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src src

CMD uvicorn src.app:app --host 0.0.0.0 --port ${PORT}