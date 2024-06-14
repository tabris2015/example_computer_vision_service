# Sample Computer Vision Service

## Running locally
Create virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run development server with Uvicorn:

```bash
uvicorn src.app:app --port 8000 --reload
```

Test the service locally through: http://127.0.0.1:8000/docs

## Running with Docker
Start the service with docker compose:

```bash
docker compose up
```

