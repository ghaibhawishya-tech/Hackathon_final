FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Environment var for FastAPI port
ENV PORT=7860

# Expose port
EXPOSE 7860

# Run FastAPI server
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
