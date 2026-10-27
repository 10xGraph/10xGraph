FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app GRAPH_PATH=/app/10xgraph.json

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
# One process on purpose: the demo keeps threads in InMemoryCheckpointer, so an
# interrupt and its resume must reach the same worker. For multiple workers, switch
# to PgCheckpointer and use the gunicorn image `10xgraph build` generates.
CMD ["uvicorn", "tenxgraph_api.src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
