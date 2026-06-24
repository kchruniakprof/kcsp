# Stage 1: frontend build
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY webapp/frontend/package*.json ./
RUN npm ci --legacy-peer-deps
COPY webapp/frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.11-slim AS runtime
WORKDIR /app

# Install Python deps
COPY webapp/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY webapp/backend/ ./webapp/backend/
COPY webapp/__init__.py ./webapp/__init__.py
COPY parquet/ ./parquet/

# Copy built frontend
COPY --from=frontend-build /app/frontend/../backend/static ./webapp/backend/static

VOLUME ["/data"]
ENV DB_PATH=/data/kcsp.db
ENV PORT=8000
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
