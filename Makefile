.PHONY: help install test lint format run-api run-ui web-install web-build mlflow-sync mlflow-ui docker-up docker-down health seed

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies in editable mode"
	@echo "  test        - Run pytest test suite"
	@echo "  lint        - Check code style with ruff"
	@echo "  format      - Format code with ruff and black"
	@echo "  run-api     - Start FastAPI backend"
	@echo "  web-install - Install the React interface's packages (web/)"
	@echo "  run-ui      - Start the React interface in development mode (http://localhost:5173)"
	@echo "  web-build   - Build the React interface (served by the API at http://localhost:8000)"
	@echo "  mlflow-sync - Log every evaluation of data/evaluation/ into MLflow"
	@echo "  mlflow-ui   - Start the MLflow interface (http://localhost:5000)"
	@echo "  docker-up   - Launch Docker services (PostgreSQL, Qdrant, n8n)"
	@echo "  docker-down - Stop Docker services"
	@echo "  health      - Run system health checks"
	@echo "  seed        - Generate synthetic sample data"

install:
	pip install --upgrade pip
	pip install -e ".[all]"

test:
	pytest tests -v

lint:
	ruff check .

format:
	ruff format .
	black .

run-api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

web-install:
	cd web && npm install

run-ui:
	cd web && npm run dev

web-build:
	cd web && npm run build

mlflow-sync:
	python scripts/mlflow_sync.py

mlflow-ui:
	python -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

docker-up:
	docker compose up -d

docker-down:
	docker compose down

health:
	python scripts/health_check.py

seed:
	python scripts/seed_data.py
