.PHONY: help install test lint format run-api run-ui docker-up docker-down health seed

help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies in editable mode"
	@echo "  test        - Run pytest test suite"
	@echo "  lint        - Check code style with ruff"
	@echo "  format      - Format code with ruff and black"
	@echo "  run-api     - Start FastAPI backend"
	@echo "  run-ui      - Start Streamlit UI"
	@echo "  docker-up   - Launch Docker services (PostgreSQL, Qdrant, API, Frontend)"
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

run-ui:
	streamlit run frontend/streamlit_app.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down

health:
	python scripts/health_check.py

seed:
	python scripts/seed_data.py
