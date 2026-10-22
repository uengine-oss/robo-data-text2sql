.PHONY: help install start neo4j stop clean test lint format

help:
	@echo "Neo4j Text2SQL - Available commands:"
	@echo "  make install    - Install dependencies with uv"
	@echo "  make neo4j      - Start Neo4j container"
	@echo "  make init       - Initialize Neo4j schema"
	@echo "  make start      - Start API server"
	@echo "  make stop       - Stop all services"
	@echo "  make clean      - Clean up containers and volumes"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
	@echo "  make ingest     - Run schema ingestion (requires running API)"

install:
	uv sync

neo4j:
	docker-compose up -d
	@echo "Waiting for Neo4j to be ready..."
	@sleep 10
	@echo "✓ Neo4j is running at http://localhost:7474"

init: neo4j
	uv run python scripts/init_schema.py

start:
	uv run python main.py

stop:
	docker-compose down

clean:
	docker-compose down -v
	rm -rf .venv __pycache__ **/__pycache__

test:
	# FR-003(spec 007 T0): 이전 경로 `tests/` 는 존재하지 않아 이 타깃이 항상 실패했다.
	# PYTHONPATH=. 가 없으면 `app` 패키지를 찾지 못한다(pyproject 에 pytest 설정이 없다).
	# 제외한 파일은 존재하지 않는 모듈을 import 하는 사전 파손 상태다(수집 단계에서 죽는다).
	PYTHONPATH=. uv run pytest app/tests -v --ignore=app/tests/react/test_intent_keyword_json_parsing.py

lint:
	uv run ruff check app/

format:
	uv run ruff format app/

ingest:
	@echo "Make sure API is running first!"
	curl -X POST "http://localhost:8000/ingest" \
		-H "Content-Type: application/json" \
		-d '{"db_name": "postgres", "schema": "public", "clear_existing": true}'

health:
	curl http://localhost:8000/health | jq .

# Quick setup for first time
setup: install neo4j init
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Edit .env file with your configuration"
	@echo "  2. Run 'make start' to start the API"
	@echo "  3. Run 'make ingest' to load your schema"
	@echo "  4. Visit http://localhost:8000/docs"

# Test with Docker Compose PostgreSQL
test-setup:
	@echo "Setting up test environment..."
	@if [ ! -f .env ]; then \
		cp env.test.example .env; \
		echo "✓ Created .env from env.test.example"; \
		echo "⚠️  Please edit .env and add your OPENAI_API_KEY (and OPENAI_COMPATIBLE_API_KEY if using llm_provider=openai_compatible)"; \
	else \
		echo "⚠️  .env already exists, skipping..."; \
	fi
	$(MAKE) setup
	@echo ""
	@echo "🧪 Test environment ready!"
	@echo ""
	@echo "PostgreSQL test database:"
	@echo "  - Host: localhost:5432"
	@echo "  - Database: testdb"
	@echo "  - User: readonly / Password: readonly123"
	@echo ""
	@echo "Sample data includes:"
	@echo "  - 8 categories"
	@echo "  - 50 products"
	@echo "  - 30 customers"
	@echo "  - 30 orders with items"
	@echo "  - 50 reviews"

test-db:
	@echo "Testing PostgreSQL connection..."
	docker exec -it postgres_text2sql psql -U testuser -d testdb -c "SELECT 'PostgreSQL is ready!' AS status;"
	@echo ""
	@echo "Sample queries:"
	@echo "  docker exec -it postgres_text2sql psql -U testuser -d testdb"

test-query:
	@echo "Running sample test queries..."
	@echo ""
	@echo "1. Simple query - List customers"
	curl -s -X POST "http://localhost:8000/ask" \
		-H "Content-Type: application/json" \
		-d '{"question": "고객 목록 5명만 보여줘", "limit": 5}' | jq '.table.rows'
	@echo ""
	@echo "2. Aggregation - Category sales"
	curl -s -X POST "http://localhost:8000/ask" \
		-H "Content-Type: application/json" \
		-d '{"question": "카테고리별 상품 수", "limit": 10}' | jq '.table'

