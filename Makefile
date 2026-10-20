# Spoorcast. Every target runs inside the uv-managed environment.
UV ?= uv
RUN := $(UV) run
SPOORCAST := $(RUN) spoorcast

MONTHS ?= 2025-09..2026-08
VALIDATE_MONTH ?= 2026-08
REPLAY_MONTH ?= 2026-02
REPLAY_LIMIT ?= 1000
PORT ?= 8000
IMAGE ?= spoorcast-api:latest
MODEL_DIR ?= models/latest

.PHONY: help sync lint format typecheck test download build-dataset train evaluate \
	snapshot-features serve serve-replay replay docker-build docker-run

help:  ## Show the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

sync:  ## Create/update the virtual environment from uv.lock
	$(UV) sync --frozen

lint:  ## Ruff lint + format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:  ## Auto-fix lint issues and format
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck:  ## mypy (a bonus, not a gate)
	$(RUN) mypy src

test:  ## Run the test suite (offline)
	$(RUN) pytest

download:  ## Download services, stations and weather for MONTHS
	$(SPOORCAST) download --months $(MONTHS)

build-dataset:  ## Build the dataset for MONTHS
	$(SPOORCAST) build-dataset --months $(MONTHS)

train:  ## Train on the 12 months before VALIDATE_MONTH, validate on it
	$(SPOORCAST) train --validate-month $(VALIDATE_MONTH) --output $(MODEL_DIR)

evaluate:  ## Evaluate the saved model on VALIDATE_MONTH
	$(SPOORCAST) evaluate --month $(VALIDATE_MONTH) --model-dir $(MODEL_DIR)

snapshot-features:  ## Write the serving feature snapshot (as of the day after the data)
	$(SPOORCAST) snapshot-features

serve:  ## Run the API on PORT (live Open-Meteo forecasts)
	SPOORCAST_MODEL_DIR=$(MODEL_DIR) $(SPOORCAST) serve --port $(PORT)

serve-replay:  ## Run the API for a replay of REPLAY_MONTH (historical weather, matching snapshot)
	$(SPOORCAST) snapshot-features --as-of $(REPLAY_MONTH)-01 \
		--output data/processed/feature_snapshot_$(REPLAY_MONTH)
	SPOORCAST_MODEL_DIR=$(MODEL_DIR) SPOORCAST_WEATHER_PROVIDER=historical \
		SPOORCAST_SNAPSHOT_DIR=data/processed/feature_snapshot_$(REPLAY_MONTH) \
		SPOORCAST_INFERENCE_LOG_PATH=data/inference/replay-$(REPLAY_MONTH).jsonl \
		$(SPOORCAST) serve --port $(PORT)

replay:  ## Replay REPLAY_MONTH against the API on PORT (start `make serve-replay` first)
	$(SPOORCAST) replay --month $(REPLAY_MONTH) --limit $(REPLAY_LIMIT) \
		--api-url http://127.0.0.1:$(PORT)

docker-build:  ## Build the API image
	docker build -t $(IMAGE) .

docker-run:  ## Run the API image with the local model and snapshot mounted
	docker run --rm -p $(PORT):8000 \
		-v $(abspath $(MODEL_DIR)):/models/model:ro \
		-v $(abspath data/processed/feature_snapshot):/models/feature_snapshot:ro \
		$(IMAGE)
