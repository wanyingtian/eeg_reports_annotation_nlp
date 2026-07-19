UV ?= uv

.PHONY: sync sync-all lint test audit-sample verify verify-llm-receipt

sync:
	$(UV) sync

sync-all:
	$(UV) sync --extra reports --extra baselines --extra llm --extra evidence

lint:
	$(UV) run ruff check src/eeg_review tests scripts/smoke_inference_receipt.py

test:
	$(UV) run pytest

audit-sample:
	$(UV) run eeg-review audit \
		--dataset data/zoe_reports_sample.db \
		--dataset-id zoe-sample \
		--output-dir outputs/review/sample-audit

verify: lint test audit-sample

verify-llm-receipt:
	$(UV) sync --extra llm
	PYTHONPATH=src/LLM_pipeline $(UV) run python scripts/smoke_inference_receipt.py
