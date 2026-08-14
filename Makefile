UV ?= uv

.PHONY: sync sync-all lint test audit-sample verify verify-llm-receipt study-status study-ledger

sync:
	$(UV) sync

sync-all:
	$(UV) sync --extra reports --extra baselines --extra llm --extra evidence

lint:
	$(UV) run ruff check src/eeg_review tests scripts/smoke_inference_receipt.py scripts/study_job.py

test:
	$(UV) run --extra baselines pytest

audit-sample:
	$(UV) run eeg-review audit \
		--dataset data/zoe_reports_sample.db \
		--dataset-id zoe-sample \
		--output-dir outputs/review/sample-audit

verify: lint test audit-sample

verify-llm-receipt:
	$(UV) sync --extra llm
	PYTHONPATH=src/LLM_pipeline $(UV) run python scripts/smoke_inference_receipt.py

study-status:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to the governed run directory" && exit 2)
	$(UV) run --extra reports --extra baselines --extra llm python scripts/study_job.py status --run-dir "$(RUN_DIR)"

study-ledger:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to the governed run directory" && exit 2)
	$(UV) run --extra reports --extra baselines --extra llm python scripts/study_job.py ledger --run-dir "$(RUN_DIR)"
