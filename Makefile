UV ?= uv

.PHONY: sync sync-all lint test audit-sample verify verify-llm-receipt preload-model smoke-model smoke-classification study-status study-ledger medgemma-readiness medgemma-prepare medgemma-tier-dry-run medgemma-tier-status medgemma-tier-run

sync:
	$(UV) sync

sync-all:
	$(UV) sync --extra reports --extra baselines --extra llm --extra evidence

lint:
	$(UV) run ruff check src/eeg_review tests scripts/smoke_inference_receipt.py scripts/smoke_model.py scripts/smoke_classification.py scripts/preload_model.py scripts/study_job.py scripts/prepare_medgemma_study.py scripts/run_tiered_medgemma_study.py src/LLM_pipeline/llm_models.py

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

preload-model:
	@test -n "$(MODEL)" || (echo "Set MODEL to a registered model name" && exit 2)
	@test -n "$(RECEIPT)" || (echo "Set RECEIPT to the output receipt path" && exit 2)
	$(UV) run --extra llm python scripts/preload_model.py --model "$(MODEL)" --receipt "$(RECEIPT)"

smoke-model:
	@test -n "$(MODEL)" || (echo "Set MODEL to a registered model name" && exit 2)
	@test -n "$(RECEIPT)" || (echo "Set RECEIPT to the output receipt path" && exit 2)
	$(UV) run --extra llm python scripts/smoke_model.py --model "$(MODEL)" --receipt "$(RECEIPT)"

smoke-classification:
	@test -n "$(MODEL)" || (echo "Set MODEL to a registered model name" && exit 2)
	@test -n "$(RECEIPT)" || (echo "Set RECEIPT to the output receipt path" && exit 2)
	$(UV) run --extra llm python scripts/smoke_classification.py --model "$(MODEL)" --receipt "$(RECEIPT)"

study-status:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to the governed run directory" && exit 2)
	$(UV) run --extra reports --extra baselines --extra llm python scripts/study_job.py status --run-dir "$(RUN_DIR)"

study-ledger:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to the governed run directory" && exit 2)
	$(UV) run --extra reports --extra baselines --extra llm python scripts/study_job.py ledger --run-dir "$(RUN_DIR)"

medgemma-readiness:
	@test -n "$(SOURCE_RUN)" || (echo "Set SOURCE_RUN to the completed governed reproduction run" && exit 2)
	@test -n "$(RECEIPT_DIR)" || (echo "Set RECEIPT_DIR to the private MedGemma receipt directory" && exit 2)
	@test -n "$(OUTPUT_DIR)" || (echo "Set OUTPUT_DIR to a governed aggregate receipt directory" && exit 2)
	$(UV) run eeg-review medgemma-study-readiness \
		--plan review/model-receipts/medgemma-independent-comparator.preregistered.json \
		--source-run "$(SOURCE_RUN)" --receipt-dir "$(RECEIPT_DIR)" \
		--check-local --output-dir "$(OUTPUT_DIR)"

medgemma-prepare:
	@test -n "$(SOURCE_RUN)" || (echo "Set SOURCE_RUN to the completed governed reproduction run" && exit 2)
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to a new governed MedGemma run directory" && exit 2)
	$(UV) run --extra reports python scripts/prepare_medgemma_study.py \
		--plan review/model-receipts/medgemma-independent-comparator.preregistered.json \
		--source-run "$(SOURCE_RUN)" --output-dir "$(RUN_DIR)" \
		--acknowledge-governed-output

medgemma-tier-dry-run:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to the governed MedGemma run directory" && exit 2)
	$(UV) run --extra llm python scripts/run_tiered_medgemma_study.py dry-run \
		--run-dir "$(RUN_DIR)" $(TIER_RUN_ARGS)

medgemma-tier-status:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to the governed MedGemma run directory" && exit 2)
	$(UV) run python scripts/run_tiered_medgemma_study.py status \
		--run-dir "$(RUN_DIR)" $(TIER_RUN_ARGS)

medgemma-tier-run:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to the governed MedGemma run directory" && exit 2)
	$(UV) run --extra llm python scripts/run_tiered_medgemma_study.py run \
		--run-dir "$(RUN_DIR)" $(TIER_RUN_ARGS)
