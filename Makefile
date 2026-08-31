UV ?= uv

.PHONY: sync sync-all lint test audit-sample verify verify-llm-receipt preload-model smoke-model smoke-classification study-status study-ledger medgemma-readiness medgemma-prepare medgemma-tier-dry-run medgemma-tier-status medgemma-tier-run medgemma-native-authorization-check medgemma-native-prepare medgemma-native-dry-run medgemma-native-launch medgemma-native-finalize medgemma-native-author-bundle governed-run-eclipse

sync:
	$(UV) sync

sync-all:
	$(UV) sync --extra reports --extra baselines --extra llm --extra evidence

lint:
	$(UV) run ruff check scripts/mistral_interface_followup.py scripts/run_fixed_classification_explanations.py scripts/medgemma_prompt_v2.py
	$(UV) run ruff check src/eeg_review tests scripts/smoke_inference_receipt.py scripts/smoke_model.py scripts/smoke_classification.py scripts/preload_model.py scripts/study_job.py scripts/prepare_medgemma_study.py scripts/run_tiered_medgemma_study.py scripts/medgemma_mission_control.py scripts/benchmark_medgemma_runtime.py scripts/finalize_medgemma_result_candidate.py scripts/finalize_medgemma_native_development.py scripts/finalize_medgemma_native_protected_result.py scripts/render_medgemma_native_author_bundle.py scripts/check_medgemma_native_protected_authorization.py scripts/eclipse_governed_run.py src/LLM_pipeline/llm_models.py

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

medgemma-native-authorization-check:
	@test -n "$(AUTHORIZATION)" || (echo "Set AUTHORIZATION to the governed documentary receipt" && exit 2)
	@test -n "$(UNLOCK_RECEIPT)" || (echo "Set UNLOCK_RECEIPT to a governed output path" && exit 2)
	PYTHONPATH=src $(UV) run python scripts/check_medgemma_native_protected_authorization.py \
		--authorization "$(AUTHORIZATION)" --output "$(UNLOCK_RECEIPT)"

medgemma-native-prepare:
	@test -n "$(SOURCE_RUN)" || (echo "Set SOURCE_RUN to the governed source run" && exit 2)
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to a new governed native run" && exit 2)
	@test -n "$(AUTHORIZATION)" || (echo "Set AUTHORIZATION to the governed receipt" && exit 2)
	$(UV) run --extra reports python scripts/prepare_medgemma_study.py \
		--plan review/model-receipts/medgemma-native-protected-comparator.preregistered.json \
		--source-run "$(SOURCE_RUN)" --output-dir "$(RUN_DIR)" \
		--runtime-amendment review/model-receipts/medgemma-metal-runtime-amendment.promoted.json \
		--authorization "$(AUTHORIZATION)" --acknowledge-governed-output

medgemma-native-dry-run:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to the governed native run" && exit 2)
	@test -n "$(AUTHORIZATION)" || (echo "Set AUTHORIZATION to the governed receipt" && exit 2)
	$(UV) run --extra llm python scripts/run_tiered_medgemma_study.py dry-run \
		--run-dir "$(RUN_DIR)" \
		--tier-plan review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json \
		--authorization "$(AUTHORIZATION)" $(TIER_RUN_ARGS)

medgemma-native-launch:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to the governed native run" && exit 2)
	@test -n "$(AUTHORIZATION)" || (echo "Set AUTHORIZATION to the governed receipt" && exit 2)
	$(UV) run --extra llm python scripts/run_tiered_medgemma_study.py launch \
		--run-dir "$(RUN_DIR)" \
		--tier-plan review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json \
		--authorization "$(AUTHORIZATION)" $(TIER_RUN_ARGS)

medgemma-native-finalize:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to the completed governed native run" && exit 2)
	@test -n "$(AUTHORIZATION)" || (echo "Set AUTHORIZATION to the governed receipt" && exit 2)
	@test -n "$(RESULT_CANDIDATE)" || (echo "Set RESULT_CANDIDATE to the aggregate output" && exit 2)
	$(UV) run python scripts/finalize_medgemma_native_protected_result.py \
		--run-dir "$(RUN_DIR)" \
		--study-plan review/model-receipts/medgemma-native-protected-comparator.preregistered.json \
		--tier-plan review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json \
		--authorization "$(AUTHORIZATION)" --output "$(RESULT_CANDIDATE)"

medgemma-native-author-bundle:
	@test -n "$(RESULT_CANDIDATE)" || (echo "Set RESULT_CANDIDATE to the aggregate receipt" && exit 2)
	@test -n "$(AUTHOR_BUNDLE_DIR)" || (echo "Set AUTHOR_BUNDLE_DIR to the output directory" && exit 2)
	$(UV) run python scripts/render_medgemma_native_author_bundle.py \
		--candidate "$(RESULT_CANDIDATE)" --output-dir "$(AUTHOR_BUNDLE_DIR)" \
		$(if $(strip $(ADMISSION)),--admission "$(ADMISSION)",)

governed-run-eclipse:
	@test -n "$(RUN_DIR)" || (echo "Set RUN_DIR to the governed run directory" && exit 2)
	@test -n "$(ACTOR)" || (echo "Set ACTOR to the recorded actor" && exit 2)
	@test -n "$(REASON)" || (echo "Set REASON to the governance reason" && exit 2)
	PYTHONPATH=src $(UV) run python scripts/eclipse_governed_run.py \
		--run-dir "$(RUN_DIR)" --actor "$(ACTOR)" --reason "$(REASON)"
