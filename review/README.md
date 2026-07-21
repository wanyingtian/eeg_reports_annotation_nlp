# JBHI revision execution toolkit

This directory turns the comments for manuscript `JBHI-02463-2026` into a
data-custodian-run workflow. The code branch contains commands and tests; it
does not contain the governed Zoe/Maria cohorts, patient keys, model weights,
credentials, or case-level reviewer exports.

## Responsibility boundary

- Steven prepares and reviews code, documentation, aggregate output schemas,
  manuscript mappings, and reproducibility receipts.
- Wanying Tian verifies the historical experimental design, prompt-development
  history, annotation protocol, and which source/results match the submission.
- Vasily Vakorin or an explicitly authorized group member runs the workflow on
  governed infrastructure and returns aggregate receipts.
- The PI/data custodian and clinical/REB owners approve data use, ethics
  language, and clinical interpretation. Code output cannot substitute for
  those approvals.

## Current toolchain

The canonical environment is `pyproject.toml` plus `uv.lock`, using Python
3.11. The original `requirements.txt` remains as a historical compatibility
entry point while the revision commands migrate to named optional dependency
groups.

```bash
uv sync                         # aggregate review tools + tests
uv sync --extra reports         # tables, figures, parquet/excel support
uv sync --extra baselines       # BERT/BoW training and SHAP
uv sync --extra llm             # llama.cpp inference pipeline
uv sync --extra evidence        # explanation support/alignment
make verify                     # lint, tests, and sample aggregate audit
make verify-llm-receipt         # fake-model test; downloads no weights
```

The same locked core verification runs in GitHub Actions. The workflow has
read-only repository permissions and does not run models or access governed
data.

On FIR/Alliance, load a supported compiler/CUDA module before installing the
`llm` group. Record the exact modules and driver in the run receipt; do not
silently substitute a different model file or quantization.

## Phase 1: read-only cohort receipts

The audit opens SQLite in read-only mode and emits aggregate JSON/CSV only.

```bash
uv run eeg-review audit \
  --dataset /governed/path/zoe.db \
  --dataset-id zoe-evaluation \
  --patient-column Hashed_PatientURN \
  --split-column Split \
  --output-dir outputs/review/zoe-audit

uv run eeg-review audit \
  --dataset /governed/path/maria.db \
  --dataset-id maria-evaluation \
  --patient-column Hashed_PatientURN \
  --split-column Split \
  --output-dir outputs/review/maria-audit

uv run eeg-review overlap \
  --dataset development=/governed/path/development.db \
  --dataset zoe=/governed/path/zoe.db \
  --dataset maria=/governed/path/maria.db \
  --patient-column Hashed_PatientURN \
  --output-dir outputs/review/cross-cohort-overlap
```

If no patient column exists, omit it. The receipt will explicitly say that
patient independence was not assessed; a hashed report ID must never be
presented as proof of a unique patient.

## Phase 2: leakage-safe native baseline receipts

This command preserves the submitted BoW+LR and frozen BERT+LR model families.
It fits BoW vocabulary inside each fold, uses deterministic stratified folds
(patient-grouped when a patient column is supplied), exports out-of-fold
probabilities and fold membership, and explicitly refits the final model on all
valid development records.

```bash
uv run --extra baselines eeg-review baseline-cv \
  --dataset /governed/path/development.db \
  --model bag_of_words \
  --patient-column Hashed_PatientURN \
  --folds 5 \
  --output-dir outputs/review/development-bow

uv run --extra baselines eeg-review baseline-cv \
  --dataset /governed/path/development.db \
  --model bert_base \
  --patient-column Hashed_PatientURN \
  --folds 5 \
  --output-dir outputs/review/development-bert
```

The BERT route retains `bert-base-uncased`, a frozen final-layer CLS embedding,
and 512-token end truncation. The receipt counts reports exceeding 512 tokens.
No model checkpoint is downloaded by core CI. `oof_predictions.csv` contains
pseudonymous report keys but never report text and must remain governed.

## Phase 3: paired evaluation receipts

Processed predictions must contain one unique report ID and one four-level
prediction column per requested category. Column names can be mapped without
editing code.

```bash
uv run eeg-review evaluate \
  --reference /governed/path/zoe_reference.db \
  --predictions /governed/path/mistral_zoe.csv \
  --cluster-column Hashed_PatientURN \
  --fold-column Fold \
  --prediction-column 'Focal Epi=Focal Epi prediction' \
  --prediction-column 'Gen Epi=Gen Epi prediction' \
  --prediction-column 'Focal Non-epi=Focal Non-epi prediction' \
  --prediction-column 'Gen Non-epi=Gen Non-epi prediction' \
  --prediction-column 'Abnormality=Abnormality prediction' \
  --output-dir outputs/review/zoe-mistral
```

Each run emits:

- `evaluation_summary.json`: matched/unmatched counts, point estimates, 95%
  bootstrap intervals, and explicit interpretation limits;
- `metrics.csv`: raw TP/FP/TN/FN plus core and four-level metrics;
- `fold_metrics.csv`, when a fold column is supplied: every fold's metrics plus
  mean and sample standard deviation in the JSON summary;
- `confusion_matrices.json`: binary and four-level confusion matrices; and
- `run_manifest.json`: input checksums, Git revision, versions, parameters,
  seed, and privacy boundary.

The default intervals resample reports. Supply the patient/cluster column to
obtain cluster-bootstrap intervals. Report-level intervals must not be used to
claim patient-independent precision.

## Phase 4: instrumented LLM reruns

The existing `src/LLM_pipeline/pipeline.py` now records the exact GGUF filename,
Hugging Face snapshot, model-file SHA-256, load parameters, prompts and prompt
hashes, grammar hashes, context/truncation policy, Git state, package versions,
per-stage latency, and llama.cpp prompt/completion token counts. Its governed
CSV output contains per-report telemetry. Each result CSV has a same-stem
`.run.json` sidecar containing aggregate telemetry and immutable input/output
receipts; the historical version-config JSON is also retained for compatibility.

```bash
uv sync --extra llm --extra reports
uv run python src/LLM_pipeline/pipeline.py \
  --num-reports 2000 \
  --dataset-id zoe-historical-source \
  --dataset-path /governed/path/zoe_reports_LD_2000.db \
  --model mistral \
  --temperature 0 \
  --top-k 40 \
  --top-p 0.95 \
  --max-tokens 3000 \
  --outdir /governed/path/runs/zoe-historical-source

uv run python src/LLM_pipeline/process_output.py \
  raw_zoe-historical-source_mistral_2000_v1_run1.csv \
  --input-dir /governed/path/runs/zoe-historical-source \
  --outdir /governed/path/runs/zoe-historical-source/processed
```

The historical Zoe source surface is 2,000 ordered rows. Do **not** run
`--num-reports 1395` against the full database: that would take the first 1,395
rows, mix development and evaluation positions, and omit part of the second
historical slice. Run/retain the 2,000-row output, then apply the documented
governed analysis selection. The LD and SG databases have identical report-ID
and report-text order, so one model run per report-author cohort is sufficient;
the human-label copy is selected later during evaluation.

The submitted-candidate Mistral registry entry is pinned to Hugging Face
revision `3a6fbf4a41a1d52e415a4958cde6856d34b2db93`. The run receipt must report
model SHA-256
`b85cdd596ddd76f3194047b9108a73c74d77ba04bef49255a50fc0cfbda83d32`;
stop if it does not.

Before a full rerun, use `--num-reports 1` with a fresh governed dataset ID and
output directory to validate CUDA/Metal, grammar parsing, receipt creation,
post-processing, and output permissions. The worker is compatible with
spawn-based macOS multiprocessing and stops after three consecutive crashes;
inspect `crash_report.txt` before manually resuming. Do not select a new model
or prompt version through the interactive resume menu unless the run is being
recorded as a distinct experimental condition.

## Non-negotiable run rules

1. Work from immutable input snapshots and retain their checksums.
2. Never commit governed databases, report text, identifiers, model caches,
   credentials, or case-level errors.
3. Run development, Zoe evaluation, and Maria evaluation separately, then run
   the overlap audit.
4. Preserve unfavorable, null, invalid, and unmatched counts.
5. Do not label exact four-level agreement as probabilistic calibration.
6. Do not infer consent/waiver or REB coverage from code or dataset contents.
7. Have two people review the aggregate release before it leaves governed
   infrastructure.

See `DATA_CONTRACT.md`, `CLAIM_LEDGER.md`, and
`DATASET_NAMING.md` for the input contract, claim gates, and historical naming
semantics. The exact candidate submitted-model artifact is recorded under
`model-receipts/`. See `REVIEWER_EXECUTION_MATRIX.md` for the
reviewer-to-artifact map. The recovered result-analysis source and a read-only
audit of its assumptions are preserved under
`../historical/result_analysis/2026-07-20/`.
