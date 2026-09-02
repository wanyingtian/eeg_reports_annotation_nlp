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
uv sync --extra ci              # lightweight classical dependencies used by tests
uv sync --extra reports         # tables, figures, parquet/excel support
uv sync --extra baselines       # BERT/BoW training and SHAP
uv sync --extra llm             # llama.cpp inference pipeline
uv sync --extra evidence        # explanation support/alignment
make verify                     # lint, tests, and sample aggregate audit
make verify-llm-receipt         # fake-model test; downloads no weights
```

The same locked aggregate verification runs in GitHub Actions through
`make ci`. It is deliberately CPU-only: no model weights, `llama.cpp`, Torch,
CUDA/NVIDIA packages, hosted inference, or governed data are installed or
used. The real grammar compiler and model/runtime smoke checks remain explicit
local or governed-hardware commands. CI runs only when toolchain, source, test,
sample-audit, or model-receipt inputs change, and superseded runs on the same
branch are cancelled.

The portable, resumable full-study supervisor is documented in
[`STUDY_COMPUTE_WORKFLOW.md`](STUDY_COMPUTE_WORKFLOW.md). The completed native
macOS execution is recorded in
[`NATIVE_REPRODUCTION_FINAL_CHECKPOINT_2026-08-15.md`](NATIVE_REPRODUCTION_FINAL_CHECKPOINT_2026-08-15.md).
It documents completion, hardware/runtime, fresh-to-historical agreement,
baseline OOF variability, explanation quality, publication implications, and
the remaining patient/clinical/ethics gates without report text or case
identifiers. The earlier active checkpoint is retained as operational history.

The thesis-era learned-polarity explanation surface is separately reconciled in
[`EXPLANATION_RECONCILIATION_2026-09-01.md`](EXPLANATION_RECONCILIATION_2026-09-01.md).
It recovers the 2,180 abnormal-supporting denominator and raw
alignment/correctness counts, but it does not force the current public matcher
to reproduce the submitted 97.8% traceability numerator. The follow-up
[`REASON_TRACEABILITY_EXPERIMENT_2026-09-02.md`](REASON_TRACEABILITY_EXPERIMENT_2026-09-02.md)
records the recovered script/thesis discrepancy, a functionally equivalent
pinned MiniLM surface, and one exact-source/candidate audit for saved Mistral
and MedGemma evidence. Keyed rows and all report/reason text remain governed.

Any post-submission model proposed as a stronger comparator must use
[`CONTEMPORARY_LLM_COMPARISON_PROTOCOL.md`](CONTEMPORARY_LLM_COMPARISON_PROTOCOL.md)
and the typed gates in
[`PRODUCING_BUNDLE_INTAKE.md`](PRODUCING_BUNDLE_INTAKE.md). The public branch
contains only the schema, validator, and unreceipted template under
`model-receipts/`; completed contracts and keyed artifacts stay governed. A
rendered aggregate report is not a substitute for model, prompt, grammar,
cohort, prediction, selection-history, and ancestry receipts. Version 3 keeps
the Google/Unsloth artifact, external producing configuration, historical
evaluation framework, and integration branch as distinct DAG parents. It uses
bounded contribution and governance roles rather than an ownership field.

[`VERSIONED_CONFIGURATION_STREAMS.md`](VERSIONED_CONFIGURATION_STREAMS.md)
makes that naming rule operational: v5g is a technical configuration stream,
while individual contributions remain explicit in provenance metadata. The
shared branch versions public-safe operating parameters and receipts; governed
keys and predictions stay outside Git.

[`EVALUATION_SURFACE_FRAMEWORK.md`](EVALUATION_SURFACE_FRAMEWORK.md) adds the
comparison layer above producing-bundle intake. Its typed, result-free registry
makes model, quantization, interface, prompt, grammar, cohort, reference,
selection role, and metric explicit. It can represent the supplied v1--v10,
v5g, Q2/Q4, Zoe/Maria, and unlabeled-corpus processing categories without
copying provisional values or admitting an external configuration. It also
separates controlled interface ablations from model-native task comparisons
and complete configured-system comparisons, so fairness adjustments are
applied symmetrically to Mistral and MedGemma.

The independently frozen MedGemma native-interface sensitivity has a complete
authorization-first execution and authoring path in
[`MEDGEMMA_NATIVE_PROTECTED_EXECUTION_RUNBOOK.md`](MEDGEMMA_NATIVE_PROTECTED_EXECUTION_RUNBOOK.md).
Its aggregate finalizer retains every prespecified effect, and its deterministic authoring
bundle remains non-admitted unless a separate candidate-hash-bound decision record covers
all primary table rows and the named supplement/reviewer-response destinations.

The MedGemma interface finding also motivates one narrower, model-symmetric
development check. [`MODEL_NATIVE_INTERFACE_SENSITIVITY_PROTOCOL.md`](MODEL_NATIVE_INTERFACE_SENSITIVITY_PROTOCOL.md)
freezes a single Mistral native-interface candidate and an independently
controlled explanation-interface comparison. In both cases the historical
prompts, report order, sampling, and GBNF grammars remain fixed. A chat template
is treated as the model-facing input envelope, not as a replacement for the
structured-output contract. The explanation layer is described as
self-prompted evidence extraction and explicitly does not claim causal
faithfulness. The capped Mistral follow-up completed on August 30; its keyed
outputs and evidence receipts remain in governed storage.

One subsequent MedGemma prompt refinement is specified in
[`MEDGEMMA_PROMPT_V2_DEVELOPMENT.md`](MEDGEMMA_PROMPT_V2_DEVELOPMENT.md).
It preserves the native chat interface and historical grammar, changes only
the focal-epileptiform clarification, and caps execution at the original 100
Zoe development cases plus 20 fixed-classification evidence extractions.
This is an evaluation-informed exploratory version, not an independently
confirmed improvement. Its runner never automatically expands to protected
cohorts and retains unfavorable as well as favorable outcomes.

The completed v2 run did not satisfy its frozen development rule. Its next
step is evidence-informed diagnosis, not automatic promotion or prompt search;
see [`EVIDENCE_INFORMED_PROMPT_DIAGNOSTICS.md`](EVIDENCE_INFORMED_PROMPT_DIAGNOSTICS.md).
The additive source-span layer preserves literal scores, distinguishes
whitespace-only source recovery, and connects saved errors and cross-model
disagreement to focused review questions without claiming causal explanation.

A potential Mistral extension is governed separately by
[`MISTRAL_TASK_ADAPTATION_PROTOCOL.md`](MISTRAL_TASK_ADAPTATION_PROTOCOL.md).
It preregisters a proposed `post_submission_mistral_adapted` layer grounded in
the thesis's model-agnostic prompt, grammar, evidence, consistency, and
post-hoc certainty-mapping directions. The implemented opt-in route separates
binary core classification from the four-level mapping, fixes the 0.5 core
boundary, and selects only among the thesis-derived symmetric margins 0.1,
0.2, and 0.3. Its fitter requires the exact governed first-100 Zoe manifest and
the producing binary prediction receipt, retains every candidate score, and
reports leave-one-out and bootstrap selection-stability diagnostics without
emitting keys. The plan explicitly separates development, design-prior,
evaluation-only, and MedGemma context-only signals. It is not an admitted
evidence layer or a result: the machine gate blocks final evaluation until the
complete adapter is frozen, checksummed, and admitted by the author group.

On FIR/Alliance, load a supported compiler/CUDA module before installing the
`llm` group. Record the exact modules and driver in the run receipt; do not
silently substitute a different model file or quantization.

The private handover share is inventoried and downloaded through the existing
`proton-drive` CLI, never into tracked paths. See
[`PROTON_CLI_WORKFLOW.md`](PROTON_CLI_WORKFLOW.md) for the controlled intake
and return commands and integrity boundary. The completed laptop run was
round-trip verified after return to the private Steven-Chris share; see
[`PROTON_RETURN_RECEIPT_2026-08-15.md`](PROTON_RETURN_RECEIPT_2026-08-15.md).

## Phase 1: read-only cohort receipts

The audit opens SQLite in read-only mode and emits aggregate JSON/CSV only.

```bash
uv run eeg-review audit \
  --dataset /governed/path/zoe.db \
  --dataset-id zoe-evaluation \
  --row-range 100:500 \
  --row-range 1000:2000 \
  --require-complete-labels \
  --patient-column Hashed_PatientURN \
  --output-dir outputs/review/zoe-audit

uv run eeg-review audit \
  --dataset /governed/path/maria.db \
  --dataset-id maria-evaluation \
  --row-range 0:500 \
  --require-complete-labels \
  --patient-column Hashed_PatientURN \
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

Repeated `--row-range` values are half-open positional ranges in immutable
source-table order. `--require-complete-labels` then retains only candidates
with valid levels 1–4 in all five requested labels and records the number
excluded. Ranges must be in bounds and non-overlapping. For the first 100 Zoe
development reports, use `--row-range 0:100`.

Apply that complete-case selection to the Reference Annotator database. Do not
independently complete-case filter the Second Annotator and call the result the
same study population. To obtain SA support on the exact RA-selected study set,
run paired evaluation with RA as `--reference` and SA as `--predictions`; the
four-level matrix row marginals are RA counts and column marginals are SA counts.

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

Evaluate each completed label's own OOF assignments with:

```bash
uv run --extra baselines eeg-review baseline-oof-evaluate \
  --dataset /governed/path/development.db \
  --baseline-dir outputs/review/development-bow \
  --model bag_of_words \
  --patient-column Hashed_PatientURN \
  --output-dir outputs/review/development-bow-oof
```

The evaluator preserves unavailable labels as unavailable. It never converts a
missing OOF probability into a thresholded class.

## Phase 3: paired evaluation receipts

Processed predictions must contain one unique report ID and one four-level
prediction column per requested category. Column names can be mapped without
editing code.

```bash
uv run eeg-review evaluate \
  --reference /governed/path/zoe_reference.db \
  --predictions /governed/path/mistral_zoe.csv \
  --reference-range 100:500 \
  --reference-range 1000:2000 \
  --require-complete-reference \
  --cluster-column Hashed_PatientURN \
  --fold-column Fold \
  --prediction-column 'Focal Epi=Focal Epi prediction' \
  --prediction-column 'Gen Epi=Gen Epi prediction' \
  --prediction-column 'Focal Non-epi=Focal Non-epi prediction' \
  --prediction-column 'Gen Non-epi=Gen Non-epi prediction' \
  --prediction-column 'Abnormality=Abnormality prediction' \
  --output-dir outputs/review/zoe-mistral
```

The repeated `--reference-range` arguments implement Python-style half-open
positional selection against the immutable source table. For the historical
Zoe evaluation, `100:500` plus `1000:2000` yields 1,400 candidates and
`--require-complete-reference` removes the five rows lacking a complete
five-label RA reference, leaving 1,395. Maria uses `--reference-range 0:500`
and the same complete-case rule, leaving 499. Predictions are selected and
ordered by report ID after this reference selection, so Mistral's 2,000-row
surface and the baselines' 1,900-row post-development surface use the same
reference command.

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

## Phase 3b: paired same-case comparisons

Do not infer model differences by comparing two independently rounded table
cells. Align both prediction surfaces to the same reference cases and emit the
discordant-pair and paired-bootstrap receipt:

```bash
uv run eeg-review compare \
  --reference /governed/path/zoe_reference.db \
  --predictions-a /governed/path/mistral_zoe.xlsx \
  --predictions-b /governed/path/second_annotator_zoe.db \
  --model-a-id mistral-7b-submitted \
  --model-b-id second-annotator \
  --reference-range 100:500 \
  --reference-range 1000:2000 \
  --require-complete-reference \
  --cluster-column Hashed_PatientURN \
  --output-dir outputs/review/zoe-mistral-vs-sa
```

The command emits model-A-minus-model-B differences for core accuracy,
certainty-adjusted accuracy, and false-negative rate with paired 95% bootstrap
intervals. It also emits the two discordant-correctness cells and a two-sided
exact McNemar test for core and exact four-level correctness. Holm correction is
applied by default across all requested categories and both correctness tests;
the declared family is recorded in the receipt.

McNemar operates on report pairs and does not account for repeated reports from
one patient. When a stable patient key exists, the patient-cluster bootstrap is
the primary inference and McNemar is a report-level sensitivity analysis. When
no patient key is supplied, both limitations are printed in the result. Exact
Zoe baseline comparisons must wait for the producing report-level BoW/BERT
exports; the recovered aggregate matrices are insufficient for paired tests.

For the final three-layer model analysis, first materialize governed prediction
views containing exactly the confirmed intake-manifest keys. Then add
`--require-exact-key-set` and `--require-patient-grouping`. The strict key gate
is not appropriate for a larger all-reports prediction export that
intentionally contains records outside the selected reference surface.

## Phase 3c: probability calibration for baselines

Calibration is evaluated only for models that emit an actual positive-class
probability. The recovered BoW/BERT files use `Prob_<category>` for the
estimated probability of core-positive levels 3–4:

```bash
uv run eeg-review calibrate \
  --reference /governed/path/maria_reference.db \
  --predictions /governed/path/maria_inference_results_bert_base.csv \
  --model-id bert-lr-submitted \
  --reference-range 0:500 \
  --require-complete-reference \
  --cluster-column Hashed_PatientURN \
  --bins 10 \
  --output-dir outputs/review/maria-bert-calibration
```

The aggregate receipt contains Brier score, log loss, fixed-width expected
calibration error, bin counts, mean predicted probability, observed event rate,
and cluster-bootstrap intervals. The bin count and boundaries are part of the
run manifest because ECE is binning-dependent. Empty bins remain explicit.

Do not apply this command or the word “calibration” to Mistral's generated
four-level category as though it were a probability. LLM confidence-level
agreement is an ordinal/exact-agreement analysis unless the authors separately
define and validate a probabilistic mapping. Only Maria's recovered baseline
rows are currently confirmed as the exact submitted prediction artifacts; Zoe
baseline calibration remains artifact-version-specific until the producing
rows are recovered.

## Phase 3d: governed clinical error-review packet

False-negative and false-positive consequences require authorized clinical
judgment. The tool can prepare a reproducible case-review worksheet, but it
does not read or export report text and it does not adjudicate the Reference
Annotator as clinical truth:

```bash
uv run --extra reports eeg-review error-review \
  --reference /governed/path/zoe_reference.db \
  --predictions /governed/path/mistral_zoe.xlsx \
  --model-id mistral-7b-submitted \
  --reference-range 100:500 \
  --reference-range 1000:2000 \
  --require-complete-reference \
  --cluster-column Hashed_PatientURN \
  --max-per-stratum 25 \
  --handle-salt STUDY_CONTROLLED_SALT \
  --acknowledge-governed-output \
  --output-dir /governed/path/error-review/zoe-mistral
```

The explicit acknowledgement is mandatory because the worksheet is a
case-level governed artifact even though source report and patient identifiers
are replaced by deterministic case handles. Sampling is stratified by label
and RA-relative core false-negative/false-positive direction. When a patient
key is available, distinct patient clusters are preferred before additional
reports are selected. Keep the packet inside the approved environment and use
[`CLINICAL_ERROR_REVIEW_PROTOCOL.md`](CLINICAL_ERROR_REVIEW_PROTOCOL.md) before
clinical review.

## Independent MedGemma comparator

An additive matched-interface MedGemma comparison is specified in
[`MEDGEMMA_INDEPENDENT_COMPARATOR_STUDY.md`](MEDGEMMA_INDEPENDENT_COMPARATOR_STUDY.md).
Its machine-readable plan and readiness gate are separate from exact intake of
Vasily's v5g configuration. The latter is an additive, nonblocking external
configuration until its producing bundle arrives.

Prepare exact complete-case manifests, governed SQLite snapshots, selected
submitted/reproduced Mistral comparison surfaces, and a resumable command plan
without starting inference:

```bash
make medgemma-prepare \
  SOURCE_RUN=/governed/path/jbhi-native-20260814 \
  RUN_DIR=/governed/path/jbhi-medgemma-independent-v1
```

The preparation step writes governed case-level material and therefore must
not target the public repository. It executes no model call.

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

### Development-only Mistral adaptation gate

The post-submission binary-core Mistral work package is intentionally separate
from the historical rerun. Before any 100-report development inference, use
`eeg-review development-manifest-create` to create the immutable governed Zoe
RA key sequence, then `eeg-review adaptation-development-prepare` to bind the
reference and manifest identities to an unfrozen execution plan. Both commands
fail on missing or duplicate keys, wrong population arithmetic, incomplete
labels, receipt mismatch, or attempted overwrite. See
[`MISTRAL_TASK_ADAPTATION_PROTOCOL.md`](MISTRAL_TASK_ADAPTATION_PROTOCOL.md) and
[`MISTRAL_TASK_ADAPTATION_IMPLEMENTATION_CHECKPOINT_2026-08-27.md`](MISTRAL_TASK_ADAPTATION_IMPLEMENTATION_CHECKPOINT_2026-08-27.md).

This preparation does not authorize a development run and cannot make the
adapter ready for protected Zoe or Maria evaluation. The keyed manifest,
bound execution plan, predictions, and fit receipts remain governed.

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
