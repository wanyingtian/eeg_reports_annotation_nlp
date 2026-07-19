# Reviewer-to-execution matrix

This is an engineering map, not a completed rebuttal. `Implemented` means the
branch can generate the named aggregate receipt when an authorized operator
supplies the required inputs. It does not mean the scientific result is known.

| Review concern | Branch command or artifact | Required governed/team input | Status |
|---|---|---|---|
| Cohort sizes and class support | `eeg-review audit`; `label_counts.csv` | Authoritative development/Zoe/Maria snapshots | Implemented |
| Patient independence | `eeg-review audit --patient-column ...` | Patient-level pseudonymous key semantics | Implemented; semantics require custodian confirmation |
| Development/evaluation leakage | `eeg-review overlap` | All three cohort snapshots | Exact ID/text overlap implemented; semantic near-duplicate phase pending |
| Raw four-level and binary counts | `label_counts.csv` | RA and SA label tables | Implemented per supplied label set |
| Category TP/FP/TN/FN | `eeg-review evaluate`; `metrics.csv` | Paired predictions/reference labels | Implemented |
| Confidence intervals | Cluster bootstrap in `evaluation_summary.json` | Patient/cluster key | Implemented |
| Four-level agreement | Certainty-adjusted accuracy, 4x4 matrix, kappa | Paired four-level labels | Implemented |
| Core agreement | Binary metrics, 2x2 matrix, kappa | Paired four-level labels | Implemented |
| Favorable and unfavorable results | Complete matrices and unmatched/invalid counts | All model outputs, without cherry-picking | Implemented output contract |
| Five-fold variability | `eeg-review evaluate --fold-column ...`; `fold_metrics.csv` | Out-of-fold predictions and fold assignment | Implemented; historical trainer must export folds |
| Calibration | Probability calibration metrics/curves | Per-class probabilities, not ordinal labels alone | Next analysis phase; terminology guardrail documented |
| False-negative consequences | FN counts plus governed case-review packet | Clinical reviewer and approved case-review process | Counts implemented; case review is clinical/team work |
| Stronger LLM comparisons | Named model registry and identical-run matrix | Team-approved models, weights, compute budget | Next inference phase |
| Timing and token characteristics | Per-report telemetry and aggregate run receipt | Authorized rerun on target hardware | Implemented in LLM pipeline |
| Exact prompt/model reproducibility | Prompt/grammar/model/dataset/output hashes and environment receipt | Submitted prompt version and GGUF file | Implemented for new runs; historical provenance still required |
| 14% prompt-improvement claim | Prompt-version comparison receipt | Frozen prompt versions and development outputs | Blocked on historical artifacts |
| 25% explanation-error claim | Named effect measure, raw 2x2 counts, CI | Paired alignment/error data | Evaluation framework available; claim-specific adapter pending |
| Technician/reference justification | Annotation protocol and bounded wording | Wanying, technicians, and clinical lead | Documentation/team decision; not computable |
| Four-level rubric | Versioned annotation guide | Historical instructions/examples | Blocked on source material |
| Clinical workflow comparison | Workflow and escalation description | Vasily/clinical collaborators | Team-authored clinical material |
| Ethics, consent/waiver, secondary use | Approved statement and document receipt | PI/data custodian/REB records for H18-02728 | Non-code blocker |
| Proposed-pipeline figure | Reproducible diagram source | Actual authoring source and verified pipeline stages | Manuscript phase |
| Table/text inconsistencies | Generated table-to-claim ledger | Unrounded authoritative result files | Next reporting phase |

## Planned next implementation phases

1. Instrument the existing LLM pipeline with immutable prompt, grammar, model,
   tokenizer, context/truncation, token-count, latency, hardware, and failure
   receipts while preserving historical defaults.
2. Refactor baseline training so every fold exports IDs, predictions,
   probabilities, seed, preprocessing, fitted thresholds, and fold membership.
3. Add patient-clustered paired model comparisons, fold summaries, calibration
   only where probabilities exist, and multiplicity-aware hypothesis tests.
4. Add an aggregate claim ledger and deterministic manuscript table/figure
   generation, including null and unfavorable outcomes.
5. Add a governed clinical error-review export containing pseudonymous case
   handles only, with no report text leaving the approved environment.
