# Reviewer-to-execution matrix

This is an engineering map, not a completed rebuttal. `Implemented` means the
branch can generate the named aggregate receipt when an authorized operator
supplies the required inputs. It does not mean the scientific result is known.

| Review concern | Branch command or artifact | Required governed/team input | Status |
|---|---|---|---|
| Cohort sizes and class support | `eeg-review audit`; `label_counts.csv` | Authoritative development/Zoe/Maria snapshots | Implemented |
| Patient independence | `eeg-review audit --patient-column ...` | Patient-level pseudonymous key semantics | Implemented; semantics require custodian confirmation |
| Development/evaluation leakage | `eeg-review overlap --patient-column ...` | All three cohort snapshots and stable patient key | Exact patient/ID/text overlap implemented; semantic near-duplicate phase pending |
| Raw four-level and binary counts | `label_counts.csv` | RA and SA label tables | Implemented per supplied label set |
| Category TP/FP/TN/FN | `eeg-review evaluate`; `metrics.csv` | Paired predictions/reference labels | Implemented |
| Confidence intervals | Cluster bootstrap in `evaluation_summary.json` | Patient/cluster key | Implemented |
| Paired model differences | `eeg-review compare`; paired effect CIs, discordant counts, exact McNemar, Holm adjustment | Two same-case prediction surfaces; patient key for primary clustered inference | Implemented; report-level McNemar is a sensitivity analysis |
| Four-level agreement | Certainty-adjusted accuracy, 4x4 matrix, kappa | Paired four-level labels | Implemented |
| Core agreement | Binary metrics, 2x2 matrix, kappa | Paired four-level labels | Implemented |
| Favorable and unfavorable results | Complete matrices, paired effects, discordance, and unmatched/invalid counts | All model outputs, without cherry-picking | Implemented output contract; initial Mistral-vs-SA and Maria-baseline receipts generated privately |
| Five-fold variability | `eeg-review baseline-cv`; `eeg-review evaluate --fold-column ...` | Authorized development data and stable patient key | Leakage-safe OOF folds and explicit full-data refit implemented for submitted BoW/BERT families |
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
| Table/text inconsistencies | Generated table-to-claim ledger | Exact Zoe baseline report rows still needed | Submitted aggregate matrices recover all baseline displays; current Zoe CSVs are a different prediction version, while Maria and all other result families match |

## Planned next implementation phases

1. Instrument the existing LLM pipeline with immutable prompt, grammar, model,
   tokenizer, context/truncation, token-count, latency, hardware, and failure
   receipts while preserving historical defaults.
2. Rerun the implemented paired model comparison with the governed patient key;
   add fold summaries and calibration only where probabilities exist.
3. Add deterministic manuscript table/figure
   generation, including null and unfavorable outcomes.
4. Add a governed clinical error-review export containing pseudonymous case
   handles only, with no report text leaving the approved environment.
