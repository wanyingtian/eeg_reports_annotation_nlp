# Reviewer-to-execution matrix

This is an engineering map, not a completed rebuttal. `Implemented` means the
branch can generate the named aggregate receipt when an authorized operator
supplies the required inputs. It does not mean the scientific result is known.

| Review concern | Branch command or artifact | Required governed/team input | Status |
|---|---|---|---|
| Cohort sizes and class support | `eeg-review audit`; `label_counts.csv` | Authoritative development/Zoe/Maria snapshots | Implemented |
| Patient independence | `eeg-review audit --patient-column ...` | Patient-level pseudonymous key semantics | Implemented; semantics require custodian confirmation |
| Development/evaluation leakage | `eeg-review overlap --patient-column ...` | All three cohort snapshots and stable patient key | Completed exact report-ID and normalized-text audit found no overlap; patient overlap and semantic near-duplicates remain pending |
| Raw four-level and binary counts | `label_counts.csv` | RA and SA label tables | Implemented per supplied label set |
| Category TP/FP/TN/FN | `eeg-review evaluate`; `metrics.csv` | Paired predictions/reference labels | Implemented |
| Confidence intervals | Report or cluster bootstrap in `evaluation_summary.json` | Patient/cluster key for clustered inference | Report-bootstrap intervals completed for historical/fresh Mistral and historical/fresh baselines; clustered inference awaits the key |
| Paired model differences | `eeg-review compare`; paired effect CIs, discordant counts, exact McNemar, Holm adjustment | Two same-case prediction surfaces; patient key for primary clustered inference | Implemented; report-level McNemar is a sensitivity analysis |
| Four-level agreement | Certainty-adjusted accuracy, 4x4 matrix, kappa | Paired four-level labels | Implemented |
| Core agreement | Binary metrics, 2x2 matrix, kappa | Paired four-level labels | Implemented |
| Favorable and unfavorable results | Complete matrices, paired effects, discordance, and unmatched/invalid counts | All model outputs, without cherry-picking | Complete fresh/historical receipts generated; rare-label majority-negative failures, lower fresh results, logical violations, and explanation fallbacks are retained in the final checkpoint |
| Five-fold variability | `eeg-review baseline-cv`; `eeg-review baseline-oof-evaluate` | Authorized development data and stable patient key | Completed report-level OOF fold metrics, mean and sample SD; generalized epileptiform has only 3 positive development records, so OOF is explicitly unavailable and the final model is `external_fit_only`; patient grouping awaits the key |
| Calibration | `eeg-review calibrate`; Brier, log loss, fixed-bin ECE and bin supports with cluster intervals | Per-class probabilities and patient key; not ordinal labels alone | Historical and fresh Zoe/Maria BoW/BERT receipts completed; clustered intervals await the patient key |
| False-negative consequences | `eeg-review error-review`; governed FN/FP worksheet with pseudonymous case handles | Clinical reviewer, approved protocol, and patient key for clustered selection | Fresh packets completed (179 Zoe and 117 Maria selected label-case rows); clinical adjudication remains team work |
| Stronger LLM comparisons | Named model registry and identical-run matrix | Team-approved models, weights, compute budget | Next inference phase |
| Timing and token characteristics | Per-report telemetry and aggregate run receipt | Authorized rerun on target hardware | Completed for 2,000 Zoe and 500 Maria reports on macOS/Apple M5; portable Linux/NVIDIA verification is optional |
| Exact prompt/model reproducibility | Prompt/grammar/model/dataset/output hashes and environment receipt | Submitted prompt version and GGUF file | Exact GGUF, snapshot, prompt, grammar, dataset, output, environment and transfer receipts completed; preserved historical outputs remain the submitted source of record pending author confirmation |
| 14% prompt-improvement claim | Prompt-version comparison receipt | Frozen prompt versions and development outputs | Blocked on historical artifacts |
| 25% explanation-error claim | Named effect measure, raw 2x2 counts, CI | Paired alignment/error data | Evaluation framework available; claim-specific adapter pending |
| Technician/reference justification | Annotation protocol and bounded wording | Wanying, technicians, and clinical lead | Documentation/team decision; not computable |
| Four-level rubric | Versioned annotation guide | Historical instructions/examples | Blocked on source material |
| Clinical workflow comparison | Workflow and escalation description | Vasily/clinical collaborators | Team-authored clinical material |
| Ethics, consent/waiver, secondary use | Approved statement and document receipt | PI/data custodian/REB records for H18-02728 | Non-code blocker |
| Proposed-pipeline figure | Reproducible diagram source | Actual authoring source and verified pipeline stages | Manuscript phase |
| Table/text inconsistencies | `study_job.py ledger`; source-hashed long-form result tables for the table-to-claim ledger | Exact Zoe baseline report rows still needed | Aggregate ledger implemented and exercised; submitted aggregate matrices recover all baseline displays, but current Zoe CSVs remain a different prediction version |

## Planned next implementation phases

1. Obtain Chris's author confirmation of the source-of-record boundary and the
   interpretation of fresh-to-historical sensitivity results.
2. Rerun clustered comparisons and grouped folds if a governed patient key is
   supplied; add exact historical Zoe baseline rows if recovered.
3. Add deterministic manuscript table/figure
   generation, including null and unfavorable outcomes.
4. Obtain clinical-team approval for the implemented governed error-review
   protocol, then review the sampled cases inside the approved environment.
