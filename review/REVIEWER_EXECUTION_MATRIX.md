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
| Confidence intervals | Report or cluster bootstrap in `evaluation_summary.json` | Patient/cluster key for clustered inference | Report-bootstrap intervals generated for historical Mistral and historical/fresh available baselines; clustered inference awaits the key |
| Paired model differences | `eeg-review compare`; paired effect CIs, discordant counts, exact McNemar, Holm adjustment | Two same-case prediction surfaces; patient key for primary clustered inference | Implemented; report-level McNemar is a sensitivity analysis |
| Four-level agreement | Certainty-adjusted accuracy, 4x4 matrix, kappa | Paired four-level labels | Implemented |
| Core agreement | Binary metrics, 2x2 matrix, kappa | Paired four-level labels | Implemented |
| Favorable and unfavorable results | Complete matrices, paired effects, discordance, and unmatched/invalid counts | All model outputs, without cherry-picking | Historical Mistral-vs-SA and historical/fresh baseline receipts generated; rare-label majority-negative failures are explicitly retained in the active checkpoint |
| Five-fold variability | `eeg-review baseline-cv`; `eeg-review evaluate --fold-column ...` | Authorized development data and stable patient key | Leakage-safe OOF folds and explicit full-data refit implemented; generalized epileptiform has only 3 positive development records, so five-fold OOF is explicitly unavailable and the final model is marked `external_fit_only` |
| Calibration | `eeg-review calibrate`; Brier, log loss, fixed-bin ECE and bin supports with cluster intervals | Per-class probabilities and patient key; not ordinal labels alone | Historical Maria and fresh Zoe/Maria BoW receipts generated; fresh BERT queued; clustered intervals await the patient key |
| False-negative consequences | `eeg-review error-review`; governed FN/FP worksheet with pseudonymous case handles | Clinical reviewer, approved protocol, and patient key for clustered selection | Packet generator implemented and provisionally exercised; clinical adjudication remains team work |
| Stronger LLM comparisons | Named model registry and identical-run matrix | Team-approved models, weights, compute budget | Next inference phase |
| Timing and token characteristics | Per-report telemetry and aggregate run receipt | Authorized rerun on target hardware | Implemented and active in the 2,000-report Zoe native run |
| Exact prompt/model reproducibility | Prompt/grammar/model/dataset/output hashes and environment receipt | Submitted prompt version and GGUF file | Exact submitted GGUF is pinned and the native run is active; preserved historical outputs remain the submitted source of record |
| 14% prompt-improvement claim | Prompt-version comparison receipt | Frozen prompt versions and development outputs | Blocked on historical artifacts |
| 25% explanation-error claim | Named effect measure, raw 2x2 counts, CI | Paired alignment/error data | Evaluation framework available; claim-specific adapter pending |
| Technician/reference justification | Annotation protocol and bounded wording | Wanying, technicians, and clinical lead | Documentation/team decision; not computable |
| Four-level rubric | Versioned annotation guide | Historical instructions/examples | Blocked on source material |
| Clinical workflow comparison | Workflow and escalation description | Vasily/clinical collaborators | Team-authored clinical material |
| Ethics, consent/waiver, secondary use | Approved statement and document receipt | PI/data custodian/REB records for H18-02728 | Non-code blocker |
| Proposed-pipeline figure | Reproducible diagram source | Actual authoring source and verified pipeline stages | Manuscript phase |
| Table/text inconsistencies | Generated table-to-claim ledger | Exact Zoe baseline report rows still needed | Submitted aggregate matrices recover all baseline displays; current Zoe CSVs are a different prediction version, while Maria and all other result families match |

## Planned next implementation phases

1. Complete the active Zoe and Maria native inference run and compare its
   four-level/core classifications with the preserved historical surfaces.
2. Rerun paired comparisons and probability calibration with the governed
   patient key; add fold summaries for the exact producing baseline runs.
3. Add deterministic manuscript table/figure
   generation, including null and unfavorable outcomes.
4. Obtain clinical-team approval for the implemented governed error-review
   protocol, then review the sampled cases inside the approved environment.
