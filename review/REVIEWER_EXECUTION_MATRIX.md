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
| Calibration | `eeg-review calibrate`; Brier, log loss, fixed-bin ECE and bin supports with cluster intervals; `eeg-review certainty-adapter-fit` for the separate Mistral four-level mapping question | Per-class probabilities and patient key for probabilistic calibration; exact governed first-100 manifest plus binary-core run receipt for Mistral certainty mapping | Historical and fresh Zoe/Maria BoW/BERT receipts completed; clustered intervals await the patient key. Historical Mistral outputs remain ordinal-only. A thesis-derived binary-core probability surface, fixed 0.5 boundary, margin grid, support fallback, leave-one-out diagnostic, and bootstrap selection-stability fitter are implemented but have not been run on development or evaluation data |
| False-negative consequences | `eeg-review error-review`; governed FN/FP worksheet with pseudonymous case handles | Clinical reviewer, approved protocol, and patient key for clustered selection | Fresh packets completed (179 Zoe and 117 Maria selected label-case rows); clinical adjudication remains team work |
| Stronger LLM comparisons | [`CONTEMPORARY_LLM_COMPARISON_PROTOCOL.md`](CONTEMPORARY_LLM_COMPARISON_PROTOCOL.md); matched-historical Q2 receipt; frozen native-interface plan; fail-closed protected runner; source-hashed author-review finalizer | Documentary confirmation covering the protected post-submission execution; patient key only for patient-grouped inference; exact external v5g bundle for reproducing that distinct configuration | Independent Q2 evaluation is complete with unfavorable findings preserved. Native-interface development completed once on the fixed first 100 Zoe reports and was frozen before reference-metric access; the exact 1,395/499 evaluation, unattended recovery, and 90-row aggregate claim package are implemented but unrun pending the documentary gate. External v5g remains additive and separately gated on exact intake. |
| Timing and token characteristics | Per-report telemetry and aggregate run receipt | Authorized rerun on target hardware | Completed for 2,000 Zoe and 500 Maria reports on macOS/Apple M5; portable Linux/NVIDIA verification is optional |
| Exact prompt/model reproducibility | Prompt/grammar/model/dataset/output hashes and environment receipt | Submitted prompt version and GGUF file | Exact GGUF, snapshot, prompt, grammar, dataset, output, environment and transfer receipts completed; Chris confirmed the producing framework and preserved historical outputs as the submitted source of record on 2026-08-25 |
| 14% prompt-improvement claim | Prompt-version comparison receipt | Frozen prompt versions, development outputs, and exact separation from evaluation | Chris recalled first-100 prompt development and a later 100-1000 performance evaluation but could not identify the exact percentage source; claim remains blocked pending the producing variants, outputs, and selection semantics |
| 25% explanation-error claim | `reconcile_explanation_artifact.py`; named accuracy difference, raw aligned/misaligned correctness counts, conservative interval, declared unit and population sensitivity | Recovered learned-polarity artifact, historical Zoe reference and current 1,395-report manifest | Aggregate reconciliation complete. Historical overall abnormality is 1,611/1,639 vs 92/128 correct (26.42-point difference, conservative 95% CI 18.60 to 35.28); current evaluation sensitivity is 1,156/1,179 vs 70/95 (24.36 points, 15.60 to 34.66). Replace “25% more likely”; printed 72.5% remains non-exact. |
| 97.8% explanation traceability | Positive learned-polarity denominator, exact matcher stages, embedding revision and raw matched/unmatched counts | Exact producing matcher or author confirmation | Denominator 2,180 and source artifact are reconciled. Public whole-report script replay is 2,018/2,180; sentence-level diagnostic is 2,153/2,180. Submitted 2,132 numerator remains gated and no threshold was tuned to recover it. |
| Technician/reference justification | Annotation protocol and bounded wording | Wanying, technicians, and clinical lead | Documentation/team decision; not computable |
| Four-level rubric | Versioned annotation guide | Historical instructions/examples | Blocked on source material |
| Clinical workflow comparison | Workflow and escalation description | Vasily/clinical collaborators | Team-authored clinical material |
| Ethics, consent/waiver, secondary use | Approved statement and document receipt | PI/data custodian/REB records for H18-02728 | Non-code blocker |
| Deterministic contemporary-comparator display | `render_medgemma_native_author_bundle.py`; 20-row LaTeX table, methods/results fragments, reviewer-response fragment, 90-row claim ledger, and hash-bound bundle receipt | Completed protected aggregate candidate; separate author admission receipt for promotion beyond author-working status | Implemented and synthetically verified; no protected result or manuscript admission is inferred |
| Proposed-pipeline figure | Reproducible diagram source | Actual authoring source and verified pipeline stages | Manuscript phase |
| Table/text inconsistencies | `study_job.py ledger`; source-hashed long-form result tables for the table-to-claim ledger | Exact Zoe baseline report rows still needed | Aggregate ledger implemented and exercised; submitted aggregate matrices recover all baseline displays, but current Zoe CSVs remain a different prediction version |

## Planned next implementation phases

1. Freeze Chris's 2026-08-25 producing-framework and source-of-record
   confirmation in the private author record; do not rewrite the historical
   outputs.
2. Reconcile and receipt any contemporary-model bundle before using its
   rendered aggregate summaries, including prompt-selection leakage and any
   mismatch among reported cohort sizes.
3. Rerun clustered comparisons and grouped folds if a governed patient key is
   supplied; add exact historical Zoe baseline rows if recovered.
4. Render the implemented deterministic MedGemma authoring bundle after a protected
   candidate exists; promote it only through the hash-bound admission contract.
5. Obtain clinical-team approval for the implemented governed error-review
   protocol, then review the sampled cases inside the approved environment.
