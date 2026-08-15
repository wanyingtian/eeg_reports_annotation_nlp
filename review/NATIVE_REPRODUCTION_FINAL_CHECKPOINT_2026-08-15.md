# Native reproduction final checkpoint - 2026-08-15

## Decision summary

The complete laptop reproduction is scientifically usable and should be
preserved. The governed supervisor completed all 40 declared stages, including
2,000 Zoe and 500 Maria Mistral reports, both baseline families, evaluation,
calibration, paired comparisons, explanation checks, clinical-error packet
generation, and aggregate-ledger generation. There were no missing output
reports, duplicate output identifiers, invalid four-level values, unreadable
classification JSON objects, or unreadable explanation JSON objects.

Fresh Mistral classifications closely reproduce the preserved historical
surfaces: exact four-level agreement is 92.63% across 10,000 Zoe label cells and
92.00% across 2,500 Maria cells; binary/core agreement is 97.88% and 96.92%,
respectively. Performance changes against the Reference Annotator are small and
mixed rather than uniformly favorable. This is expected sensitivity to a fresh
execution, not evidence of degeneration or a reason to stop.

The submitted historical outputs remain the source of record until Wanying
Tian confirms the producing-state interpretation. The fresh run is a complete
reproduction/sensitivity layer and a sound base for reviewer-requested analyses.
No submitted manuscript number should be silently replaced before that author
checkpoint.

## Execution receipt

Run directory (governed and ignored by Git):
`data/governed/study-runs/jbhi-native-20260814`

| Item | Recorded value |
|---|---|
| Study ID | `jbhi-02463-2026-native-reproduction` |
| Job-start revision | `791498296f344a1cfdc2a1a50da2be0f81338d9f` |
| Final supervisor state | `completed`, 40/40 stages |
| Zoe inference | 2,000/2,000 reports; 51,246.6 s stage time; approximately 25.63 s/report |
| Maria inference | 500/500 reports; 12,617.2 s stage time; approximately 25.22 s/report |
| Combined Mistral inference wall time | Approximately 17 h 45 min |
| Platform | MacBook Pro `Mac17,2`; Apple M5; 10 cores (4 performance, 6 efficiency); 24 GB unified memory |
| Operating system | macOS 26.5.2 build 25F84; arm64 |
| Python | 3.11.15 |
| Key packages | NumPy 2.4.6; pandas 2.3.3; scikit-learn 1.9.0; PyTorch 2.13.0; Transformers 4.57.6; llama-cpp-python 0.3.34; huggingface-hub 0.36.2 |
| Accelerator route | Apple unified-memory/MPS for BERT; Metal-backed llama.cpp offload (`n_gpu_layers=30`) for Mistral |

Device serial number, hardware UUID, provisioning identifier, credentials, case
identifiers, and report text are deliberately excluded from this checkpoint.

### Mistral identity

| Component | Receipt |
|---|---|
| Registry | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` |
| File | `mistral-7b-instruct-v0.2.Q5_K_M.gguf` |
| Snapshot | `3a6fbf4a41a1d52e415a4958cde6856d34b2db93` |
| Size | 5,131,409,696 bytes |
| SHA-256 | `b85cdd596ddd76f3194047b9108a73c74d77ba04bef49255a50fc0cfbda83d32` |
| Load | context 4,096; 30 GPU layers |
| Sampling | temperature 0; top-k 40; top-p 0.95; maximum completion 3,000 tokens |
| Classification prompt SHA-256 | `52198221d8330e9857b51a7ad99b017aa18836e1718b08dd0ae355820f5a5e69` |
| Explanation prompt SHA-256 | `09e7e46d13d9fd1f6ebd14e4caecf32766354ead048ef22aa834c5b6064cd05f` |
| Classification grammar SHA-256 | `5237e13988062538cda9c21906f1f4e1fc8b99498e2462ea69fe24bface35016` |
| Explanation grammar SHA-256 | `718d3b0b16499d04d97723893f5e1de67aa1f342ba8b455a293a3d93084cd315` |

The longest observed prompt/completion combinations remained within the
4,096-token context. No context-overflow failure was detected.

### Revision reconciliation

The job and detached supervisor were initialized at commit `7914982`. While the
long Zoe stage ran, the branch advanced to `89c1de3` and `9e41d59` through
checkpoint documentation and aggregate-ledger additions. The Mistral inference
path, prompts, grammars, and model registry were identical between `7914982` and
`9e41d59`. The old receipt writer recorded the revision when it wrote the final
receipt, so the long-stage receipt says `9e41d59` even though execution began
under `7914982`. This is reconciled by Git diff and the supervisor start record;
it is not an inference-code ambiguity.

The branch now captures the execution-start revision separately from the
receipt-write revision. The transfer manifest also distinguishes the job-start
revision from its own generation revision. These changes improve future
receipts without retroactively rewriting the completed inference history.

## Cohort and leakage audit

| Population | Candidate rule | Candidates | Excluded incomplete | Analysis N |
|---|---|---:|---:|---:|
| Zoe development | first 100 reports | 100 | 0 | 100 |
| Zoe evaluation | positions `[100:500] + [1000:2000]` | 1,400 | 5 | 1,395 |
| Maria evaluation | positions `[0:500]` | 500 | 1 | 499 |

There is no exact report-ID overlap and no case-folded,
whitespace-normalized report-text overlap among development, Zoe evaluation,
and Maria evaluation. A stable patient key was not present in the transferred
snapshot, so patient overlap, patient-grouped folds, patient-cluster intervals,
and patient-level sampling remain unassessed. Semantic or templated
near-duplicates also remain unassessed. Neither report hashes nor exact-text
checks should be presented as proof of patient independence.

## Fresh Mistral quality audit

| Check | Zoe | Maria |
|---|---:|---:|
| Reports requested / completed | 2,000 / 2,000 | 500 / 500 |
| Duplicate or missing output IDs | 0 | 0 |
| Classification JSON/schema failures | 0 | 0 |
| Explanation JSON/schema/empty failures | 0 | 0 |
| Invalid classification levels | 0 | 0 |
| Explanation decision mismatches | 1 / 10,000 cells | 2 / 2,500 cells |
| Activity/abnormality logical-constraint violations | 110 / 2,000 (5.5%) | 25 / 500 (5.0%) |
| Historical logical-constraint violations | 94 / 2,000 (4.7%) | 34 / 500 (6.8%) |
| Fallback reasons | 4,723 / 11,522 | 1,159 / 2,773 |
| Unique whole explanation objects | 1,310 | 344 |
| Largest identical explanation group | 555 / 2,000 | 103 / 500 |

The logical violations are inherited model behavior rather than parser failure.
The explanation output is structurally robust but often formulaic. Compared
with the historical outputs, fresh non-fallback reasons are more often exact
contiguous excerpts from the report. Exact anchoring by label ranges from 0.761
to 0.917 on Zoe and 0.723 to 0.898 on Maria, versus 0.606 to 0.770 and 0.659 to
0.812 historically. This is encouraging, but anchoring is not proof of clinical
correctness.

## Fresh-to-historical Mistral results

The table compares current and historical predictions against the same
complete-case Reference Annotator population. CAA means certainty-adjusted
four-level accuracy.

| Cohort | Label | Core historical | Core fresh | Fresh - historical (95% paired report-bootstrap CI) | CAA historical | CAA fresh |
|---|---|---:|---:|---:|---:|---:|
| Zoe | Focal epileptiform | 0.9871 | 0.9871 | 0.0000 (0.0000, 0.0000) | 0.7369 | 0.7441 |
| Zoe | Generalized epileptiform | 0.9677 | 0.9685 | 0.0007 (-0.0036, 0.0050) | 0.9405 | 0.9369 |
| Zoe | Focal non-epileptiform | 0.8631 | 0.8724 | 0.0093 (-0.0014, 0.0201) | 0.6323 | 0.6810 |
| Zoe | Generalized non-epileptiform | 0.8989 | 0.8860 | -0.0129 (-0.0237, -0.0014) | 0.7943 | 0.7892 |
| Zoe | Abnormality | 0.9613 | 0.9505 | -0.0108 (-0.0179, -0.0036) | 0.7720 | 0.7771 |
| Maria | Focal epileptiform | 0.9739 | 0.9760 | 0.0020 (0.0000, 0.0060) | 0.6613 | 0.6493 |
| Maria | Generalized epileptiform | 0.9860 | 0.9880 | 0.0020 (0.0000, 0.0060) | 0.9719 | 0.9579 |
| Maria | Focal non-epileptiform | 0.8597 | 0.8517 | -0.0080 (-0.0261, 0.0120) | 0.6874 | 0.7375 |
| Maria | Generalized non-epileptiform | 0.8838 | 0.9018 | 0.0180 (approximately 0.0000, 0.0381) | 0.8337 | 0.8617 |
| Maria | Abnormality | 0.9158 | 0.9038 | -0.0120 (-0.0341, 0.0080) | 0.7275 | 0.7575 |

The direction is mixed. Some current core results are modestly lower, some are
higher, and certainty-level agreement often changes more than binary activity
presence. This is precisely the pattern that a reviewer-facing sensitivity
analysis should disclose. Exact McNemar and Holm-adjusted values remain in the
machine-readable paired-comparison receipts; they must be interpreted alongside
paired confidence intervals, not used as an isolated pass/fail device.

## Baseline development and external results

The current BoW+LR and BERT+LR pathways are fresh, leakage-safe refits of the
declared model families. They do not recreate missing historical fold
assignments. Text preprocessing is fit inside each fold, OOF probabilities and
assignments are exported, and the final external model is explicitly refit on
all 100 development reports.

### Development OOF

| Model | Label | Core accuracy | F1 | CAA | Fold result |
|---|---|---:|---:|---:|---|
| BoW+LR | Abnormality | 0.93 | 0.914 | 0.70 | core SD 0.027; F1 SD 0.035 |
| BERT+LR | Abnormality | 0.97 | 0.966 | 0.69 | core SD 0.045; F1 SD 0.049 |
| BoW+LR | Focal epileptiform | 0.95 | 0.000 | 0.94 | all fold F1 values 0 |
| BERT+LR | Focal epileptiform | 0.94 | 0.000 | 0.93 | all fold F1 values 0 |
| BoW+LR | Focal non-epileptiform | 0.80 | 0.286 | 0.67 | F1 fold SD 0.313 |
| BERT+LR | Focal non-epileptiform | 0.83 | 0.514 | 0.69 | F1 fold SD 0.175 |
| BoW+LR | Generalized non-epileptiform | 0.94 | 0.875 | 0.86 | core fold SD 0.082 |
| BERT+LR | Generalized non-epileptiform | 0.92 | 0.840 | 0.79 | core fold SD 0.076 |
| Both | Generalized epileptiform | unavailable | unavailable | unavailable | only 3 positives; `external_fit_only` |

The low-support label previously produced missing OOF probability/fold cells
but a mechanical level-4 prediction because missing probability entered the
threshold conversion. That value was not used in external evaluation, and the
training summary already labelled the category `external_fit_only`. The code
now keeps the OOF prediction missing as well, adds an explicit per-label OOF
evaluator, and tests both the low-support and ordinary paths.

### External behavior

Fresh external receipts show that apparently high rare-label accuracy can hide
zero or near-zero positive-class F1. For example, fresh BoW focal-epileptiform
F1 is 0 on both Zoe and Maria despite core accuracies of 0.962 and 0.936. Fresh
BERT focal-epileptiform F1 is 0.036 on Zoe and 0 on Maria. Mistral detects these
rare positives more successfully, while its generated four-level labels are not
probabilities and should not be described as calibrated.

For probability-bearing baselines, the run records Brier score, log loss,
fixed-bin ECE and bin support. Calibration is cohort-dependent. On Maria,
BERT's calibration is generally better than BoW's for abnormality and the
non-epileptiform labels; no pooled or model-wide calibration claim is warranted.

## Clinical-error review preparation

The deterministic packets contain pseudonymous report handles but no report
text. The fresh run selected 179 Zoe and 117 Maria label-case rows under the
configured stratum cap. Across labels, Zoe has 290 false-negative and 178
false-positive decisions; Maria has 154 and 35. These are label-decision counts,
not unique patients and not clinical-harm counts. Clinical salience, ambiguity,
workflow consequence, and escalation/override behavior require an approved
protocol and qualified reviewer.

## Publication and reviewer implications

### Cleared for author review now

- Reproducibility: exact model, prompt, grammar, dataset, environment, runtime,
  output and transfer receipts exist for a complete current run.
- Report-level uncertainty: 2,000-replicate intervals exist for all five labels,
  both cohorts and all completed model surfaces.
- Baseline variability: fold-level OOF metrics, means and sample SDs exist where
  the data support five folds; low-support failure is explicit.
- Calibration: Brier score, log loss, fixed-bin ECE and bin support exist for
  actual BoW/BERT probabilities.
- Paired evidence: same-case differences, paired intervals, discordant counts,
  exact McNemar tests and Holm corrections exist for current/historical and
  Mistral/Second-Annotator comparisons.
- Unfavorable results: rare-label majority-negative failures, lower current
  results, logical violations, formulaic explanation fallbacks and clinical
  error counts are retained.

### Not cleared by this run

- Patient independence, patient-clustered inference, patient-grouped folds and
  patient-level sample sizes.
- Demographic subgroup/generalizability analysis without authorized fields.
- Clinical consequences or diagnostic-ground-truth wording.
- REB consent/waiver, de-identification and secondary-use statements.
- The historical "up to 14%" prompt gain and "25% more likely" explanation
  claim without their producing version/alignment artifacts.
- Unbounded "near-human" or "traditional NLP failed" wording.
- Substituting current results for submitted results without Chris's author
  confirmation.

## Audience-specific next actions

### Wanying Tian

1. Confirm that the pinned model, prompt, grammar, dataset selection and report
   order describe the intended native producing framework.
2. Review the fresh-to-historical table and approve the historical outputs as
   source of record plus the fresh run as a sensitivity/reproduction layer.
3. Identify any exact submission-producing Zoe BoW/BERT rows, prompt-ablation
   artifacts, alignment table, four-level rubric or annotator-protocol notes
   still available. Missing artifacts should be recorded as unavailable rather
   than recreated from memory.
4. Confirm the source tag before reviewer-driven edits are layered onto the
   paper.

### Vasily Vakorin and clinical/data owners

1. Confirm whether a stable pseudonymous patient key exists and its linkage
   semantics across all three populations.
2. Approve any demographic fields and small-cell policy required for subgroup
   analysis.
3. Supply or approve the authoritative REB/secondary-use language.
4. Assign a qualified owner and protocol for the governed FN/FP review.
5. Optionally run the portable directory on an Alliance Linux/NVIDIA resource
   to verify cross-platform execution. The completed laptop run does not depend
   on BDH access.

### Fred Popowich / paper coordination

The computation is complete enough to begin a bounded revision after Chris's
author checkpoint. The remaining blockers are scientific/governance inputs,
not access to the historical BDH machines. Coordination should preserve Chris's
expertise and the submitted source-of-record boundary while scheduling patient,
clinical and ethics decisions with their appropriate owners.

## Preservation and transfer

The governed run directory contains a transfer manifest with relative paths,
sizes, sensitivities and SHA-256 values. Transfer only through an approved
channel, preserve relative paths, and verify every hash before resuming or
comparing on another machine. Case-level inputs, outputs, caches and error
packets remain governed and ignored by Git. The branch contains only code,
tests, aggregate documentation and public-safe receipts.
