# Reproduction checkpoint for the 2026-07-22 meeting

## Meeting and communication record

- Confirmed check-in: Wednesday, 2026-07-22, 19:30 PDT.
- The Monday handover call is complete.
- Steven's SFU address remains the preferred project correspondence address;
  the private Gmail address is not a project communication channel.
- The intended next team step is a focused Steven/Chris checkpoint before a
  meeting with Vasily Vakorin later in the week or early the following week.

## Scientific decision rule

The submitted and preprint values are a reproduction target, not an optimization
target. Source artifacts and recovered outputs remain immutable. If a maintained
recalculation differs, retain both values and determine whether the cause is:

1. a path, sheet, row-coordinate, or label-order mismatch;
2. a different historical model/run/fold artifact;
3. a documented metric-definition or rounding difference;
4. a historical implementation defect; or
5. a substantive result correction.

Only the first three may resolve through mechanical reconciliation. A defect or
correction must be disclosed to all authors and explained transparently in the
revision/response; numbers must never be silently adjusted to resemble the
submitted display.

## Materials now under control

- Four complete Zoe/Maria LD/SG SQLite corpora, frozen by SHA-256.
- Processed 2,000-row Zoe and 500-row Maria Mistral workbooks.
- Probability-bearing BoW+LR and BERT+LR inference CSVs: 1,900 Zoe rows and 500
  Maria rows for each model.
- Chris's four recovered result-analysis scripts, preserved byte-for-byte.
- Chris's native Overleaf source archive and the portal-submitted PDF in the
  private companion paper repository.
- The exact Mistral-7B-Instruct-v0.2 Q5_K_M model file, revision, and checksum;
  a one-report inference/post-processing run has succeeded on Steven's Mac.

No governed report text, report identifier, or case-level output is committed.

## First full-table reproduction result

The maintained evaluator applied source ranges `[100:500] + [1000:2000]` for
Zoe and `[0:500]` for Maria, required all five RA labels, aligned each prediction
surface by `Hashed ID`, and recomputed the manuscript's five core metrics for
five categories, four comparators, and two cohorts. This produced the expected
analysis populations of 1,395 Zoe reports and 499 Maria reports.

Comparison against the 200 two-decimal cells in supplementary Table
`tab:core_zoe_vs_maria` gave:

| Cohort / comparator | Matching cells | Different cells |
|---|---:|---:|
| Zoe / Second Annotator | 25 | 0 |
| Zoe / Mistral-7B | 25 | 0 |
| Zoe / BoW+LR | 9 | 16 |
| Zoe / BERT+LR | 2 | 23 |
| Maria / Second Annotator | 25 | 0 |
| Maria / Mistral-7B | 25 | 0 |
| Maria / BoW+LR | 25 | 0 |
| Maria / BERT+LR | 25 | 0 |
| **Total** | **161** | **39** |

Undefined precision for a model with no positive predictions was compared as
the historical code's `zero_division=0`; this display convention does not
explain the broader discrepancies.

This pattern rules out a general cohort-selection, RA-reference, label-collapse,
or current metric-formula problem: every Maria value and every human/Mistral
value reproduces. The unresolved values are isolated to the two Zoe baseline
columns.

The native paper source preserves the submitted 4x4 aggregate confusion
matrices for all baseline categories. A hash-checked transcription and
independent collapse of those matrices reproduces all 100/100 submitted
two-decimal baseline cells, including the 39 that differ from the uploaded Zoe
CSVs, all 20/20 certainty-adjusted baseline accuracies in the combined figure,
and all 40/40 RA-to-baseline values in the four core/certainty kappa grids. The
Maria figure matrices and uploaded files match cell-for-cell. The Zoe matrices
do not, but all ten Zoe RA truth-row marginals match exactly. We can therefore
distinguish the cases precisely: the evaluation cohort and RA labels are the
same, while the currently uploaded Zoe files contain different saved model
predictions from the report-level artifacts used for the submission.

This is a bounded artifact-version question, not evidence that the full study
must be rebuilt. Aggregate submitted metrics and kappa are now auditable;
report-level paired inference against the submitted Zoe baselines and their
calibration still require the exact producing Zoe predictions.

Report-level bootstrap 95% intervals were generated for all eight comparisons.
They are explicitly provisional because the supplied databases contain no
patient/cluster key; patient-clustered intervals remain blocked on governed
patient mapping and custodian confirmation.

## Reviewer-facing inference receipts now executable

The maintained branch now has two aggregate-only analysis commands that align
inputs on the same selected reports and bind results to input hashes, command
settings, seeds, and Git revision:

- `eeg-review compare` reports model-A-minus-model-B differences in binary core
  accuracy, exact four-level accuracy, and false-negative rate; paired
  bootstrap intervals; discordant correctness cells; exact McNemar tests; and
  Holm-adjusted p-values across the declared five-label family.
- `eeg-review calibrate` reports prevalence, Brier score, log loss, fixed-bin
  ECE, and bin-level support/event rates for positive-class probabilities.

The paired command was exercised on Mistral versus the Second Annotator in both
cohorts and on Mistral versus the exact submitted Maria BoW/BERT rows. These
provisional results retain unfavorable findings: Mistral is below the Second
Annotator on exact four-level accuracy for nearly every category, and a
majority-level baseline can have higher raw exact accuracy for rare focal
epileptiform findings while having zero positive recall and near-zero kappa.
Accuracy therefore cannot be interpreted without support, recall, kappa, and
the paired effect.

Calibration was exercised only where the artifact semantics permit it. On the
exact submitted Maria rows, BERT has lower Brier score and fixed-bin ECE than
BoW for abnormality and both non-epileptiform categories; the generalized
epileptiform results are similar, with only 20/499 RA core-positive reports.
Mistral's generated four-level label is not a probability and is not relabeled
as calibrated by this analysis.

## Work before Wednesday

1. Search recovered filenames/code and request only the missing report-level
   Zoe baseline files needed for paired uncertainty and error analysis.
2. Verify which historical prediction export produced the submitted Zoe
   matrices; the matching RA marginals now rule out a development-set summary
   or a different evaluation selection.
3. Obtain or define the governed patient key and rerun both paired and
   calibration intervals by patient cluster; until then, report-bootstrap
   intervals remain sensitivity analyses.
4. Preserve a discrepancy ledger with submitted, recomputed, unrounded, and
   rounded values plus artifact hashes.
5. Prepare a short Vasily-facing execution plan separating tasks already
   completed locally from patient-key, REB, and clinically governed decisions.

## Questions for Chris if the artifact search does not resolve them

1. Which exact report-level files generated the Zoe BoW+LR and BERT+LR columns
   in `tab:core_zoe_vs_maria` and the confusion-matrix figures?
2. Which saved inference run produced those 1,395-report prediction counts? The
   source figures confirm the expanded evaluation set rather than a 100-report
   development summary.
3. Are fold assignments, out-of-fold predictions, fitted vectorizers/models, or
   earlier `zoe_inference_results_*` versions still present on the BDH machine?
4. Was any manual table assembly or post-processing step applied after
   `compute_all_category_metrics`?

No request for a new base model is needed at this checkpoint. Reproducing and
auditing the native submitted framework comes first; any contemporary comparator
is a separately approved, identically evaluated revision experiment.
