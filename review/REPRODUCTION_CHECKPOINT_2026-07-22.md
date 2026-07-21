# Reproduction checkpoint for the 2026-07-22 meeting

## Meeting and communication record

- Tentative meeting: Wednesday, 2026-07-22, 19:30–20:00 PDT.
- Wanying (Chris) Tian offered to create the meeting invitation.
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
columns. The uploaded Zoe inference CSVs therefore do not appear to be the exact
artifacts used for those manuscript columns, or the manuscript used a distinct
development/CV surface for Zoe. This is a bounded artifact-provenance question,
not evidence that the full study must be rebuilt.

Report-level bootstrap 95% intervals were generated for all eight comparisons.
They are explicitly provisional because the supplied databases contain no
patient/cluster key; patient-clustered intervals remain blocked on governed
patient mapping and custodian confirmation.

## Work before Wednesday

1. Reconcile the commented core/certainty table, the rendered figures, and the
   prose kappa values against maintained aggregate calculations.
2. Search recovered filenames/code and request only the missing Zoe baseline
   fold/development artifacts needed to identify the 39 cells.
3. Verify whether the submitted Zoe baseline values were five-fold development
   summaries, external-evaluation values from another run, or a table assembly
   error.
4. Preserve a discrepancy ledger with submitted, recomputed, unrounded, and
   rounded values plus artifact hashes.
5. Prepare a short Vasily-facing execution plan separating tasks already
   completed locally from patient-key, REB, and clinically governed decisions.

## Questions for Chris if the artifact search does not resolve them

1. Which exact files generated the Zoe BoW+LR and BERT+LR columns in
   `tab:core_zoe_vs_maria` and the combined-performance figure?
2. Were those Zoe values computed from five-fold predictions on the first 100
   development reports, from the 1,395 external evaluation reports, or from a
   different saved run?
3. Are fold assignments, out-of-fold predictions, fitted vectorizers/models, or
   earlier `zoe_inference_results_*` versions still present on the BDH machine?
4. Was any manual table assembly or post-processing step applied after
   `compute_all_category_metrics`?

No request for a new base model is needed at this checkpoint. Reproducing and
auditing the native submitted framework comes first; any contemporary comparator
is a separately approved, identically evaluated revision experiment.
