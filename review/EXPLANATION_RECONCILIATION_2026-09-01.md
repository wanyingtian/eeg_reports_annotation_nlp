# Historical explanation-surface reconciliation

**Status:** aggregate author-working evidence; not manuscript admission and not
clinical validation.

## What this closes

The granular explanation surface used in the thesis has been recovered as a
strong producing-artifact candidate. The frozen CSV has 2,000 unique Zoe report
keys, 2,000 unique report texts, five four-level Mistral decisions, five reason
fields, and five saved ClinicalBERT+LR polarity fields. Its report keys occur in
the same order as the first 2,000 records of the historical Zoe source snapshot,
and all 2,000 report texts match exactly. The artifact SHA-256 is
`6e9c56bba4f87b0e130087612bbb7fdb8bee3f8c0559385c2a86f8b29c8ea605`.

This establishes that report-component access still exists. It does **not**
turn Mistral's requested reasons into causal explanations, make SHAP and reason
phrases interchangeable, or establish clinical usefulness.

## The three distinct questions

1. **Polarity classification:** can the saved ClinicalBERT+LR classifier label a
   reason as supporting normality or abnormality? The first 200 rows were the
   polarity-classifier training surface; the saved post-training polarity labels
   cover between 1,764 and 1,796 rows per category after explicit exclusions.
2. **Text traceability:** can an abnormal-supporting reason be found in or linked
   back to its source report under a declared matching rule? The learned polarity
   fields select exactly 2,180 report-category units, reproducing the submitted
   denominator.
3. **Decision consistency:** when a reason's saved polarity agrees with the
   Mistral core decision, is that decision more often correct against the
   Reference Annotator? This is an association check, not causal faithfulness.

The five-category classification evaluation and the explanation analysis do not
have the same population. The historical polarity test surface begins after the
fixed first 200 polarity-training rows. The author-confirmed current Zoe
evaluation manifest contains 1,395 reports, but 100 of those reports fall inside
the polarity-training interval and therefore have no saved post-training
polarity prediction. Results below name the surface used rather than silently
coercing the populations to match.

## Polarity-classifier check

The recovered artifact closely reproduces the thesis table without retraining
ClinicalBERT. Accuracy on saved post-training predictions is:

| Category | Usable report reasons | Recomputed accuracy |
|---|---:|---:|
| Focal epileptiform | 1,796 | 99.44% |
| Generalized epileptiform | 1,786 | 98.94% |
| Focal non-epileptiform | 1,764 | 87.41% |
| Generalized non-epileptiform | 1,775 | 97.24% |
| Overall abnormality | 1,773 | 92.72% |

The small differences from two printed thesis percentages are retained. No row
filter or label was changed to force an exact match.

## Correctness when reason polarity agrees with the decision

On the historical post-training surface, the submitted headline pattern is
substantially recoverable from raw counts:

| Category | Aligned correct / total | Misaligned correct / total | Accuracy difference (aligned minus misaligned) |
|---|---:|---:|---:|
| Focal epileptiform | 1,761 / 1,783 (98.77%) | 6 / 9 (66.67%) | 32.10 percentage points |
| Generalized epileptiform | 1,712 / 1,762 (97.16%) | 12 / 19 (63.16%) | 34.00 percentage points |
| Focal non-epileptiform | 1,361 / 1,537 (88.55%) | 146 / 221 (66.06%) | 22.49 percentage points |
| Generalized non-epileptiform | 1,569 / 1,722 (91.11%) | 31 / 47 (65.96%) | 25.16 percentage points |
| Overall abnormality | 1,611 / 1,639 (98.29%) | 92 / 128 (71.88%) | 26.42 percentage points |

For overall abnormality, a conservative interval formed from the two Wilson
intervals is 18.60 to 35.28 percentage points. The submitted 98.3% aligned value
rounds exactly from the recovered counts; the recovered misaligned value is
71.9%, not the printed 72.5%. The original phrase “25% more likely” should
therefore be replaced by a named accuracy or error-risk difference with counts,
not defended as a relative effect.

Across categories, 343 of 424 misaligned report-category decisions (80.90%)
carry a low-confidence Mistral level (2 or 3), supporting the descriptive “over
80%” statement. Pooling is descriptive because five category decisions from one
report are not independent observations.

As a population sensitivity check, the declared 1,395-report Zoe evaluation
manifest yields a 24.36-percentage-point overall-abnormality difference among
the rows with saved post-training polarity (1,156/1,179 versus 70/95;
conservative 95% interval 15.60 to 34.66 points). This sensitivity does not
replace the historical explanation surface.

## Why 97.8% remains gated

The recovered learned-polarity fields select exactly 2,180 positive units, but
the public factuality script uploaded in October 2025 is not the exact producing
method:

- unchanged, it selects Mistral labels 3/4 (2,191 units), whereas the thesis says
  the learned polarity labels selected the 2,180 units;
- after correcting only that selection, replaying its normalized substring,
  whole-report fuzzy, and whole-report MiniLM semantic stages at the documented
  thresholds matches 2,018/2,180 (92.57%), not 2,132/2,180 (97.80%);
- a separately named sentence-level diagnostic matches 2,153/2,180 (98.76%).
  It is plausible and often more sensible to compare a short reason with report
  sentences, but this is a different method and cannot be substituted because
  it happens to be near the submitted value.

The subsequent repository-and-thesis search recovered two different
specifications rather than the missing producer. Chris's October 2025 public
script splits on semicolons and compares against whole reports; the final thesis
describes explanation sentences, report-sentence fuzzy comparison, and
whole-report semantic comparison. With the fixed learned-polarity selection,
the latter reconstruction gives 1,911/2,180 (87.66%). Neither specification
reproduces 2,132. The current author draft should not restore the number, and no
question to Chris is needed merely to continue technical work. The revision can
instead use the separately named exact-source and review-candidate contract in
`REASON_TRACEABILITY_EXPERIMENT_2026-09-02.md`.

## Reproducible execution and custody

The validator is `scripts/reconcile_explanation_artifact.py`. It verifies keys,
source order, report text, model labels, polarity values, cohort manifests, and
all denominators before writing results. The semantic replay used the local
`sentence-transformers/all-MiniLM-L6-v2` snapshot at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, threshold 0.70. The latest
upstream revision predating the producing artifact is
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`; only the model card differs between
these revisions, while weights, tokenizer, pooling modules, and executable
configuration are byte-identical. The historical code was unpinned, so this is
functional equivalence rather than proof of the originally resolved revision.
No report text was sent to a remote service.

The aggregate receipt can be reviewed publicly. Report keys and case-level
traceability/alignment rows remain under `data/governed/` and are ignored by
Git. No report text, reason text, report key, or keyed prediction is copied into
this document.

## Publication decision

- **Supported now:** the 2,180 denominator; granular reason/polarity availability;
  close reproduction of the polarity table; raw alignment/correctness counts;
  the direction and magnitude of the association; and the descriptive 80.9%
  low-confidence overlap.
- **Needs corrected wording:** replace “25% more likely” with the declared
  percentage-point difference, raw counts, interval, unit, and non-causal limit.
- **Historical only:** the submitted 97.8% traceability numerator. The available
  producer search is exhausted; do not recreate it by tuning. The 72.5%
  abnormality-misaligned percentage also remains unreproduced.
- **Not claimed:** causal reasoning, faithful internal model reasoning, clinical
  validation, patient independence, or interchangeability with SHAP.
