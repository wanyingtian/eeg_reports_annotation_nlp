# Clinical error-review checkpoint — 2026-07-20

Status: internal, provisional, aggregate counts only. “False negative” and
“false positive” below are relative to the Reference Annotator after collapsing
four levels into core absent (1–2) and present (3–4). They are not clinical
adjudications.

## Complete historical-cohort counts

| Cohort | Category | RA-relative FN | RA-relative FP |
|---|---|---:|---:|
| Zoe (N=1,395) | Abnormality | 35 | 19 |
| Zoe (N=1,395) | Focal Epi | 2 | 16 |
| Zoe (N=1,395) | Gen Epi | 7 | 38 |
| Zoe (N=1,395) | Focal Non-epi | 73 | 118 |
| Zoe (N=1,395) | Gen Non-epi | 113 | 28 |
| Maria (N=499) | Abnormality | 33 | 9 |
| Maria (N=499) | Focal Epi | 5 | 8 |
| Maria (N=499) | Gen Epi | 2 | 5 |
| Maria (N=499) | Focal Non-epi | 45 | 25 |
| Maria (N=499) | Gen Non-epi | 42 | 16 |

These are complete matched-pair counts for the submitted Mistral output and
the recovered RA-selected cohorts, not sampled estimates. They should be read
with reference positive support, sensitivity/specificity, and uncertainty.

## Governed packet exercise

With a cap of 25 cases per category and error direction, the deterministic
packet selected 194 Zoe label-case rows and 145 Maria label-case rows. Strata
with fewer than 25 eligible disagreements retain every eligible case. Original
report IDs, patient IDs, and report text are absent from the worksheet.

## Gates before clinical use

1. Confirm the stable patient key and regenerate a patient-diverse selection.
2. Obtain data-custodian approval for the case-resolution/review environment.
3. Have the clinical lead approve reviewer roles and the controlled codebook.
4. Predeclare how multiple ratings, uncertainty, and adjudication are handled.
5. Return only aggregate review receipts to the manuscript workspace.

Governed case packets are under `data/governed/analysis/error-review-20260720`
and remain ignored by Git.
