# Native reproduction active checkpoint - 2026-08-14

## Status and scope

The governed native reproduction was initialized from repository commit
`791498296f344a1cfdc2a1a50da2be0f81338d9f`. At 22:47 CEST on 2026-08-14,
the detached macOS supervisor was healthy, 19 of 38 stages were complete, and
the Zoe Mistral stage had atomically stored 6 of 2,000 reports. Its early
observed ETA was approximately 18.8 hours. This is an operational checkpoint,
not a replacement for the submitted results.

All intervals available at this checkpoint resample reports. The transferred
data do not contain an authoritative patient key, so these values do not
establish patient independence and must not be described as patient-clustered
intervals.

## Early Mistral identity check

The first five completed fresh Zoe reports were aligned to Chris's preserved
historical Mistral classifications by pseudonymous report identifier. All 25 of
25 four-level cells matched exactly; consequently all 25 binary/core decisions
also matched. The comparison used classifications only and emitted no report
text or identifier. This small sequential slice is a pipeline-identity sentinel,
not a performance estimate or a sufficient reproducibility claim.

## Historical Mistral intervals now available

The completed evaluator provides deterministic point estimates and 2,000
report-bootstrap replicates for all five labels. Selected metrics are:

| Cohort | Label | N | Core accuracy (95% CI) | F1 | Certainty-adjusted accuracy |
|---|---|---:|---:|---:|---:|
| Zoe | Focal epileptiform | 1,395 | 0.987 (0.981-0.993) | 0.850 | 0.737 |
| Zoe | Generalized epileptiform | 1,395 | 0.968 (0.958-0.976) | 0.713 | 0.941 |
| Zoe | Focal non-epileptiform | 1,395 | 0.863 (0.846-0.882) | 0.762 | 0.632 |
| Zoe | Generalized non-epileptiform | 1,395 | 0.899 (0.883-0.914) | 0.783 | 0.794 |
| Zoe | Abnormality | 1,395 | 0.961 (0.951-0.971) | 0.961 | 0.772 |
| Maria | Focal epileptiform | 499 | 0.974 (0.960-0.988) | 0.806 | 0.661 |
| Maria | Generalized epileptiform | 499 | 0.986 (0.974-0.996) | 0.837 | 0.972 |
| Maria | Focal non-epileptiform | 499 | 0.860 (0.830-0.890) | 0.745 | 0.687 |
| Maria | Generalized non-epileptiform | 499 | 0.884 (0.854-0.912) | 0.540 | 0.834 |
| Maria | Abnormality | 499 | 0.916 (0.890-0.940) | 0.898 | 0.727 |

These numbers confirm why accuracy alone is not an adequate summary. The
generalized and focal epileptiform labels are uncommon, so their high core
accuracy must be read alongside F1, sensitivity, specificity, confusion counts,
and the intervals already present in the full receipts.

## Paired Mistral-to-second-annotator effects

Differences below are historical Mistral minus Second Annotator on the same
reports. They do not imply that the Second Annotator is ground truth; both are
compared against the Reference Annotator.

| Cohort | Label | Core-accuracy difference (95% CI) | CAA difference (95% CI) |
|---|---|---:|---:|
| Zoe | Focal epileptiform | -0.001 (-0.007 to 0.006) | -0.238 (-0.260 to -0.214) |
| Zoe | Generalized epileptiform | -0.024 (-0.033 to -0.014) | -0.048 (-0.060 to -0.036) |
| Zoe | Focal non-epileptiform | -0.081 (-0.099 to -0.062) | -0.142 (-0.169 to -0.115) |
| Zoe | Generalized non-epileptiform | -0.047 (-0.063 to -0.031) | -0.042 (-0.060 to -0.024) |
| Zoe | Abnormality | -0.019 (-0.031 to -0.009) | -0.115 (-0.141 to -0.089) |
| Maria | Focal epileptiform | -0.014 (-0.028 to 0.000) | -0.317 (-0.357 to -0.273) |
| Maria | Generalized epileptiform | 0.000 (-0.012 to 0.012) | -0.012 (-0.026 to 0.002) |
| Maria | Focal non-epileptiform | -0.106 (-0.136 to -0.074) | -0.152 (-0.198 to -0.108) |
| Maria | Generalized non-epileptiform | -0.088 (-0.116 to -0.062) | -0.076 (-0.106 to -0.048) |
| Maria | Abnormality | -0.068 (-0.096 to -0.042) | -0.198 (-0.242 to -0.152) |

This is a useful reviewer-facing result precisely because it includes
unfavorable effects. The Mistral-vs-human comparison is close in some core
decisions but consistently weaker in four-level/certainty agreement, with the
largest gaps in rare focal activity and abnormality.

## Fresh BoW results are a new receipt, not historical reconstruction

The current five-fold development/refit pathway is fully executable, but it
does not recreate missing historical fold assignments or the absent producing
Zoe row export. Its external predictions must therefore remain labelled as a
fresh native refit.

On Zoe, the fresh BoW model achieved core accuracy 0.938 for abnormality and
0.900 for generalized non-epileptiform activity, but focal epileptiform F1 was
0 and generalized epileptiform F1 was 0.031. On Maria, its abnormality core
accuracy was 0.772 and F1 was 0.791, while both epileptiform F1 values were 0.
The apparently strong rare-label accuracies are consequently majority-negative
behavior rather than evidence of useful event detection.

The development set itself contains only three generalized-epileptiform core
positives. Five-fold stratification is mathematically unavailable for that
label. The receipt leaves its out-of-fold values empty and records an explicit
`external_fit_only` status, while fitting the declared external model on all
valid development records.

Calibration reinforces the same limitation. For example, fresh Maria BoW
abnormality has Brier score 0.153 (report-bootstrap 95% CI 0.138-0.168) and
fixed-bin ECE 0.229. Fresh Zoe abnormality is much better aligned in this run
(Brier 0.045, 95% CI 0.038-0.053; ECE 0.057), demonstrating cohort-dependent
behavior that should be reported rather than pooled away.

## Next evidence gates

The running supervisor will automatically proceed through fresh Zoe processing
and comparison, Maria inference and comparison, governed clinical error-review
packets, and the cached BERT pathway. The next defensible checkpoints are:

1. compare the complete fresh Zoe classifications with the historical 2,000-row
   surface at both four-level and binary/core resolution;
2. confirm that the fresh 1,395-report evaluation metrics remain within, or
   explainably outside, the historical report-bootstrap intervals;
3. repeat the same checks for all 499 usable Maria reports;
4. retain full favorable and unfavorable BERT/BoW results, including rare-label
   failure modes and calibration; and
5. obtain a stable patient key before presenting any clustered interval or
   patient-level leakage claim.
