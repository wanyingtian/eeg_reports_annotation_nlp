# MedGemma v1 governed error-review packet

Status: packet generated for authorized clinical review; no clinical
adjudication performed. No model inference was run. This closes the tooling
gap noted in `REVIEWER_EXECUTION_MATRIX.md` ("Obtain clinical-team approval
for the implemented governed error-review protocol, then review the sampled
cases") for the completed MedGemma v1 comparator: until now, `eeg-review
error-review` had only been run against Mistral outputs.

## What was run

`eeg-review error-review` (unchanged tool, `CLINICAL_ERROR_REVIEW_PROTOCOL.md`
contract) against the completed, protected `medgemma-27b-native-protected-v1`
predictions from `data/governed/study-runs/jbhi-medgemma-native-protected-20260830/`,
matched to the same evaluation reference DBs used for the full-cohort result in
`COMPLETED_MODEL_COMPARISON_FINDINGS_2026-08-30.md`. Producing code revision
`d026c07`. Its receipt correctly records a dirty worktree because of a
pre-existing, unrelated `.gitignore` edit.

No `--cluster-column` was supplied: the reference snapshot's `Cluster code`
field has only one distinct value per cohort and was already established as
not a patient identifier (`MEDGEMMA_INDEPENDENT_COMPARATOR_STUDY.md`). Sampled
rows may therefore include repeated patients; the packet's own
`interpretation_limits` field states this explicitly. Each cohort used a fresh
32-byte random handle salt. The repaired generator emits a separate governed
lookup from each portable case handle to its source report key. All 139 Zoe
handles and all 57 Maria handles resolve exactly once. The packet is therefore
ready for a qualified reviewer inside the authorized environment; the lookup
is not portable and must remain governed.

```
data/governed/analysis-runs/jbhi-medgemma-error-review-20260907-v2/
  zoe-medgemma/{clinical_error_review_summary.json, clinical_error_review_packet.csv,
                clinical_error_review_lookup.csv, run_manifest.json}
  maria-medgemma/{clinical_error_review_summary.json, clinical_error_review_packet.csv,
                  clinical_error_review_lookup.csv, run_manifest.json}
```

Governed, gitignored; not for Git, email, or circulation, per protocol.

## What the aggregate confirms

The per-label totals reproduce the confusion-matrix counts already reported in
`COMPLETED_MODEL_COMPARISON_FINDINGS_2026-08-30.md`, together with the intended
stratified sampling counts:

| Cohort | Label | FN total | FN sampled | FP total | FP sampled |
|---|---|---:|---:|---:|---:|
| Zoe (1,395) | Focal Epi | 0 | 0 | 36 | 25 |
| Zoe | Abnormality | 24 | 24 | 53 | 25 |
| Zoe | Focal Non-epi | 16 | 16 | 105 | 25 |
| Zoe | Gen Epi | 1 | 1 | 15 | 15 |
| Zoe | Gen Non-epi | 46 | 25 | 37 | 25 |
| Maria (499) | Gen Epi | 2 | 2 | 5 | 5 |
| Maria | Gen Non-epi | 21 | 21 | 6 | 6 |

(Full five-label table for both cohorts is in the packet summaries; only the
labels most relevant to the focal-epileptiform question are excerpted here.)

Zoe selected 181 label-case rows total; Maria selected 90. This is the same
protocol, sampling cap (25/stratum), and seed (`20260718`) as the existing
Mistral packets (194 Zoe / 145 Maria rows, `error-review-20260720/`), so the
two are directly comparable in method, not yet in content — no cross-model
case linkage was requested or produced.

The regenerated artifacts are hash-bound by their run manifests. The Zoe
packet and lookup SHA-256 values begin `69c1b4907a29` and `ccee1fe4d370`;
the Maria values begin `1b7e7a500d7b` and `b7f550b81bb6`. The first,
unresolvable packet remains retained as an audit trail and is superseded by
the `-v2` directory above.

## What this does not do

- No clinical salience, workflow-consequence, or escalation judgment has been
  entered. All reviewer fields are blank, per protocol.
- Does not establish patient independence; sampling is report-level.
- Does not resolve the focal-epileptiform trade-off itself — it only makes 25
  of the 36 Zoe focal-epi false positives (capped at the per-stratum limit)
  reviewable.
- Requires the same unresolved external steps as before: clinical-lead sign-off
  on reviewer qualifications, a codebook for the review fields, and the
  approved environment to open the packet in.

## Next step

Name a qualified clinical reviewer and agree on the review-field codebook,
then hand off the packet and its separate lookup inside the approved
environment. No further model inference is needed.
