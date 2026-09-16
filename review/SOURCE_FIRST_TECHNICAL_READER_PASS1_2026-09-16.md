# Source-first evidence review: technical-reader pass 1

**Status:** complete blinded technical read on the frozen 20-case development
package. The reader was not EEG-qualified. This is a feasibility result, not
clinical adjudication, a prevalence estimate, a system ranking or evidence for
the current JBHI revision.

The reader did not inspect `unblinding_key.json` or `analysis_metadata.json`.
The governed response, blinded summary and pass-specific receipt remain under
`review-source-first-v2/`; report text and free-text notes do not enter this
repository record.

## Completion

- 20/20 source-first report-category cases reviewed.
- 40/40 blinded configured-system evidence sets reviewed.
- Source judgment recorded before evidence reveal.
- System A/B identities remain blinded.
- Raw response and blinded summary are hash-bound to the frozen review package.

## Blinded aggregate

| Judgment | System A | System B |
|---|---:|---:|
| Source presence: yes | 14 / 20 | 14 / 20 |
| Category relevance: yes | 12 / 20 | 10 / 20 |
| Supports stated decision: yes | 12 / 20 | 9 / 20 |
| Sufficient as explanation: yes | 12 / 20 | 9 / 20 |
| No-evidence omission judged unreasonable | 1 | 3 |

Pair preference was tie in 12 cases, System A in four, System B in two,
neither in one and unclear in one. Because the sample is purposive and the
reader is not clinically qualified, these counts must not be interpreted as
comparative performance estimates.

## What the pass established

1. **The rubric is usable.** The reader completed every source-first,
   individual-system and paired field without unblinding.
2. **Literal location is insufficient.** At least one case had a plausible
   shared decision but evidence that was literal or topically adjacent rather
   than the most relevant reason for that category.
3. **Evidence omission is inspectable.** Several paired cases exposed a useful
   asymmetry: one system returned no phrase while the other located directly
   reviewable material in the same report.
4. **The clinical boundary worked.** The reader marked genuine EEG
   interpretation uncertainty rather than converting textual confidence into
   clinical authority.

## Priority cases for clinical interpretation

- `C004` and `C009`: the technical reader's source interpretation disagreed
  with the shared configured-system decision; generalized non-epileptiform
  interpretation needs an EEG-qualified read.
- `C006` and `C007`: bilateral/focal versus generalized epileptiform language
  was not resolved by a text-only reading.
- `C014`: useful test of a correct-seeming decision accompanied by evidence
  that may not state the category-defining reason.
- `C019`: internal report/category ambiguity needs domain interpretation.

These handles are review priorities, not error labels.

## Next gate

Keep the identities blinded. Obtain an independent pass over the same 20 cases
from an EEG-qualified reader, without showing this first reader's answers.
Then freeze the second response, compare agreement and disagreements while
still blinded, adjudicate the clinical-priority cases, and only then unblind.

The first pass is promising enough to justify the qualified review. It is not
yet sufficient to justify the approximately 25-hour full-cohort evidence run.
After adjudication, choose explicitly among: stop; run the configured-system
full cohort; or preregister the smaller 2-by-2 model-by-evidence-schema study
needed for factor attribution.
