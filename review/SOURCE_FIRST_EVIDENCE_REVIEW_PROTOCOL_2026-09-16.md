# Source-first evidence-review protocol

**Status:** frozen feasibility review for a future methods study. This is not
part of, and does not delay, the current JBHI revision.

## Question

When two configured local systems make the same four-level classification on
the same development report and EEG category, are their extracted phrases
present in the source, relevant to the category, supportive of the declared
decision and useful to a human reviewer?

This is a configured-system question. The saved Mistral and MedGemma streams
use different evidence schemas; the review cannot attribute differences to
model weights alone.

## Frozen sample

- Surface: the declared 100-report Zoe development set.
- Unit: one report-category pair.
- Size: 20 units and 40 blinded system reviews.
- Pairing: both systems must have made the same exact four-level decision.
- Coverage: one unit for each of five EEG categories within each of four
  automated focus conditions: exact quotation, normalization-only candidate,
  unresolved phrase and no evidence.
- Selection: deterministic content hashes; pure two-system instances of a
  focus condition are preferred before one-system contrasts.
- Blinding: A/B labels are counterbalanced by case. Model identities, report
  keys, automated focus conditions and match stages are withheld.

The sample is purposive. Counts from it are not prevalence or performance
estimates.

## Review order

1. Read the source report and category without model evidence.
2. If qualified, record a provisional category judgment and the passages that
   matter. “Not qualified” is an allowed response.
3. Reveal System A and System B.
4. Judge each evidence set independently for source presence, category
   relevance, support, contradiction and sufficiency. When no phrase was
   returned, judge whether that omission was reasonable.
5. Only then record A/B/tie/neither comparative usefulness.
6. Save the response before unblinding.

The review does not ask for hidden reasoning, a new diagnosis or clinical
correctness beyond the reviewer's qualifications.

## Decision gate

Do not launch the full 1,894-report evidence extraction automatically. Record a
go/no-go decision after review. Stop if the rubric is unusable, the schemas
cannot be compared meaningfully, exact quotes are commonly irrelevant or
contradictory, or no-evidence outputs commonly miss clear source material.
Consider expansion only if review is interpretable across all five categories
and full-cohort rates would answer a specific follow-up question.

If model-versus-schema attribution becomes the question, preregister a small
2-by-2 development experiment crossing model and evidence schema. The current
configured-system review is not that factorial experiment.

## Governed implementation

The generated package lives under the completed development run in
`review-source-first-v2/`. Open `review_form.html` locally. It makes no network
requests and can download a JSON response. The CSV files provide the same
three stages for table-oriented review. Do not inspect `analysis_metadata.json`
or `unblinding_key.json` until the blinded review is complete.
