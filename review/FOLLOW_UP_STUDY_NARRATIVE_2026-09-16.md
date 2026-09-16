# Follow-up study narrative: from portability to grounded explanation

**Status:** framing document for a future methods paper. Not part of the
current JBHI revision, not a manuscript draft, and not required by anything
already submitted or in review.

## Where this picks up

The JBHI revision under review already establishes one bounded fact: the
report-to-result contract — five categories, four-level labels, a
grammar-constrained classification call — transports to a second, newer local
model (MedGemma) without changing the task. That claim is deliberately narrow.
It says the classification layer survived a model change. It says nothing
about the second, thesis-originated call: whether a model's stated reasons for
a decision are actually grounded in the report it read.

This follow-up closes that gap, on the same development surface, without
touching anything the current revision claims.

## What is now established, concretely

On the frozen 100-report Zoe development set, under one strict rule (a
verified quotation is an unchanged, nonblank substring of the source report —
no fuzzy or semantic credit in the primary count):

- MedGemma: 330 of 728 substantive phrases were exact quotations (45.3%);
  484 of 500 report-category units carried any substantive evidence at all.
- Saved Mistral: 53 of 295 were exact quotations (18.0%); 245 of 500 units
  carried substantive evidence.

This is a real configured-system difference on this development surface:
MedGemma both engages more often (fewer declared-no-evidence units) and, when
it does, quotes the source more literally. It is not yet a claim that
MedGemma's reasons are better — only that they are more often present and more
often traceable by this measure. The saved streams also use different evidence
schemas, so this result cannot be attributed to model weights alone.

## The honest gap this doesn't close

Source presence is not entailment. A phrase can be an exact quotation and
still be the wrong reason, an irrelevant one, or a restatement that doesn't
actually justify the category it's attached to. The technical audit that
produced the two numbers above cannot see that distinction — only a human
reading the report and the phrase together can. That is precisely the
question a blinded, stratified review sample answers, and precisely why nothing
past that review is appropriate to run yet: scaling to the full 1,894-report
evaluation surface (~25 hours of local compute) would sharpen the precision of
the same two numbers without answering whether either model's phrases are
actually *right*.

The frozen review therefore uses 20 report-category cases, not 203 phrase
rows. It includes one same-four-level-decision pair for every combination of
the five EEG categories and four automated focus conditions: exact quotation,
normalization-only candidate, unresolved phrase and no evidence. The reviewer
reads the report first, then judges counterbalanced System A/System B evidence.
Model identity and automated match labels remain in separate governed files.

## Why this is completion, not expansion

Chris's original two-call design — classify, then extract supporting
phrases — already embodied the idea that a model's confidence should be
checked against something legible to a human reader, not taken on faith. The
project's existing tools (`evidence_alignment.py`, `evidence_factuality.py`,
now the stricter `reason_traceability.py` audit) are refinements of a question
Chris was already asking. Running that same second call against a second
model, and building the machinery to compare the two fairly, is finishing that
line of work on a new instance, not inventing a new direction. Nothing here
required a new theory of the pipeline — it required the pipeline to be used
completely, once, on the model it hadn't yet been used on.

## What a positive review result would open, and what it would not

If the blinded review shows that the rubric can distinguish relevant,
decision-supporting evidence from merely located text, three things become
legitimately askable, in order of how much new work each needs. The purposive
sample itself cannot establish which system succeeds more often:

1. **A full-cohort evidence run**, now justified by review evidence rather
   than by the development-set numbers alone, giving a real evaluation-surface
   estimate rather than a development diagnostic.
2. **Category-conditioned analysis of *where* grounding is strong or weak** —
   does either model quote well on abnormal-presence categories but poorly on
   the rare epileptiform ones? That's a direct, low-cost extension of data
   already collected.
3. **Explanation-informed prompt refinement** — using the specific reports
   where a model's phrase was ungrounded to sharpen the extraction prompt,
   replacing this project's historical trial-and-error refinement process
   (already documented honestly in the current revision as exploratory) with
   a refinement loop that has a stated reason for each change.

A stronger causal question would require one more bounded experiment before
making model-level claims: cross the model and evidence schema on the same
development reports and fixed decisions. That 2-by-2 design would separate a
model-family effect from the effect of decision-conditioned versus independent
evidence prompts. The present review does not answer that question and does not
silently treat the two schemas as equivalent.

None of this requires, and this narrative deliberately does not reach for,
mechanistic comparison of the two models' internals. That remains a separate,
much longer-horizon research question, useful to keep in view but not a
prerequisite for anything above.

## What a null or mixed review result would mean

Equally reportable: if the review shows both models' "exact quotations" are
frequently ungrounded despite passing the substring test, that is itself a
finding — it would mean literal traceability is a necessary but insufficient
proxy for good explanation, which is exactly the kind of boundary this
project's discipline is built to state plainly rather than paper over.

## Scope discipline

This document exists to keep the direction legible across sessions, not to
commit to writing a second paper on any timeline. The current JBHI submission
does not reference, depend on, or wait for any part of this.

The review is a feasibility gate, not a miniature performance study. Expansion
stops if reviewers find the rubric unusable, the schemas incomparable, exact
quotes commonly irrelevant or contradictory, or no-evidence outputs commonly
miss obvious source material. Expansion becomes worth considering only if the
review works across all five categories, identifies interpretable successes and
failures, and shows that full-cohort rates would answer a defined follow-up
question. That decision and its reasons must be recorded before any full-cohort
run is launched.

The first blinded technical-reader pass is now complete across all 20 cases.
It confirmed that the rubric distinguishes literal presence, category relevance,
decision support and evidence omission, while leaving genuine EEG ambiguities
unresolved. This passes the tooling-feasibility gate and advances the study to
an independent EEG-qualified read; it does not yet authorize unblinding or the
full-cohort run. See `SOURCE_FIRST_TECHNICAL_READER_PASS1_2026-09-16.md`.

The independent clinical-reader delivery is now frozen separately from the
original instrument. It contains the same 20 cases and A/B assignments but none
of the first reader's answers, model identities, selection metadata or automated
focus strata. A blinded two-reader comparison is also implemented: it reports
raw field-level agreement and produces a disagreement queue without copying
report text, evidence phrases or notes. The study remains at **awaiting
independent EEG-qualified review**. See
`INDEPENDENT_CLINICAL_EVIDENCE_REVIEW_HANDOFF_2026-09-16.md`.

The retained Reference Annotator labels have now also been harvested without
opening the blinded review. Across 500 development report-category units per
configured system, binary/exact agreement was 462/416 for MedGemma and 475/411
for Mistral. MedGemma supplied substantive evidence in 484 units and an
unchanged quotation in 305; the saved Mistral stream did so in 245 and 48.
Conditioning on agreement revealed different directions across the two evidence
interfaces, so literal traceability cannot be treated as a universal proxy for
reference agreement. See
`REFERENCE_ALIGNED_EVIDENCE_TRACEABILITY_RESULT_2026-09-16.md`.

The resulting model-by-evidence-schema crossing is now complete on the frozen
20-report development surface. Only the missing Mistral independent-evidence
cell required new inference; three invalid outputs were retained without retry.
Changing from decision-conditioned to independent evidence reduced MedGemma
coverage from 97% to 68% but increased Mistral coverage from 57% to 75%.
Unchanged-quotation coverage likewise moved from 60% to 34% for MedGemma and
from 26% to 36% for Mistral. The opposite directions establish that evidence
behavior is a model-by-interface property, not a model-weight ranking. This
closes the computational factor-attribution step and keeps the full-cohort run
on hold. See `EVIDENCE_SCHEMA_FACTORIAL_RESULT_2026-09-16.md`.

That finding is now operationalized as a two-axis compatibility contract for
future work. Evidence generation names either the thesis-lineage
decision-conditioned pathway or independent category evidence; evidence claims
separately name the accepted provenance/review layer. Unknown combinations and
the attempt to apply the original-thesis source-support calculation to the
independent schema fail closed. Historical runners remain untouched. See
`EVIDENCE_PROFILE_IMPLEMENTATION_2026-09-16.md`.

The publication paths are now separated explicitly. The current JBHI revision
retains only the completed classification-portability result and its
classification-only boundary. The evidence-schema interaction, profile
registry and any future relevance review belong to the follow-up methods paper.
See `JBHI_TO_FOLLOW_UP_CLAIM_FIREWALL_2026-09-16.md` and
`FOLLOW_UP_METHODS_PAPER_DRAFT_2026-09-16.md`.
