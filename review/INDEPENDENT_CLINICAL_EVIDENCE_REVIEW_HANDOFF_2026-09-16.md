# Independent clinical evidence review handoff

**Status:** the independent-reader instrument is ready in governed storage. No
second review, adjudication or unblinding has yet occurred.

## Purpose

The first technical read established that the 20-case instrument can separate
five questions that automated quotation matching cannot: whether a source
passage is present, relevant to the named EEG category, supportive of the
stated decision, contradictory, and sufficient as a short explanation. It also
identified cases where a technically literate reader should not substitute for
EEG judgment.

The next gate is therefore one independent read by an EEG-qualified reviewer.
This is not a request to review model performance or select a preferred model.
The reviewer sees the report first, then two counterbalanced blinded evidence
panels, and records clinical judgments before system identity is known.

## What the reviewer receives

The governed delivery folder contains only:

- the local, network-free HTML review form;
- three equivalent staged CSV files;
- a short independent-review instruction file; and
- a hash receipt for the delivered material.

It excludes the first reader's answers and summary, model identities,
selection metadata, automated match strata and the unblinding key. The same
20 cases and A/B assignments are retained so that reader agreement can later
be assessed without changing the instrument.

## Review procedure

1. Complete all 20 cases without consulting the first reader's results.
2. Read the report before opening either system panel.
3. Use `unclear` rather than infer beyond the reader's clinical basis.
4. Save the response JSON in the authorized governed return location; do not
   return report-bearing material through ordinary email.
5. Keep system identity blinded after submission of the response.

## What happens after return

The comparison tool produces raw field-by-field agreement counts and a
case-level, fillable adjudication queue while both systems remain blinded. It does not
copy report text, evidence phrases or free notes into the summary, and it does
not calculate a headline kappa from this purposive 20-case set. The two readers
then resolve or explicitly retain disagreements. Only after that record is
frozen is unblinding scientifically interpretable.

## Decision boundary

This review can justify one of three recorded decisions:

1. stop because the rubric or evidence is not useful enough to scale;
2. run the same configured-system comparison on the full evaluation surface;
   or
3. first run a development-only model-by-evidence-schema experiment to
   separate model effects from prompt/schema effects.

It cannot by itself establish clinical correctness, model superiority,
population performance or readiness for clinical use. It is follow-up methods
work and does not block the current JBHI revision.

## Governed locations

- Independent-reader delivery:
  `data/governed/study-runs/jbhi-medgemma-v1-evidence-development-20260916/clinical-reader-independent-v1/`
- Frozen source instrument and later blinded comparison outputs:
  `data/governed/study-runs/jbhi-medgemma-v1-evidence-development-20260916/review-source-first-v2/`
