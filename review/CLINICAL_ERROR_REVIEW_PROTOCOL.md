# Governed clinical error-review protocol

Status: operational packet schema; clinical/team approval required before case
review. This document does not authorize data access or establish a clinical
adjudication rule.

## Purpose and boundary

The reviewer requested analysis of false negatives, false positives, clinical
consequences, and workflow implications. `eeg-review error-review` creates a
deterministic, stratified worksheet for that discussion. “False negative” and
“false positive” mean disagreement with the declared Reference Annotator after
collapsing levels 1–2 versus 3–4. They do not mean that the reference is an
adjudicated clinical ground truth.

The portable worksheet contains no report text, original report identifier, or
patient identifier. It remains governed case-level material and must not be
committed, emailed, or moved outside the approved environment. An authorized
review interface may resolve each case handle back to its source report only
inside that environment.

## Sampling receipt

For each model, cohort, category, and error direction, record:

- source and prediction file hashes;
- immutable cohort ranges and complete-case rule;
- model identifier and prediction-column mapping;
- sampling seed and maximum cases per stratum;
- whether sampling used a stable patient key; and
- total eligible and selected cases.

When a patient key exists, the generator first prefers one report per patient
within each stratum. Without that key, report-level selection is provisional
and may overrepresent patients with repeated reports.

## Review roles

- The data custodian confirms access, patient-key semantics, and the approved
  review environment.
- The clinical lead defines the acceptable reviewer qualifications and whether
  one or more reviewers are required.
- The methods team supplies the frozen case packet and aggregate receipt but
  does not pre-classify clinical consequence.
- The author team decides how aggregate findings change the manuscript and
  response letter, retaining null and unfavorable results.

## Worksheet fields

The generated worksheet pre-populates only the case handle, category, model,
RA-relative error direction, and the two four-level values. Authorized
reviewers complete controlled fields for:

- review status;
- clinical salience;
- reference ambiguity;
- likely workflow consequence;
- whether an escalation or override mechanism would catch the case;
- reviewer role; and
- bounded review notes.

Before use, the clinical lead should define a small codebook for each field,
including an explicit “uncertain/not assessable” value. Free text should be
minimal and must not reproduce patient details.

## Analysis and reporting

Aggregate the completed worksheet by cohort, category, error direction,
clinical-salience code, and workflow-consequence code. Always report the number
eligible, sampled, reviewed, unresolved, and excluded. If more than one
clinical reviewer participates, preserve independent ratings and define the
adjudication process before inspecting disagreement results.

Do not generalize sampled proportions to all errors unless the final sampling
design and dependence structure support that inference. Do not use the packet
to claim safety, clinical equivalence, or deployment readiness. Its immediate
purpose is bounded error characterization and workflow-risk discussion.

## Current provisional exercise

The command was exercised against the recovered Mistral outputs and RA-selected
historical cohorts. The Zoe packet contains 194 selected label-case rows and the
Maria packet 145 under a cap of 25 cases per category and error direction. No
patient key was available, so these are report-level preparation artifacts, not
approved clinical review samples. They remain under ignored governed storage.
