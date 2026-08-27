# Producing-bundle intake and comparison gates

This workflow admits a model run into the JBHI revision without treating an
unreceipted aggregate summary as a study result. The public branch contains
the contract, validators, and analysis plan. Completed contracts, cohort
manifests, report and patient keys, and prediction rows remain in authorized
storage.

## Three fixed evidence layers

The layer names and intended comparisons are preregistered in
`review/model-receipts/evidence-layers.preregistered.json`:

1. `submitted_mistral` is the immutable submitted source of record.
2. `reproduced_mistral` is post-submission reproduction and runtime-sensitivity
   evidence; it does not overwrite the submission.
3. `post_submission_medgemma` is a separately named contemporary comparator
   that remains blocked until its exact producing bundle is validated.

The received difference between provisional populations of 1,994 and 2,493
is a diagnostic clue only. Its arithmetic difference is 499, but the contract
does not assign that difference to Maria, an author, a selection rule, v5g, or
any other cohort definition. Only the producing manifests and selection
history may resolve it.

## Typed contract

Copy `review/model-receipts/contemporary-llm-intake.template.json` into the
authorized bundle and complete one contract per evidence layer. The JSON shape
is described by `review/model-receipts/comparator-intake.schema.json` and is
parsed into typed Python records by `eeg_review.intake`.

The contract fixes:

- upstream repository, revision, model artifact checksum, byte size,
  quantization, and license;
- inference engine version/revision, hardware, sampling parameters, and the
  exact chat-template mode and artifact when one was applied;
- prompt, grammar, and prompt-selection-history artifacts and checksums;
- report-key and patient-key namespaces and exact-string key semantics;
- cohort source, candidate, included, excluded, reference-complete, and
  prediction-expected arithmetic; and
- the keyed prediction surface, canonical label mapping, invalid count, and
  unfinished count for every cohort.

`invalid_records` and `unfinished_records` are disjoint. Their sum must equal
the number of keyed rows that do not contain all five valid four-level labels;
those unfavorable rows remain on the surface instead of disappearing before
validation.

Null template fields are intentional blockers, not wildcards. The status must
be `frozen` or `source_of_record` before analysis; `draft` and
`template_unreceipted` are blockers. A grammar mode
of `none` must still have a stated purpose. A raw-completion transport check
must identify the chat template as not applied; it cannot be promoted into an
evaluation surface merely because it produced syntactically valid output.

## Validation inside authorized storage

Structural validation does not open the governed artifacts:

```bash
uv run eeg-review intake-validate \
  --contract /authorized/bundle/post-submission-medgemma-intake.json \
  --output-dir /authorized/receipts/medgemma-intake
```

The full gate verifies every declared artifact checksum, cohort arithmetic,
manifest and prediction row counts, missing and duplicate report keys, exact
manifest-to-prediction membership, and patient-key completeness:

```bash
uv run eeg-review intake-validate \
  --contract /authorized/bundle/post-submission-medgemma-intake.json \
  --bundle-root /authorized/bundle \
  --check-files \
  --output-dir /authorized/receipts/medgemma-intake
```

`intake_validation.json` contains aggregate counts and key-set digests only.
It never emits report or patient keys. `ready_for_analysis` is true only after
file-level checks pass; merely filling the JSON is not sufficient.

## Three-layer readiness without evaluation

Once all three producing contracts exist, check the preregistered pairings:

```bash
uv run eeg-review comparison-readiness \
  --intake submitted_mistral=/authorized/bundle/submitted-mistral-intake.json \
  --intake reproduced_mistral=/authorized/bundle/reproduced-mistral-intake.json \
  --intake post_submission_medgemma=/authorized/bundle/medgemma-intake.json \
  --bundle-root /authorized/bundle \
  --output-dir /authorized/receipts/comparison-readiness
```

This command computes no accuracy, agreement, confidence interval, or model
ranking. It reports whether each pair has:

- a valid producing bundle in both layers;
- the same cohort and report-key namespace;
- an exactly identical report-key set;
- the same canonical label surface; and
- confirmed, complete patient keys in the same namespace for patient-grouped
  inference, with an identical report-to-patient mapping digest.

Only after the authors confirm the historical selection surface and every
gate passes should the existing `evaluate` and `compare` commands run. Use
both `--require-exact-key-set` and `--require-patient-grouping` when the
confirmed patient key exists. These switches prevent an unnoticed inner join
or report-level bootstrap from becoming the primary analysis.

## Current decision gates

- exact MedGemma producing model artifact and quantization;
- runtime/build receipt and exact chat template;
- exact prompt, grammar, and wrapper behavior;
- prompt-development history and the status of each attempted variant;
- immutable cohort manifests and reconciliation of every population count;
- keyed predictions with invalid and unfinished rows retained; and
- Chris's confirmation of the submitted selection and processing lineage.

Until these arrive, Q2 work remains an interface/transport check. Do not infer
the producing `v5g` configuration, run a replacement evaluation surface, or
publish a new result artifact.
