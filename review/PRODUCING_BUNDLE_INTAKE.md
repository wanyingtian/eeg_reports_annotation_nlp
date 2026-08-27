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
  quantization, license label, authoritative terms URL, and a non-legal
  provenance notice;
- inference engine version/revision, hardware, sampling parameters, and the
  exact chat-template mode and artifact when one was applied;
- prompt, grammar, and prompt-selection-history artifacts and checksums;
- report-key and patient-key namespaces and exact-string key semantics;
- cohort source, candidate, included, excluded, reference-complete, and
  prediction-expected arithmetic; and
- the keyed prediction surface, canonical label mapping, invalid count, and
  unfinished count for every cohort.

## Multi-parent provenance DAG

Version 3 embeds a typed ancestry graph in the existing intake contract. It
does not create another canonical repository and it does not redistribute a
model artifact. The graph uses hierarchical technical identifiers rooted at:

```text
jbhi-02463/comparator/medgemma-27b-text-it/configuration/v5g
```

Run and integration descendants use identifiers such as:

```text
.../integration/jbhi-revision-toolchain@<commit>
.../evaluation/<cohort-manifest-hash>
.../run/<receipt-id>
.../result/<aggregate-receipt-id>
```

The integrated configuration has four independently receipted scientific
parent types:

1. `upstream_weights_quantization`: Google's MedGemma release and the exact
   Unsloth GGUF artifact, revision, quantization, byte size, and SHA-256;
2. `producing_configuration`: the external v5g source bundle, prompt, grammar,
   wrapper, selection history, predictions, and manifests received through
   governed intake;
3. `inherited_evaluation_framework`: Wanying Tian's repository revision, the
   submitted Mistral study, and its historical cohort and metric semantics;
4. `integration`: Steven Bergner's independent reproduction and the exact
   comparator-intake branch revision used to admit the bundle.

No ambiguous `owner` field is accepted. Assertions instead use the bounded
roles `originator`, `contributor`, `custodian`, `maintainer`, `received_from`,
and `scientific_governance`. Each assertion records its scope, confirmation
state, revision, and optional evidence-artifact hash. Pending or unconfirmed
states remain explicit; a frozen contract cannot silently omit a parent
revision or every role assertion for a root parent. Frozen node and role
revisions must be immutable 40- or 64-hex digests. The v5g template records
Vasily Vakorin as the pending originator and transfer source until the exact
producing bundle confirms that assertion.

The admission state is monotonic and machine-checked:

```text
external_pending_intake
  -> external_receipted
  -> validated_pending_author_admission
  -> author_group_admitted
  -> integrated
```

Before exact receipt, validation, and author-group admission, v5g remains an
external producing bundle. After admission it becomes an integrated
configuration node of the shared harness. Integration records ancestry; the
contract explicitly states that it does not transfer ownership.

Distribution states are recorded separately: ancestry receipts may be
`receipt_only`, non-sensitive source/configuration may be
`source_config_publishable`, keyed outputs are `outputs_governed`, and model
weights are `weights_not_redistributed`. Completed contracts, report and
patient keys, manifests, and keyed predictions remain in authorized storage.

The existing license field is preserved. MedGemma receipts also reference the
official [Google HAI-DEF Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).
This is provenance metadata only; the contract makes no legal conclusion.

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
ranking. It carries each layer's provenance graph/root and admission state
into the aggregate readiness receipt, then reports whether each pair has:

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
- exact Google upstream and Unsloth distribution revisions, artifact hash,
  ancestry roles, and admission receipt;
- runtime/build receipt and exact chat template;
- exact prompt, grammar, and wrapper behavior;
- prompt-development history and the status of each attempted variant;
- immutable cohort manifests and reconciliation of every population count;
- keyed predictions with invalid and unfinished rows retained; and
- Chris's confirmation of the submitted selection and processing lineage.

Until these arrive, Q2 work remains an interface/transport check. Do not infer
the producing `v5g` configuration, run a replacement evaluation surface, or
publish a new result artifact.
