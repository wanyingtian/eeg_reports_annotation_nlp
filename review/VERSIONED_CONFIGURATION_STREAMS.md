# Versioned producing-configuration streams

## Purpose

A model-family name is not a complete experiment. Each runnable variant enters
the JBHI framework as a technically named configuration stream whose operating
parameters can be reviewed, reproduced, and compared without turning a
contributor's name into the model label.

The stream name identifies the configuration. Contribution and custody are
recorded separately through the typed provenance roles `originator`,
`contributor`, `maintainer`, `received_from`, `custodian`, and
`scientific_governance`. Integration does not transfer ownership and does not
erase contribution.

## v5g stream

The separately reported v5g configuration has the stable technical root:

```text
jbhi-02463/comparator/medgemma-27b-text-it/configuration/v5g
```

Until its exact producing bundle is received and validated, its lifecycle state
remains `external_pending_intake`. That status means the stream has a prepared
route into the shared framework; it does not characterize or reject its
reported result.

The existing public-safe template is
`review/model-receipts/contemporary-llm-intake.template.json`. A completed
stream receipt resolves, at minimum:

- upstream model revision, distribution artifact, checksum, and quantization;
- runtime, hardware, context length, and sampling/decoding parameters;
- input serialization or chat template;
- task prompt, grammar, wrapper, and output schema;
- prompt/configuration selection history;
- cohort manifest identity, inclusion arithmetic, reference, and endpoint;
- run and aggregate-result receipts; and
- bounded provenance roles and distribution status.

## Git and governed-storage boundary

The shared revision branch can version non-sensitive operating material:

- configuration manifests and parameter receipts;
- publishable prompt, grammar, and wrapper sources or their hashes;
- model/runtime identifiers and immutable revisions;
- selection-history and validation receipts without report keys; and
- aggregate-result receipts approved for the collaboration surface.

Model weights are referenced, not redistributed. Report and patient keys, raw
reports, keyed predictions, and case-level review material remain in authorized
storage. Their Git-visible receipts contain only approved aggregate counts and
digests.

## Versioning rule

A semantic or operating change receives a new stream revision or descendant;
it does not silently overwrite `v5g`. Runs and results use descendants such as:

```text
.../run/<receipt-id>
.../result/<aggregate-receipt-id>
```

This is the same rule applied to submitted Mistral, reproduced Mistral, and the
independently specified MedGemma configurations: technical labels identify
experiments, while provenance metadata records who contributed what.
