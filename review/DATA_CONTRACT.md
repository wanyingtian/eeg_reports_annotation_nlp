# Governed revision data contract

This contract describes the minimum inputs accepted by the revision commands.
It does not authorize data movement or reinterpret an existing identifier.

| Field | Required for | Contract |
|---|---|---|
| Report key | all alignment | Complete and unique within a snapshot; pseudonymous; not assumed to identify a patient |
| Patient key | grouped folds, cluster CIs, patient overlap | Stable across relevant snapshots, with custodian-confirmed linkage semantics |
| Report text | inference/baselines | Exact submitted content or a separately named section variant; never emitted by review commands |
| Cohort/split | provenance | Development, Zoe evaluation, or Maria evaluation; immutable membership |
| Five category labels | audit/evaluation | Integer levels 1--4; annotator and missingness explicitly identified |
| Demographics/setting | cohort description | Approved aggregate fields with small-cell release policy |

## Required snapshot receipt

For each input, record the file checksum, extraction date, source system,
eligibility/exclusion rules, sampling unit/method/seed, study period, sites,
report and unique-patient counts, and responsible custodian. Cross-cohort
patient comparisons are invalid if keys were salted or generated differently.

## Output classes

- Aggregate JSON/CSV may leave the governed environment only after the local
  release process approves it.
- OOF predictions and clinical error-review packets are case-level even when
  they contain only pseudonymous keys. They remain governed.
- Report text, direct identifiers, credentials, databases, and model caches
  must never be committed to this repository.
