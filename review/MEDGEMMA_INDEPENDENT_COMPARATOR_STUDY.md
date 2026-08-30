# Independent MedGemma comparator study

This document specifies a runnable, post-submission MedGemma comparison for
`JBHI-02463-2026`. It uses the inherited evaluation framework in this
repository and does not depend on receipt of another author's producing
bundle.

The machine-readable preregistration is
`review/model-receipts/medgemma-independent-comparator.preregistered.json`.
It freezes the scientific distinction that matters:

- `independent-matched-interface-q2-v1` is our independently specified,
  executable configuration;
- `v5g` is Vasily's additive example configuration and remains a separate
  pending intake node until its exact producing materials arrive; and
- neither configuration replaces the submitted Mistral source of record.

## Question and contrast

The primary question is whether the public MedGemma-27B text model, when
transported through the exact submitted classification prompt, grammar, and
four-level output semantics, changes agreement with the reference annotation
on the same reports relative to submitted and reproduced Mistral.

The primary configuration changes the model artifact and quantization while
holding the historical task interface fixed. It deliberately uses raw
completion because that is how the preserved pipeline serialized the Mistral
task. It does not apply MedGemma's embedded chat template. A native-chat
configuration could answer a different interface-adaptation question, but it
must receive a new identifier and be preregistered before its results are
seen. It cannot displace the matched-interface primary result after the fact.

The comparison is classification-only. Explanations are part of the
historical methodological lineage, but they are not required to add a
five-label performance row or bar. If MedGemma explanations are later studied,
that is a separately receipted experiment rather than an invisible expansion
of this one.

## Frozen model and runtime

The primary laptop configuration is:

- upstream release:
  [`google/medgemma-27b-text-it`](https://huggingface.co/google/medgemma-27b-text-it),
  model version `1.0.0`;
- public distribution: `unsloth/medgemma-27b-text-it-GGUF` at revision
  `334fbf6811c963d223f6ac107a459347353f068d`;
- artifact: `medgemma-27b-text-it-Q2_K.gguf`, 10,503,437,312 bytes, SHA-256
  `b137aac80f2bcb1c1ed35bfe13387bc496eb18898d5f46425687604f0f714481`;
- engine: `llama-cpp-python 0.3.34`, 4,096-token context, 30 GPU layers;
- temperature 0, top-k 40, top-p 0.95, and 256 output tokens; and
- historical prompt SHA-256
  `52198221d8330e9857b51a7ad99b017aa18836e1718b08dd0ae355820f5a5e69`
  with classification grammar SHA-256
  `5237e13988062538cda9c21906f1f4e1fc8b99498e2462ea69fe24bface35016`.

The license record points to the official
[Google HAI-DEF terms](https://developers.google.com/health-ai-developer-foundations/terms)
and the plan preserves Google's
[MedGemma technical documentation](https://developers.google.com/health-ai-developer-foundations/medgemma)
as the upstream reference. This is provenance metadata, not a legal
conclusion. Weights are cached locally and are not redistributed.

## Populations and lock

The producing Mistral run generated all 2,000 Zoe and 500 Maria reports. The
new comparator need only execute the already fixed complete-case surfaces:

| Surface | Candidate | Complete/executable | Excluded incomplete | Role |
|---|---:|---:|---:|---|
| Zoe first 100 | 100 | 100 | 0 | operational transport check |
| Zoe `[100:500, 1000:2000]` | 1,400 | 1,395 | 5 | evaluation |
| Maria `[0:500]` | 500 | 499 | 1 | evaluation |

The executable total is therefore 1,994, including the 100-report transport
check; the evaluation total is 1,894. The same 1,994 aggregate has appeared in
provisional correspondence. That equality is not evidence that the external
run used this selection and must not be used to infer its provenance.

The 100-report stage is not prompt selection. Its role is to detect transport
failure, invalid JSON, truncation, constant output, unsafe resource behavior,
or another operational defect before committing to the evaluation run. Its
reference performance may not select a prompt, template, seed, or
quantization. Any semantic change receives a new configuration identifier and
preserves the stopped attempt.

## Analysis

For each of five labels and both cohorts, retain the complete four-level and
binary-core confusion matrices, raw counts, precision, recall, specificity,
F1, core accuracy, certainty-adjusted accuracy, and four-level and binary
kappa. Retain invalid and unfinished outputs rather than removing unfavorable
records before reporting.

The paired contrasts are MedGemma minus submitted Mistral and MedGemma minus
reproduced Mistral on byte-identical report-key sets. The prespecified effects
are core-accuracy difference, certainty-adjusted-accuracy difference, and
false-negative-rate difference. Use 2,000 paired bootstrap replicates with
seed `20260718`, exact report-level McNemar sensitivity tests, and Holm
adjustment across five labels by core and four-level correctness.

The current snapshots do not contain a confirmed patient key. `Cluster code`
has only one distinct value in each evaluation snapshot and is not accepted as
a patient identifier. This does not block model inference or exact same-report
comparisons. It does mean patient-grouped intervals and patient-level splitting
remain gated. Until a stable report-to-patient mapping is confirmed, any
report-level interval must carry the within-patient non-independence
limitation.

## Readiness check

The following command performs no inference and emits no report keys:

```bash
uv run eeg-review medgemma-study-readiness \
  --plan review/model-receipts/medgemma-independent-comparator.preregistered.json \
  --source-run data/governed/study-runs/jbhi-native-20260814 \
  --receipt-dir /governed/path/to/2026-08-27-medgemma \
  --check-local \
  --output-dir /governed/path/to/medgemma-independent-readiness
```

It checks the plan against the live prompt, grammar, model registry, cached
model, private preload/runtime/classification receipts, governed database
hashes, complete-case arithmetic, and duplicate report keys. Its output
separates three statuses:

1. readiness to start governed inference;
2. readiness for patient-grouped analysis; and
3. readiness for a manuscript claim.

Only the first can be true before a run. External v5g receipt is not among its
execution blockers.

## Execution and portability

The study must checkpoint after every report and resume by exact report key.
Run products remain in governed storage and carry the clean repository commit,
model receipt, prompt/grammar hashes, input and output hashes, runtime settings,
token/time totals, invalid/unfinished counts, and transfer manifest. The same
bundle can move to Linux/NVIDIA; hardware may change, but model bytes,
interface bytes, report manifests, and output semantics may not.

Based on the single Mac classification compatibility probe, a sequential
classification-only run is approximately 22.7 hours for all 1,994 records or
21.6 hours for the 1,894 evaluation records. This is a planning estimate, not
an SLA; report length and thermal conditions will change wall time. The run can
start on the laptop, be interrupted safely, or be transferred to a larger
authorized host.

After generation, validate exact key coverage, produce a completed version-3
intake for this independent configuration, run the existing evaluation and
paired-comparison commands, and seek author-group admission before placing a
new row or bar in the manuscript. Vasily's exact v5g bundle can later enter the
same route under its own identifier.

## Post-primary native-interface sensitivity

The matched-historical primary run is now complete and remains immutable. Its
structured-output validity shows that the unfavorable comparison is not a
transport failure, while its sensitivity/specificity pattern makes interface
adaptation a scientifically plausible follow-up rather than a repair of the
primary result.

`review/model-receipts/medgemma-native-interface-sensitivity.preregistered.json`
therefore freezes a narrow sensitivity plan. It changes only raw-prompt
serialization to an exactly receipted model-native chat template and frozen
task message, initially on the 100-report Zoe development surface. Protected
Zoe and Maria evaluation remains governance-locked until a PI, data-custodian,
or approved-study record confirms that this new comparator execution is covered
by H18-02728 and its secondary-use authorization. The interface bytes are now
frozen. The singleton development candidate completed all 100 frozen keys with
valid outputs and 16 distinct full-label patterns, so the preregistered
result-blind structural rule selected it for an immutable configuration freeze.
That selection occurred before reference-label metrics were accessed. The
freeze receipt is
`review/model-receipts/medgemma-native-interface-development.freeze.json`.
The plan prohibits prompt, template, quantization, or seed search on protected
outcomes and cannot replace or suppress the completed matched-interface result.

The authorization boundary is executable rather than narrative. Populate
`review/model-receipts/medgemma-native-protected-authorization.template.json`
only from the applicable approved-study record or written PI/data-custodian
confirmation, preserve the source record and its SHA-256 in governed storage,
and run:

```bash
PYTHONPATH=src python scripts/check_medgemma_native_protected_authorization.py \
  --authorization /governed/path/authorization.json \
  --output /governed/path/protected-evaluation-unlock.json
```

The checker fails closed for pending status, authorship-only assertions,
configuration or cohort drift, missing secondary-use coverage, or relaxed
distribution controls. Passing it validates the documented technical gate; it
does not make an ethics or legal determination and does not admit results into
the manuscript.
