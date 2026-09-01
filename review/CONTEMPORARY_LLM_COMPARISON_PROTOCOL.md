# Contemporary LLM comparison protocol

This protocol governs any post-submission LLM comparator proposed for
`JBHI-02463-2026`. It preserves the submitted Mistral-7B outputs as the source
of record and evaluates a contemporary model as a separately named experiment.

The companion [`EVALUATION_SURFACE_FRAMEWORK.md`](EVALUATION_SURFACE_FRAMEWORK.md)
defines the model, interface, prompt, grammar, quantization, population,
reference, selection, and metric axes that make a comparison identifiable. Its
typed public-safe registry distinguishes one-factor ablations, model-native
task comparisons, complete configured-system comparisons, cohort
stratifications, and unlabeled descriptive analyses. This prevents a model
family name from silently standing in for a whole producing configuration.

Rendered summaries are useful for scientific discussion but do not constitute
a reproducible receipt. Do not commit private author correspondence, governed
prediction rows, report text, patient keys, model weights, or unpublished
figures to this repository.

## Required intake

Before a comparator can support a manuscript claim, preserve:

1. exact upstream model repository, revision, GGUF filename, byte size,
   SHA-256, quantization, and license;
2. runtime and hardware receipt, including llama.cpp revision/build, chat
   template, context size, truncation, GPU layers, sampling parameters, seed,
   maximum output, and token/time totals;
3. byte-identical prompt and grammar files with hashes;
4. a prompt-development ledger naming the hypothesis, development population,
   outcomes inspected, stopping rule, and decision for every attempted variant;
5. immutable report manifests for development and each evaluation cohort,
   with candidate, excluded, and complete-case counts;
6. one unique pseudonymous report key and five four-level outputs per model,
   plus every invalid, missing, and unfinished result; and
7. LD/SG roles, stable patient-key semantics when available, and the approved
   boundary for exporting aggregate receipts; and
8. the multi-parent provenance DAG: Google/Unsloth weights, the external
   producing configuration, the inherited submitted-study framework, and the
   exact integration revision, with bounded role assertions and distribution
   states.

Use `review/model-receipts/contemporary-llm-intake.template.json` as the
transfer checklist. Its version-3 structure has a JSON Schema and a typed
aggregate-only validator; see
`review/PRODUCING_BUNDLE_INTAKE.md`. Null fields mean the comparator remains
unreceipted.

The branch includes candidate Q2_K and Q4_K_S registry entries pinned to a
public Unsloth GGUF revision. They are preload candidates, not proof of
Vasily's producing artifacts. After accepting the upstream Google HAI-DEF
terms, preload Q2_K without allocating it in llama.cpp:

```bash
make preload-model \
  MODEL=medgemma-27b-q2-candidate \
  RECEIPT=/private/path/medgemma-q2-preload-receipt.json
```

The Hugging Face cache resumes partial downloads. The command verifies the
10.5 GB file against the pinned SHA-256 before writing the receipt. Compare
that hash with Vasily's producing receipt when it arrives.

After preloading, verify that the local llama.cpp build can load the artifact,
apply its embedded chat template, and obey a trivial GBNF constraint:

```bash
make smoke-model \
  MODEL=medgemma-27b-q2-candidate \
  RECEIPT=/private/path/medgemma-q2-runtime-smoke-receipt.json
```

This is a runtime preflight, not an EEG-result reproduction. It does not by
itself authorize a cohort run.

For a bounded compatibility check against the preserved submitted
classification prompt and grammar, run one sample report without the separate
explanation stage:

```bash
make smoke-classification \
  MODEL=medgemma-27b-q2-candidate \
  RECEIPT=/private/path/medgemma-q2-classification-smoke-receipt.json
```

The private receipt hashes, rather than copies, the report text and source
identifier. This raw-completion route deliberately records that no embedded
chat template was applied. It verifies only that the candidate can execute the
historical classification interface; it is not v5g and is not a study result.

The smoke completion cannot be promoted into a result and cannot identify an
external producing configuration. It may, however, support a separately
preregistered independent comparator. The frozen independent study is defined
in `MEDGEMMA_INDEPENDENT_COMPARATOR_STUDY.md` and
`model-receipts/medgemma-independent-comparator.preregistered.json`. Vasily's
v5g remains a distinct example configuration and does not block that
independent study.

## Development and evaluation lock

A prompt or grammar selected after inspecting reference performance on a
cohort has been tuned on that cohort. Its score is exploratory selection
performance even if the base model weights were not fine-tuned.

Choose one defensible route and record it before final analysis:

- lock the prompt using the 100-report historical development set, then run it
  once on the untouched Zoe and Maria evaluation populations;
- use nested prompt selection/evaluation if sufficient independent data exist;
  or
- report all prompt results as an exploratory ablation without calling the
  selected maximum an independent generalization estimate.

A held-out second annotator can test sensitivity to the reference surface, but
does not repair reuse of the same report outcomes for prompt selection and
final evaluation.

## Same-case analysis

Run each locked comparator through the existing aggregate workflow:

```bash
uv run eeg-review evaluate \
  --reference /governed/path/zoe_reference.db \
  --predictions /governed/path/contemporary_zoe_selected.csv \
  --reference-range 100:500 \
  --reference-range 1000:2000 \
  --require-complete-reference \
  --require-exact-key-set \
  --require-patient-grouping \
  --cluster-column Hashed_PatientURN \
  --output-dir outputs/review/zoe-contemporary

uv run eeg-review compare \
  --reference /governed/path/zoe_reference.db \
  --predictions-a /governed/path/contemporary_zoe_selected.csv \
  --predictions-b /governed/path/mistral_zoe_selected.csv \
  --model-a-id contemporary-locked \
  --model-b-id mistral-7b-submitted \
  --reference-range 100:500 \
  --reference-range 1000:2000 \
  --require-complete-reference \
  --require-exact-key-set \
  --require-patient-grouping \
  --cluster-column Hashed_PatientURN \
  --output-dir outputs/review/zoe-contemporary-vs-mistral
```

Repeat on Maria with `--reference-range 0:500`. Omit the cluster column only
when no authoritative patient key exists; the receipt will then retain the
report-level limitation. Use the same selected report keys for the comparator,
Mistral, LD, and SG surfaces. The `_selected` files above are governed derived
surfaces containing exactly the intake manifest's included report keys; do not
point the strict command at a larger all-reports prediction export.

Before either command, run `eeg-review comparison-readiness` across the three
preregistered evidence layers. That command is a gate only and deliberately
produces no performance result.

## Required displays

- candidate, excluded, matched, unmatched, invalid, and unfinished counts;
- four-level and binary confusion matrices for every label and cohort;
- raw TP, FP, TN, FN, support, precision, recall, specificity, F1, accuracy,
  and certainty-adjusted accuracy;
- 95% intervals at the patient-cluster level when possible;
- paired model differences, discordant counts, exact McNemar sensitivity
  analyses, and the declared multiplicity family;
- all prompt variants and quantizations, including null and unfavorable
  results; and
- inference time, prompt/completion tokens, model size, and hardware.

The external prompt-version, consistency-grammar, quantization, population,
reference, and endpoint categories can be registered before exact intake as a
design family. This preserves their useful structure without importing summary
scores. In particular, an externally named exact-level F1 remains formula
pending until its averaging rule is received; it is not silently treated as
the native evaluator's exact four-level accuracy.

## Claim boundaries

- Inter-annotator agreement is not a human ceiling or diagnostic ground truth.
- A separate report author is not automatically an independent site or patient
  population.
- Better agreement with one annotator does not establish clinical reasoning.
- On a largely unlabeled corpus, model agreement and positive-call rates are
  descriptive; disagreement cannot identify which model is clinically right.
- Grammar-enforced consistency is a decoding constraint. Report its effect
  separately from prompt content and base-model choice.
- Forcing a new model through a historical serialization and giving only the
  new model a tuned native configuration are opposite asymmetries. Preserve
  both as named surfaces; use a controlled interface ablation when asking what
  serialization changed and a configured-system comparison when several axes
  differ.
- An independently preregistered contemporary result may be generated under
  its own producing contract. Any external result, including v5g, requires its
  own exact producing-bundle intake. Manuscript use of either still requires
  validation and author agreement.

An adapted-Mistral comparison is governed by
`MISTRAL_TASK_ADAPTATION_PROTOCOL.md`. Until both frozen adapters have been
transported across both base models, adapted Mistral versus v5g MedGemma is an
overall-configuration contrast; it cannot isolate the effect of base weights
from task adaptation.
