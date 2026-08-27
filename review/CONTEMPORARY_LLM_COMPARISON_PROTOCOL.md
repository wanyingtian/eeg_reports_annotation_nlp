# Contemporary LLM comparison protocol

This protocol governs any post-submission LLM comparator proposed for
`JBHI-02463-2026`. It preserves the submitted Mistral-7B outputs as the source
of record and evaluates a contemporary model as a separately named experiment.

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
   boundary for exporting aggregate receipts.

Use `review/model-receipts/contemporary-llm-intake.template.json` as the
transfer checklist. Null fields mean the comparator remains unreceipted.

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
  --predictions /governed/path/contemporary_zoe.csv \
  --reference-range 100:500 \
  --reference-range 1000:2000 \
  --require-complete-reference \
  --cluster-column Hashed_PatientURN \
  --output-dir outputs/review/zoe-contemporary

uv run eeg-review compare \
  --reference /governed/path/zoe_reference.db \
  --predictions-a /governed/path/contemporary_zoe.csv \
  --predictions-b /governed/path/mistral_zoe.csv \
  --model-a-id contemporary-locked \
  --model-b-id mistral-7b-submitted \
  --reference-range 100:500 \
  --reference-range 1000:2000 \
  --require-complete-reference \
  --cluster-column Hashed_PatientURN \
  --output-dir outputs/review/zoe-contemporary-vs-mistral
```

Repeat on Maria with `--reference-range 0:500`. Omit the cluster column only
when no authoritative patient key exists; the receipt will then retain the
report-level limitation. Use the same selected report keys for the comparator,
Mistral, LD, and SG surfaces.

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

## Claim boundaries

- Inter-annotator agreement is not a human ceiling or diagnostic ground truth.
- A separate report author is not automatically an independent site or patient
  population.
- Better agreement with one annotator does not establish clinical reasoning.
- On a largely unlabeled corpus, model agreement and positive-call rates are
  descriptive; disagreement cannot identify which model is clinically right.
- Grammar-enforced consistency is a decoding constraint. Report its effect
  separately from prompt content and base-model choice.
- The contemporary result may strengthen the revision only after its producing
  bundle passes this contract and the authors agree on its role.
