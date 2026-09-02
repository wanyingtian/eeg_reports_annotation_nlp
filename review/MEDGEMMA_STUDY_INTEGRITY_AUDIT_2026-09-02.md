# MedGemma v1 study-integrity audit

**Status:** complete public-safe audit of the already-finished local comparison;
no model inference, parameter change, or new performance calculation.

## Finding

The independent native-interface MedGemma v1 comparison follows the same core
discipline as the submitted Mistral study: a named development surface, a
fixed producing configuration, and separately held-out evaluation cohorts.
The audit establishes all of the following from the actual governed artifacts:

- the only candidate configuration was committed before the 100-report
  development transport run;
- the selection rule used structural completion and output diversity, not
  reference-label performance;
- weights and training were unchanged;
- the protected execution plan was committed after its documentary
  authorization and before evaluation inference;
- configuration search and partial reference metrics were prohibited;
- the 100 development, 1,395 Zoe evaluation, and 499 Maria evaluation reports
  are pairwise disjoint by both report key and normalized report text;
- all 1,894 evaluation reports completed using the same model, prompt, grammar,
  embedded chat template, task-message template, and sampling contract; and
- all 82 files named by the final transfer manifest still match their hashes.

Inference used a local `llama.cpp` process and an already-cached pinned model
artifact. Remote model lookup, remote inference, and report or prediction
egress were disabled.

## What this closes

This closes the concern that the reported native-interface result might be a
configuration tuned against the 1,894 held-out outcomes. It was not. Later v2
and v2.1 experiments used the 100-report development surface only and were not
promoted into a protected-cohort rerun.

The comparison remains a **configured-system comparison**. Model family,
quantization, and input serialization differ from historical Mistral, so it
cannot identify a pure base-weight effect. That is a transparent system-level
comparison, not a defect in the evaluation.

## What this does not close

- Report separation does not prove patient independence; no validated patient
  key is present in the transferred snapshot.
- Reference agreement is not prospective clinical validation.
- The integrity receipt does not itself admit the result to the manuscript.
- Vasily's v5g stream remains a separate configuration pending exact intake.

The public-safe machine record is
`review/model-receipts/medgemma-v1-study-integrity.record.json`. The governed
aggregate is
`data/governed/analysis-runs/jbhi-medgemma-v1-study-integrity-20260902/aggregate-study-integrity.json`
(SHA-256 `2e58aa005b0166d36fb8331de40b2ddeb79f83434ead8fc4d53ba459edc4e6d0`).
It contains no report keys, text, labels, or predictions.
