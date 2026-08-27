# Mistral task-adaptation work package

This protocol defines a possible fourth, post-submission evidence layer without
admitting it into the paper or computing a result. Its public-safe plan is
`model-receipts/mistral-task-adaptation.preregistered.json`; the machine-readable
shape is `model-receipts/mistral-task-adaptation-plan.schema.json`.

The working name is **schema-guided inference-time task adaptation and post-hoc
calibration**. This preserves the thesis lineage while avoiding ambiguity with
learned soft-prompt tuning. The initial route does not update Mistral weights,
does not use MedGemma as a teacher, and does not select a configuration from
Zoe or Maria evaluation results.

## Thesis lineage

The work package operationalizes existing directions in Wanying Tian's thesis:

- iterative refinement on the first 100 RA-annotated Zoe reports;
- clinical definitions and examples rather than report text as prompt-design
  material;
- grammar-constrained structured decoding;
- evidence extraction and internal label consistency;
- a model-agnostic pipeline that can transport to other LLMs; and
- the proposed lightweight post-hoc calibration layer using token evidence and
  per-category thresholds without retraining the model.

It does not retroactively change the submitted Mistral configuration. If run,
it becomes `post_submission_mistral_adapted`, derived from but distinct from
`reproduced_mistral`.

## Signal vocabulary

The plan uses four bounded signal roles:

1. `development`: human-reference outcomes that may select configuration or
   calibration parameters. For this route, that is only the historical
   first-100 Zoe RA development manifest.
2. `design_prior`: clinical definitions, annotation semantics, consistency
   rules, and methodological reporting requests. These may shape the method but
   do not supply evaluation outcomes.
3. `evaluation_only`: the frozen Zoe 1,395 and Maria 499 reference surfaces.
   Their outcomes remain uninspected until the adapter is frozen.
4. `context_only_prohibited_for_selection`: provisional or receipted MedGemma
   predictions and aggregates. They may motivate a scientific question but may
   not select this adapter, its thresholds, or its stopping point.

If MedGemma predictions ever supervise Mistral, that is a separately named
teacher-student or pseudo-label distillation experiment. It is not this work
package and it cannot reuse the same evaluation surface as an independent
generalization estimate.

## Initial adapter boundary

The preregistered route contains:

- iterative prompt engineering grounded in the historical task schema;
- the existing GBNF output constraint;
- evidence extraction;
- deterministic cross-label consistency checks; and
- per-category post-hoc calibration learned only on the development surface.

Only calibration thresholds may be fitted. Soft prompts, LoRA, full
fine-tuning, and teacher-student distillation require new identifiers, new
receipts, and an independently defensible development/evaluation boundary.

Every attempted component variant and unfavorable development result must be
retained. The stopping rule is fixed before implementation: complete the
prespecified component ablation and calibration procedure on the development
surface, then freeze. Do not continue after inspecting evaluation outcomes.

## Probability instrumentation

The historical Mistral outputs contain generated four-level decisions but no
token-probability surface. They therefore cannot be retroactively described as
probabilistically calibrated. The pipeline now has an opt-in instrument for a
new, separately governed run:

```bash
python src/LLM_pipeline/pipeline.py \
  --num-reports 100 \
  --model mistral \
  --dataset-path /governed/path/zoe_reference.db \
  --dataset-id zoe-development-calibration \
  --output-csv /governed/path/mistral-development-logprobs.csv \
  --capture-classification-logprobs
```

The historical route remains unchanged because the flag is disabled by
default. When enabled, the model is loaded with `logits_all=true` and
llama.cpp is asked for the top 64 completion-token log probabilities; both
settings are written into the run receipt. At each grammar-constrained decision
position, the pipeline requires explicit alternatives for all four levels and
records:

```text
P(core positive) = [P(level 3) + P(level 4)] / sum(P(level 1..4))
```

If even one level alternative is absent, that category's probability is
recorded as unavailable rather than renormalized over a truncated surface. The
run receipt records the feature definition and availability count for every
label. These probabilities are governed case-level outputs, not human
confidence and not calibrated values. They become calibration inputs only
through the frozen development-only procedure in this work package.

## Comparison design

The primary contrast is:

```text
reproduced_mistral vs post_submission_mistral_adapted
```

An adapter-effect interpretation is allowed only if the Mistral artifact,
quantization, runtime, report keys, and all non-adapter settings are otherwise
held fixed. Use paired same-case estimates and patient-cluster inference when
the patient key is confirmed.

The contrast:

```text
post_submission_mistral_adapted vs post_submission_medgemma
```

is an overall-configuration comparison. It cannot by itself distinguish base
weights from prompt, grammar, wrapper, calibration, or runtime. A true
model-by-adapter attribution requires transporting both frozen adapters across
both base models as a separate factorial experiment. Vasily's v5g configuration
is not silently treated as one of those cells; its exact producing bundle and
selection history must first pass governed intake.

Matching or exceeding MedGemma is a hypothesis, never the selection or stopping
rule. Null and unfavorable results remain reportable.

## Machine gate

Validate the public preregistration before implementation:

```bash
uv run eeg-review adaptation-plan-validate \
  --contract review/model-receipts/mistral-task-adaptation.preregistered.json \
  --output-dir /tmp/jbhi-mistral-adaptation-plan
```

The initial receipt may be `ready_for_implementation` but cannot be
`ready_for_evaluation`. Before evaluation, create a governed freeze receipt,
hash the complete adapter artifact, record author-group admission, change the
status to `frozen_before_evaluation`, and validate both files with
`--check-files`. The validator blocks:

- evaluation labels used for parameter or variant selection;
- inspected Zoe or Maria evaluation outcomes before freeze;
- MedGemma teacher use;
- silent LoRA, soft-prompt, or distillation additions;
- a frozen status without author admission and immutable artifacts; and
- changes to the historical development and evaluation identifiers.

`ready_for_implementation` means the signal boundary and instrumentation are
specified. It does not mean that a threshold adapter has been fitted or that a
model result exists. No development or evaluation inference was run while
adding this instrument.

The validation receipt contains no performance estimate and does not admit the
proposed fourth layer into final analysis. Completed manifests, signal
artifacts, freeze receipts, report and patient keys, and predictions remain in
authorized storage.
