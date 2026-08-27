# Mistral task-adaptation work package

This protocol defines a possible fourth, post-submission evidence layer without
admitting it into the paper or computing a result. Its public-safe plan is
`model-receipts/mistral-task-adaptation.preregistered.json`; the machine-readable
shape is `model-receipts/mistral-task-adaptation-plan.schema.json`.

The working name is **binary-core schema-guided task adaptation and post-hoc
certainty mapping**. This preserves the thesis lineage while avoiding two
ambiguities: a generated four-level label is not itself a calibrated
probability, and this route is not learned soft-prompt tuning. The route does
not update Mistral weights, does not use MedGemma as a teacher, and does not
select a configuration from Zoe or Maria evaluation results.

## Thesis lineage

The work package operationalizes existing directions in Wanying Tian's thesis:

- iterative refinement on the first 100 RA-annotated Zoe reports;
- clinical definitions and examples rather than report text as prompt-design
  material;
- grammar-constrained structured decoding;
- evidence extraction and internal label consistency;
- a model-agnostic pipeline that can transport to other LLMs; and
- the proposed lightweight post-hoc mapping from binary token evidence to the
  four certainty levels using per-category thresholds without retraining the
  model.

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
- an explicit binary core-decision interface using the same schema definitions
  and consistency constraints;
- per-category post-hoc certainty margins learned only on the development
  surface.

Only certainty threshold margins may be fitted. Soft prompts, LoRA, full
fine-tuning, and teacher-student distillation require new identifiers, new
receipts, and an independently defensible development/evaluation boundary.

Every attempted component variant and unfavorable development result must be
retained. The stopping rule is fixed before implementation: complete the
prespecified component ablation and certainty-mapping procedure on the development
surface, then freeze. Do not continue after inspecting evaluation outcomes.

The component sequence is fixed before a development run:

1. `reproduced_mistral_historical_four_level`: the exact reproduced historical
   prompt, four-level grammar, and output semantics;
2. `binary_core_historical_margin`: the mechanically narrowed binary 1/4
   interface and the thesis's historical symmetric margin of 0.1; and
3. `binary_core_fitted_per_category_margin`: the same binary inference surface
   with one margin selected per category from the fixed grid 0.1, 0.2, 0.3.

This sequence separates the binary language-interface effect from the
development-fitted margin effect. No open-ended prompt search is part of this
route. Any clinically substantive change to definitions, examples, or
uncertainty anchors must become a newly identified prompt artifact and be
admitted before its outcomes are inspected.

## Probability instrumentation

The historical Mistral outputs contain generated four-level decisions but no
token-probability surface. They therefore cannot be retroactively described as
probabilistically calibrated. Directly normalizing the four generated level
tokens also fails to cleanly separate the core decision from its certainty.
The pipeline therefore has an opt-in binary-core mode for a new, separately
governed run:

```bash
python src/LLM_pipeline/pipeline.py \
  --num-reports 100 \
  --model mistral \
  --dataset-path /governed/path/zoe_reference.db \
  --dataset-id zoe-development-calibration \
  --output-csv /governed/path/mistral-development-logprobs.csv \
  --classification-mode binary_core_certainty_adapter
```

The historical route remains unchanged because
`historical_four_level` is the default. Binary mode mechanically retains the
historical definitions, examples, JSON keys, and cross-label constraints, but
narrows each decision to `1 = core absent` or `4 = core present`. It uses a
separate GBNF artifact and writes a mode marker on every resumable prediction
row. The model is loaded with `logits_all=true` and llama.cpp is asked for the
top 64 completion-token log probabilities; both settings are written into the
run receipt. At each grammar-constrained decision position, the pipeline
requires both explicit binary alternatives and records:

```text
P(core positive) = P(token 4) / [P(token 1) + P(token 4)]
```

If either binary alternative is absent, that category's probability is
recorded as unavailable rather than renormalized over a truncated surface. The
run receipt records the feature definition and availability count for every
label. These probabilities are governed case-level outputs, not human
confidence and not calibrated values. They become certainty-mapping inputs only
through the development-only procedure in this work package.

## Fixed certainty mapping and confidence machinery

The binary decision boundary remains 0.5. For a selected symmetric margin
`m`, the mapping is:

```text
p < 0.5-m       -> 1, confident no
0.5-m <= p < .5 -> 2, low-confidence no
.5 <= p < .5+m  -> 3, low-confidence yes
p >= .5+m       -> 4, confident yes
```

For each category, select `m` from 0.1, 0.2, or 0.3 by maximizing exact
four-level agreement on the fixed first-100 Zoe RA development manifest. Ties
go to the margin closest to the historical 0.1 choice and then to the smaller
margin. The binary core decision therefore cannot be improved by moving its
boundary during this procedure.

At least 80 valid probability/reference pairs and at least five pairs on each
side of the core boundary are required for category-specific fitting. If that
support is absent, retain margin 0.1 and mark the category as not fitted. Do
not pool a rare category into a different clinical label or use MedGemma
outputs to overcome sparse support.

The fitter retains every candidate margin and unfavorable development score.
It reports a leave-one-report-out selection diagnostic, a 2,000-replicate
bootstrap distribution of threshold-selection stability stratified by RA core
side, and descriptive 95% Wilson intervals. These quantities describe the
small development surface; none is an independent generalization interval or
patient-cluster estimate.

The fitter requires the exact 100-key governed manifest, the binary prediction
CSV, and its producing run receipt. Checksums, the binary mode marker, model,
prompt, grammar, and exact prediction-key coverage must all agree:

```bash
uv run eeg-review certainty-adapter-fit \
  --contract /governed/path/mistral-task-adaptation.execution.json \
  --reference /governed/path/zoe-ra.db \
  --predictions /governed/path/mistral-development-logprobs.csv \
  --prediction-run-receipt /governed/path/mistral-development-logprobs.run.json \
  --development-manifest /governed/path/zoe-development-first-100.csv \
  --output-dir /governed/path/mistral-certainty-adapter \
  --acknowledge-governed-inputs
```

The public preregistration deliberately leaves the governed manifest identity
null. Before fitting, make an execution copy in authorized storage, declare
the manifest path and SHA-256, and validate that copy. The command emits only
aggregate development diagnostics, the thresholds, hashes, and receipts; no
report key or case-level prediction is copied out.

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

`ready_for_implementation` means the signal boundary, binary language
interface, component sequence, mapping objective, fallback, and stability
diagnostics are specified. It does not mean that a threshold adapter has been
fitted or that a model result exists. No development or evaluation inference
was run while adding this instrument.

The validation receipt contains no performance estimate and does not admit the
proposed fourth layer into final analysis. Completed manifests, signal
artifacts, freeze receipts, report and patient keys, and predictions remain in
authorized storage.
