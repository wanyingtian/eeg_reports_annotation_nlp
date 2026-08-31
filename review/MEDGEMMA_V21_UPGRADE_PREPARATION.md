# MedGemma v2.1: prepared development upgrade, not yet run

This is a separately named, Steven-led post-submission configuration. It does
not overwrite the completed v2, replace the full-cohort native v1 comparison,
renumber Chris's historical prompt versions, or reproduce external v5g.
Preparation was requested on August 31, 2026. Preparation does not start a job.

## Narrow classification change

The new `medgemma-native-category-scope-v2.1` prompt adds three category-scope
checks to the exact v2 prompt:

1. Interpret negation within its category: absence of epileptiform abnormalities
   alone does not establish overall normality or negate non-epileptiform changes.
2. Inspect generalized non-epileptiform findings independently, retaining the
   report's normal-state, activation, artifact and history qualifications.
3. Re-read category evidence when the five answers conflict; do not mechanically
   flip overall abnormality to conceal a missed subtype.

The actual wording is in `src/eeg_review/prompt_versions.py`. Existing focal
clarifications, definitions, examples, four-level meanings, confidence
instructions, native chat template, classification grammar, quantized model,
sampling and Metal runtime are unchanged. No case text, report key, expected
label, rare-case exception or keyword ban is inserted into the prompt.

## Independent evidence audit

For the original first 20 development positions, a separate fresh call asks for
present, absent and qualifying source passages for each category. Each list is
capped at two quotes; empty lists are explicit. It uses the original category
definitions and a new fixed JSON grammar with correct string escaping.

The audit is generated after the classification stage for operational simplicity,
but **no classification or RA label enters its model message**. There is no prior
conversation context. Predictions are stored only as an exact-key linkage for
subsequent analysis. The audit cannot feed back into this run's classification
or select its prompt. This isolates classification-prompt effects from the
changed explanation question.

This audit differs from v2's decision-conditioned explanations. Compare source
traceability and review usefulness descriptively, not as an interchangeable
thesis alignment score or causal explanation. Model-assigned evidence roles
are review suggestions, not clinical adjudication. The all-empty response is
retained as missing evidence, not successful factuality or a normal decision.

## Frozen development comparison

One candidate; exactly 100 classifications and 20 evidence calls. The same
first-100 Zoe development DB/manifest, v1 outputs and v2 outputs are hash-bound
in `model-receipts/medgemma-native-scope-v21.development-plan.json`.

The new descriptive rule is explicitly different from the earlier focal-only
v2 rule. It requires fewer total category errors than v2, a reduction in false
negatives for generalized non-epileptiform activity or overall abnormality, no
category with more binary errors than either parent, and no increase in focal
or generalized epileptiform false negatives against either parent. All five
labels and both parent comparisons remain reported even if the rule fails.

The analysis reuses the existing metrics and paired report-bootstrap machinery
(2,000 resamples, existing seed, Holm correction separately within each parent
comparison). Intervals are descriptive development intervals, not patient-grouped
or independent-test inference. Repeated development use and earlier protected
results informed the hypothesis; a successful rule is not evidence of population
superiority and does not automatically launch protected-cohort evaluation.

## Existing runner, distinct output bundle

`scripts/medgemma_prompt_v2.py --variant v21` reuses the original pipeline,
checkpointed supervisor, pause/resume, local-only model resolution, model/input
receipts and scientific manifest. Default `--variant v2` retains the prior
protocol; completed worktrees remain untouched. Outputs use `v21.csv`, not
`v2.csv`, in a new governed run directory.

```bash
python scripts/medgemma_prompt_v2.py prepare --variant v21 \
  --source-run /governed/path/to/completed-v2 \
  --run-dir /governed/path/to/new-v21 \
  --public-status /private/path/to/v21-status.json
python scripts/medgemma_prompt_v2.py dry-run --variant v21 \
  --run-dir /governed/path/to/new-v21
```

Use the existing pinned environment without syncing/upgrading dependencies.
The dry-run hashes the local model, checks the frozen source/configuration,
and does not perform inference. Synthetic interruption/resume and completed-run
no-op tests precede it. Actual execution is a separate later action:

```bash
python scripts/medgemma_prompt_v2.py smoke --variant v21 --run-dir /governed/path/to/new-v21
python scripts/medgemma_prompt_v2.py launch --variant v21 --run-dir /governed/path/to/new-v21
python scripts/medgemma_prompt_v2.py status --variant v21 --run-dir /governed/path/to/new-v21
```

The live smoke saves the first three planned classifications; launch resumes
from them, not from zero. A checksum-verified smoke snapshot is required before
background execution. `pause`/`resume` are the same actions with `--variant v21`.
Committed rows are checkpointed individually; only an interrupted uncommitted
call may repeat. Scientific failures, source/runtime drift or an eclipse stop
execution for review. Each stage has the existing two-hour limit; there is no
automatic expansion, model search or repeat-until-favorable loop.

The self-contained bundle preserves inputs, parent predictions, prompts,
receipts and derived products for transfer. A different machine/runtime needs
an explicit migration receipt rather than silent mixed-platform resume.

Based on the completed v2 timing (about 17 seconds per classification and
49 seconds per explanation), the old workload took about 45 minutes of record
processing. Plan roughly **45–90 minutes** for this upgrade, including a larger
classification prompt and a differently structured audit; this is not a measured
v2.1 ETA. Live timings will replace that estimate after execution starts.

No study inference, LaunchAgent, circulation PDF or email is created by this
preparation. Governed report text, keys and quotes never enter the shared branch.
