# Mistral endpoint-guidance ablation

This is a small, post-submission methods experiment. It does not alter the
submitted Mistral study, its independent reproduction, or the removal of the
unsupported historical `14%` prompt-gain statement.

## Why this contrast is recoverable

Chris Tian's November 19, 2024 prompt-development deck states `V2: V1 +
Confidence Guideline`. The exact submitted prompt and the named confidence
guideline survive. The candidate therefore removes only these two sentences:

> Err on the side of confident decisions. Use 1 or 4 whenever possible. Only
> use 2 or 3 if there is strong, unavoidable ambiguity. Choose 2 or 3
> sparingly, only when absolutely necessary.

The model, report order, grammar, interface, definitions, examples, questions,
cross-field constraints, sampling parameters, and reference annotations remain
fixed. This supports a one-factor prompt-component interpretation on the fixed
development reports. It does not recreate every historical prompt version:
the deck's version labels are presentation-local, and complete producing
artifacts for the broader historical sequence were not recovered.

## Frozen scope

- Model: the checksum-pinned Mistral-7B-Instruct-v0.2 Q5_K_M artifact used in
  the reproduction.
- Surface: the exact first 100 Zoe Reference-Annotator development reports.
- Parent: the saved first-100 subset from the completed raw-completion
  reproduction.
- Candidate: the same raw-completion pipeline with only the endpoint-guidance
  block removed.
- Calls: one classification call per report; no explanation call and no
  protected-cohort execution.
- Outcomes: all 100 results, favorable or unfavorable, are retained.

A smaller prefix is useful only as an execution smoke test. Because a complete
100-report parent already exists and the full candidate run is short, the
scientific comparison uses all 100 cases rather than selecting a favorable
subsample.

## What the experiment can explain

The useful question is not whether a historical percentage can be recovered.
It is whether an explicit preference for levels 1 and 4 changes:

1. use of the four certainty levels;
2. binary present/absent decisions;
3. exact and binary agreement with the Reference Annotator; and
4. the two declared overall/subtype consistency conditions.

These are paired, same-report measurements. If classification changes, a later
source-traceability review may show which report passages are available to a
reader for the changed cases. Such quotations do not reveal the model's hidden
reasoning or prove why the instruction caused a change.

## Publication boundary

The default placement, if the completed result is useful, is one compact
supplementary table and a short point-by-point response paragraph. The main
manuscript needs at most one sensitivity sentence. Null or adverse findings are
equally reportable, and no result is admitted automatically.

The exact machine-readable freeze is
[`model-receipts/mistral-endpoint-guidance-ablation.preregistered.json`](model-receipts/mistral-endpoint-guidance-ablation.preregistered.json).
