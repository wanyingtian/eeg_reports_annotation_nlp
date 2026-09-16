# Development-only model × evidence-schema crossing

**Status:** pre-execution plan. Three saved cells verified; one 20-report cell
remains to be generated. This is follow-up methods work, not part of the current
JBHI revision and not a held-out model comparison.

## Why this experiment is now warranted

The completed reference-aligned traceability analysis found that literal source
quotation relates differently to Reference Annotator agreement across the two
saved configured systems. A full-cohort run would make that confounded contrast
more precise without separating model behavior from evidence-interface design.

The smallest useful follow-up crosses two local models with two evidence
contracts on the identical first 20 development reports:

| Local model | Decision-conditioned evidence | Independent category evidence |
|---|---|---|
| Mistral-7B | Saved, native-chat, 20 reports; retain four inspection-invalid records | **Missing: run once on 20 reports** |
| MedGemma-27B Q2_K | Saved, native-chat, 20 reports | Saved, native-chat, 20 reports |

All three saved cells have the same report-key manifest and exact source
database. The decision-conditioned prompt and grammar are shared across models.
The independent prompt and grammar will likewise be shared across models.

## Factor definitions

### Model factor

- Mistral-7B-Instruct-v0.2 Q5_K_M, local GGUF.
- MedGemma-27B-text-it Q2_K, local GGUF.

Each uses its declared embedded chat template. This is a configured local-model
factor, not a pure architecture or parameter-count experiment.

### Evidence-schema factor

1. **Decision-conditioned:** the model receives a frozen four-level
   classification and returns reasons tied to that decision.
2. **Independent category evidence:** no model or reference classification is
   supplied. The model returns present, absent and qualifying source passages
   separately for each category.

Within each model, the same saved classifications are used for downstream
reference alignment across both evidence schemas. MedGemma's independent output
can be re-associated with its v2 classifications because the run receipt proves
that classifications were not supplied to the evidence call and did not alter
the evidence generation.

## One permitted inference operation

Generate only the missing Mistral independent-category-evidence cell on the
frozen 20-report manifest. Use native chat, temperature 0, the frozen
independent prompt and grammar, the existing Mistral model artifact, and the
same Mistral classification file used by its decision-conditioned cell. The
classification file is an execution/key contract only; its values must not
enter the model message.

Stop after exactly 20 records. Retain every output, including invalid,
unresolved, empty or unfavorable results. Do not run either model on held-out
reports and do not regenerate any completed cell.

## Frozen descriptive endpoints

For each of the four cells report:

- structured-output-valid records out of 20;
- substantive evidence coverage out of 100 report-category units;
- units containing at least one unchanged source quotation;
- substantive and unchanged segment counts;
- normalization/location-candidate and unresolved segment counts; and
- the same measures stratified by category.

Reference alignment is secondary and uses the same fixed classifications within
each model. Report within-model schema contrasts and the descriptive
difference-of-differences, but no hypothesis test or confidence interval on this
purposive development sample.

## Interpretation boundary

The experiment can show whether evidence coverage and literal traceability move
with the evidence contract, the configured local model, or both. It cannot
establish clinical sufficiency, entailment, causal faithfulness, calibrated
confidence, model-family superiority or held-out performance.

The optional blinded EEG review remains the separate bridge from source
location to clinical relevance. It is not required to execute or interpret the
technical 2 × 2 crossing.
