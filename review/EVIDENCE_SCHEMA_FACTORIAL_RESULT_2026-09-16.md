# Development evidence-schema factorial: completed result

**Status:** complete, hash-bound 2 × 2 descriptive study on the frozen first 20
Zoe Reference Annotator development reports. One missing 20-report cell was
generated locally; the other three cells were reused unchanged. No held-out
report, blinded-review identity or protected evaluation result was accessed.

## Question answered

The earlier comparison mixed two factors: the configured local model and the
evidence contract. This experiment crossed them on the same reports:

| Configured local model | Decision-conditioned evidence | Independent category evidence |
|---|---:|---:|
| MedGemma-27B Q2_K | saved | saved |
| Mistral-7B Q5_K_M | saved | **generated once for this study** |

Decision-conditioned evidence asks for phrases supporting an already frozen
four-level decision. Independent category evidence supplies no classification
to the model and asks separately for present, absent and qualifying source
passages. Within each model, the same saved classifications were used only for
the subsequent reference-alignment analysis.

## Complete four-cell result

All fractions below retain the fixed denominator of 100 report-category units
per cell. A record that did not produce valid structured output contributes
five no-parseable-evidence units rather than being dropped or retried.

| Model and evidence contract | Valid records | Units with evidence | Units with ≥1 unchanged quote | Unchanged segments |
|---|---:|---:|---:|---:|
| MedGemma, decision-conditioned | 20/20 | 97/100 (97%) | 60/100 (60%) | 62/124 (50.0%) |
| MedGemma, independent category evidence | 20/20 | 68/100 (68%) | 34/100 (34%) | 36/107 (33.6%) |
| Mistral, decision-conditioned | 16/20 | 57/100 (57%) | 26/100 (26%) | 26/70 (37.1%) |
| Mistral, independent category evidence | 17/20 | 75/100 (75%) | 36/100 (36%) | 40/105 (38.1%) |

The one new Mistral run processed all 20 reports. Seventeen outputs were valid;
two reached the frozen 3,000-token ceiling and one failed the evidence-schema
validator. All three were retained without retry or parameter change.

## Within-model schema effects

Switching from decision-conditioned to independent category evidence had
opposite coverage effects:

- **MedGemma:** evidence coverage decreased by 29 percentage points (97% to
  68%); units with an unchanged quotation decreased by 26 points (60% to 34%);
  and the unchanged-segment fraction decreased by 16.4 points (50.0% to
  33.6%). On paired units, 31 lost evidence and two gained it; 33 lost an
  unchanged quotation and seven gained one.
- **Mistral:** evidence coverage increased by 18 points (57% to 75%) and units
  with an unchanged quotation increased by 10 points (26% to 36%). Its
  unchanged-segment fraction was nearly stable (37.1% to 38.1%). On paired
  units, 22 gained evidence and four lost it; 23 gained an unchanged quotation
  and 13 lost one.

The descriptive interaction is therefore 47 percentage points for evidence
coverage and 36 points for having any unchanged quotation. These are
difference-of-differences descriptions on a purposive development sample, not
hypothesis-tested population effects.

## Category pattern

The direction of the overall coverage change was not produced by one category.
Under the independent contract, MedGemma evidence coverage was lower in all
five categories; Mistral coverage was higher in all five. Literal quotation
coverage remained weakest for the two non-epileptiform subcategories under
both independent-schema cells, which is a useful target for later qualitative
review but not a clinical error claim.

## What this resolves

The large difference in the previously saved evidence streams cannot be
described as a model-weight effect. Evidence-interface design materially changes
what each configured model returns, and the direction of that change is itself
model-dependent. The result supports treating Chris's original evidence module
and the newer independent-category module as named, selectable compatibility
profiles rather than as an old and a replacement method.

It also explains why simply scaling the original confounded comparison to all
1,894 held-out reports would not have been informative. The 20-report crossing
isolates the methodological issue at far lower computational and governance
cost.

## Decision

The planned development experiment is complete. Do **not** launch a full-cohort
evidence run for the current JBHI revision. This exploratory result belongs to
the follow-up methods record and can guide a later, separately frozen study.
The next scientific bridge, if pursued, is the already prepared independent
EEG-qualified review of whether located phrases are relevant and sufficient;
it is optional and is not a prerequisite for the current classification paper.

## Boundaries

- Purposive 20-report development surface; no held-out inference.
- Configured models differ in family, size, quantization and declared chat
  template; this is not an architecture ranking.
- Exact source location is not entailment, clinical relevance, sufficiency,
  calibration or causal faithfulness.
- Invalid structured outputs were retained and affect the all-unit metrics.
- No confidence interval or hypothesis test is claimed from this sample.
- The blinded 20-case review remains sealed.
