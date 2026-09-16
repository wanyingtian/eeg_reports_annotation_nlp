# Reference-aligned evidence traceability: development result

**Status:** completed, hash-bound descriptive analysis on the frozen first 100
Zoe Reference Annotator development reports. No new model inference, held-out
evaluation access or blinded-review unmasking occurred.

## What was joined

For each of five EEG categories in each report, the analysis joined:

1. the retained Reference Annotator four-level label;
2. the saved configured-system four-level decision; and
3. the saved evidence output's source-traceability status.

This produces 500 report-category units per configured system. The phrase
**reference-aligned evidence traceability** means that agreement with the
retained reference and source location of evidence can be examined together.
It does not mean that source-located text has been clinically judged relevant,
sufficient or correct.

## Complete overall result

| Configured system | Binary agreement | Exact four-level agreement | Units with evidence | Units with ≥1 unchanged quote | Unchanged evidence segments |
|---|---:|---:|---:|---:|---:|
| MedGemma native v1, fixed-decision evidence | 462/500 (92.4%) | 416/500 (83.2%) | 484/500 (96.8%) | 305/500 (61.0%) | 330/728 (45.3%) |
| Saved Mistral historical-interface evidence | 475/500 (95.0%) | 411/500 (82.2%) | 245/500 (49.0%) | 48/500 (9.6%) | 53/295 (18.0%) |

The classification agreement values are close and mixed: Mistral has the
higher binary agreement on this development surface, while MedGemma has the
higher exact four-level agreement. The evidence streams behave very
differently, but they were produced under different evidence schemas and
prompts. The evidence counts therefore describe configured systems, not model
weights alone.

## What happens when agreement is conditioned first

| Configured system and stratum | Units | Units with evidence | Evidence-bearing units with ≥1 unchanged quote | Unchanged segments |
|---|---:|---:|---:|---:|
| MedGemma, binary agreement | 462 | 450 (97.4%) | 290/450 (64.4%) | 311/655 (47.5%) |
| MedGemma, binary disagreement | 38 | 34 (89.5%) | 15/34 (44.1%) | 19/73 (26.0%) |
| Mistral, binary agreement | 475 | 228 (48.0%) | 44/228 (19.3%) | 49/276 (17.8%) |
| Mistral, binary disagreement | 25 | 17 (68.0%) | 4/17 (23.5%) | 4/19 (21.1%) |
| MedGemma, exact four-level agreement | 416 | 404 (97.1%) | 268/404 (66.3%) | 285/539 (52.9%) |
| MedGemma, exact four-level disagreement | 84 | 80 (95.2%) | 37/80 (46.3%) | 45/189 (23.8%) |
| Mistral, exact four-level agreement | 411 | 182 (44.3%) | 29/182 (15.9%) | 31/221 (14.0%) |
| Mistral, exact four-level disagreement | 89 | 63 (70.8%) | 19/63 (30.2%) | 22/74 (29.7%) |

For MedGemma's configured evidence call, unchanged quotations are more common
when its classification agrees with the reference. The saved Mistral stream
does not show the same direction: it returns evidence more selectively, and
its smaller disagreement strata have higher literal-traceability fractions
than its agreement strata. Category composition and the distinct evidence
schemas both contribute to these aggregates.

This is the central methods result: **literal traceability is not a universal
proxy for reference agreement.** It is a property of the configured evidence
interface and its outputs. A model can agree with the reference yet provide no
unchanged quotation, or disagree while quoting the source exactly.

## Category pattern

MedGemma supplied substantive evidence in 94–100% of units across the five
categories and at least one unchanged quotation in 52–69% of all units. The
saved Mistral stream supplied evidence in 46–50% and at least one unchanged
quotation in 4–21%. These are complete category ranges, not selected favorable
cells. Rare disagreement cells—particularly epileptiform categories—are too
small for standalone inference and remain visible in the aggregate ledger.

## Decision after the frozen analysis

Do **not** launch the approximately 25-hour full-cohort evidence run merely to
make these configured-system percentages more precise. It would scale the same
model-by-schema confounding.

The next computationally informative experiment is a development-only
model-by-evidence-schema crossing with fixed classifications: apply the same
decision-conditioned and independent evidence contracts to both local models.
That experiment can determine how much of the traceability difference follows
the model and how much follows the evidence interface. Existing saved outputs
should be inventoried first so only genuinely missing cells are generated.

The independent EEG-qualified 20-case read remains an optional bridge from
source location to clinical relevance and sufficiency. It is not required for
the current JBHI classification revision or for the model-by-schema methods
experiment.

## Boundaries

- Development surface only; no held-out or population inference.
- Reference agreement is not independently established clinical correctness.
- Source location is not entailment, clinical sufficiency or causal reasoning.
- The two saved evidence schemas differ, preventing model-only attribution.
- No hypothesis tests, confidence intervals or model-ranking claim are made.
- The 20-case blinded review remains sealed and available for later use.
