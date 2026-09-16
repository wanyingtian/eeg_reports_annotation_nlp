# Follow-up methods paper: evidence interfaces as experimental factors

**Status:** separate working manuscript scaffold. Not part of the current JBHI
revision, response letter or submission package. Completed text and verified
numbers are distinguished below from work that would require a separately
frozen study.

## Working title

**When the Evidence Interface Changes the Comparison: Factor-Aware Evaluation
of Local Language Models for Structured Clinical Reports**

Alternative, more domain-specific title:

**Evidence-Interface Sensitivity in Local Language Models for Structured EEG
Report Annotation**

## Provisional abstract

**Objective:** Model-generated evidence is often compared as though it were an
intrinsic model property, although its form also depends on the prompt, output
schema and whether a prior decision is supplied. We tested whether changing the
evidence interface altered a cross-model comparison in a local clinical-text
pipeline.

**Methods:** Two configured local language models were crossed with two
grammar-constrained evidence contracts on the same 20 EEG development reports.
Decision-conditioned extraction received a frozen four-level category decision;
independent-category extraction received no decision and returned present,
absent and qualifying passages. The same reports and Reference Annotator labels
were used throughout, and within each model the same frozen classifications
were used for downstream alignment. We retained invalid outputs and measured
evidence coverage and unchanged source quotation at report-category and segment
levels.

**Results:** Under decision-conditioned extraction, evidence coverage was 97%
for MedGemma and 57% for Mistral; under independent-category extraction it was
68% and 75%, respectively. Units containing an unchanged quotation moved from
60% to 34% for MedGemma and from 26% to 36% for Mistral. The descriptive
difference-of-differences was 47 percentage points for coverage and 36 points
for unchanged-quotation presence.

**Conclusion:** On this purposive development surface, the evidence contract
changed both the magnitude and direction of the configured-system comparison.
Evidence generation should therefore be registered as an experimental factor,
separately from the model and separately from the policy used to judge source
traceability. Source location alone does not establish relevance, sufficiency
or clinical correctness.

## 1. Introduction

Local clinical language-model studies commonly report a model name alongside a
performance or explanation-quality value. That shorthand hides the configured
system that produced the value: model artifact, quantization, prompt,
serialization, output grammar, decision policy and evidence schema. A
comparison can consequently attribute an interface effect to model weights.

The originating EEG study provides a useful test bed because it separates
classification from source-phrase extraction, preserves a four-level decision
contract and has saved local-model outputs under multiple interface choices.
The present follow-up asks a narrow methodological question: **can an evidence
schema change the apparent ordering of two configured models even when reports,
categories and downstream reference alignment remain fixed?**

The contribution is not another model leaderboard. It is:

1. a controlled model-by-evidence-schema crossing;
2. a typed separation between evidence generation and evidence claims;
3. a demonstrated interaction large enough to reverse the observed ordering;
   and
4. a fail-closed compatibility registry that preserves the original thesis
   method while permitting newer evidence interfaces.

## 2. Relationship to the originating study

The JBHI revision remains the classification and framework-portability paper.
It establishes the five-category, four-level task and compares configured local
systems on reference agreement. Its MedGemma analysis is classification-only.

This follow-up starts where that paper deliberately stops: evidence-interface
behavior. It preserves Chris Tian's classify-then-extract pathway as the
decision-conditioned lineage and adds independent-category evidence as a
separate profile. It neither rewrites the submitted study nor treats the newer
profile as a replacement.

## 3. Methods

### 3.1 Factorial development surface

The completed experiment used 20 fixed Zoe Reference Annotator development
reports and five categories, yielding 100 report-category units per cell. The
two configured-model factors were Mistral-7B-Instruct-v0.2 Q5_K_M and
MedGemma-27B-text-it Q2_K, each using its declared local chat template.

The evidence-schema factors were:

- **Decision-conditioned:** supply the saved four-level classification and
  request source reasons supporting that decision.
- **Independent category evidence:** supply no classification and request
  separate present, absent and qualifying source passages.

Reports and reference labels were identical across all cells. Within each
model, frozen classifications were identical across the two schemas and were
used only for subsequent reference-alignment analysis. Three cells reused
saved outputs. The missing Mistral independent-category cell was generated once
under a pre-execution plan. Invalid and unfavorable results were retained.

### 3.2 Evidence and traceability endpoints

- Valid structured outputs out of 20 reports.
- Evidence coverage out of 100 report-category units.
- Units containing at least one unchanged source quotation.
- Unchanged quotation segments out of all substantive segments.
- Paired unit transitions between evidence schemas within each model.

These are descriptive development endpoints. No confidence interval,
hypothesis test or held-out population claim is made.

### 3.3 Evidence-profile contract

The reusable system represents evidence generation and evidence claims as two
independent axes. Generation profiles state whether a classification enters the
model call and which schema is returned. Claim profiles state which matching or
review stages are accepted and which language those stages authorize.

The original-thesis source-support calculation remains a named compatibility
profile for decision-conditioned outputs. Verbatim, normalized,
retrieval-assisted and adjudicated profiles are separately named. Unknown or
incompatible selections fail closed, and adjudicated decision support cannot be
selected without recording that human review is required.

## 4. Results

The complete verified result and its interpretation are drafted in
`FOLLOW_UP_PAPER_DRAFT_SCHEMA_CONFOUND_SECTION_2026-09-16.md`. Its central table
is:

| Configuration | Evidence coverage | Units with unchanged quotation | Unchanged-segment fraction |
|---|---:|---:|---:|
| MedGemma, decision-conditioned | 97/100 | 60/100 | 62/124 |
| MedGemma, independent-category | 68/100 | 34/100 | 36/107 |
| Mistral, decision-conditioned | 57/100 | 26/100 | 26/70 |
| Mistral, independent-category | 75/100 | 36/100 | 40/105 |

The schema effect was negative for MedGemma and positive for Mistral on both
unit-level coverage endpoints. Category-stratified results retained the same
coverage direction in all five categories. Two new Mistral outputs reached the
frozen token ceiling and one failed schema validation; all three remained in
the denominator.

## 5. Discussion

### 5.1 The interface is part of the measured system

The interaction demonstrates why evidence generation cannot be treated as an
unreported implementation detail. Under one reasonable contract, MedGemma had
the higher observed quotation coverage; under the other, Mistral did. The study
therefore supports configured-system language and factor registration rather
than a model-grounding ranking.

### 5.2 The methodology caught a consequential confound

The framework's practical contribution is demonstrated, not merely asserted:
factor typing prevented a large interaction from being published as a model
effect. This supports registries that name model, interface, prompt, grammar,
schema, decision dependence and claim policy before interpretation.

### 5.3 Source location is only one rung

An unchanged quotation is auditable provenance, but it may be irrelevant,
contradictory or insufficient. Normalized, fuzzy and semantic localization can
increase recall without becoming entailment. Decision support and clinical
usefulness require separate validation layers.

### 5.4 Relation to the original thesis method

The thesis pathway is not an obsolete baseline. It is the decision-conditioned
compatibility profile: a model first produces a structured decision and then a
second call returns source phrases in relation to it. The independent profile
answers a different question. Their coexistence is the methodological result.

## 6. Limitations

- Twenty purposively selected development reports; no held-out generalization.
- One deterministic run per configured cell; no run-to-run variability.
- Models differ in family, size, quantization and native chat templates.
- Literal quotation does not establish relevance, sufficiency or entailment.
- Three invalid Mistral independent-category outputs were retained, so output
  validity contributes to the observed configured-system behavior.
- The existing technical-reader review does not replace EEG-qualified review.

## 7. What is complete and what remains optional

### Complete and manuscript-ready as a development result

- Frozen 2 × 2 design and exact run receipts.
- All four cells and invalid-output accounting.
- Paired transitions and category strata.
- Typed compatibility registry and immutable catalog.
- Bounded interpretation and claim firewall.

### Optional work before choosing a publication form

- Independent EEG-qualified review of the sealed 20-case packet, if the paper
  will claim evidence relevance or sufficiency.
- A separately frozen held-out replication, if a venue requires population
  inference rather than a methods demonstration.
- A public-safe figure showing the crossover interaction.
- Venue-specific references and positioning against explanation evaluation,
  structured generation and clinical NLP audit literature.

No item in this optional list should delay or modify the current JBHI
submission.

