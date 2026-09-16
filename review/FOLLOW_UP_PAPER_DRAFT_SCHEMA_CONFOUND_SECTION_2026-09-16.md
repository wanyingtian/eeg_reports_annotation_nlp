# Draft section: evidence-schema confounding in cross-model explanation comparison

**Status:** draft prose for the follow-up methods paper. Not part of the
current JBHI revision. Numbers below are independently re-derived from the
governed run at `data/governed/study-runs/jbhi-evidence-schema-factorial-development-20260916/`
(`factorial_units.csv`, 400 rows) and match `review/EVIDENCE_SCHEMA_FACTORIAL_RESULT_2026-09-16.md`
exactly. This document is prose to be adapted into the eventual manuscript,
not a new governed artifact itself.

---

## Motivation

A recurring failure mode in cross-model evaluation is attributing a measured
difference to the model when it is at least partly a property of the
interface built around it. Prior sections of this framework already
formalize this concern for classification: a comparison between two
configured systems cannot isolate a base-weight effect unless every other
factor — quantization, prompt, grammar, decision policy — is either held
fixed or explicitly varied as the manipulated factor. The same concern
applies, with less obvious force, to model-generated *explanations*.

Our initial cross-model evidence comparison held the evidence-generation
schema fixed at *decision-conditioned* extraction — the frozen four-level
classification is supplied to the model, which is then asked only for
phrases supporting that already-fixed decision. Under this schema, on the
full 100-report development surface, a configured MedGemma system returned
an unchanged source quotation for 45.3% of its substantive phrases, versus
18.0% for a saved historical Mistral configuration. Read naively, this
invites the claim that MedGemma explanations are more source-grounded than
Mistral's. That claim would be premature for a reason distinct from the
usual caveats about entailment and relevance: the two saved evidence streams
were never generated under an experimentally crossed design, so a schema
effect and a model effect are observationally identical in that comparison.

We therefore ran a small factorial study to separate them.

## Design

We crossed two factors on the same 20 Zoe Reference-Annotator development
reports, fixing the reference labels, report identifiers, and five-category
task across all four cells:

| | Decision-conditioned evidence | Independent-category evidence |
|---|---|---|
| **MedGemma-27B (Q2_K, native chat)** | saved | saved |
| **Mistral-7B (Q5_K_M)** | saved | generated once, for this study |

*Decision-conditioned* evidence supplies the model with its own already-fixed
four-level decision and asks only for supporting phrases. *Independent-category*
evidence supplies no decision at all; the model is asked separately, for each
category, to identify present, absent, and qualifying source passages. Three
of the four cells reused already-saved, previously generated output; one new
Mistral run (20 reports, independent-category prompt) was executed for this
study, with the frozen classification, report set, and reference labels held
identical to the other three cells. Two of its 20 outputs reached the token
ceiling and one failed schema validation; all three were retained as
non-parseable rather than dropped or retried, following this project's
standing rule against silently discarding unfavorable cases.

## Result

All figures are report-category units (100 per cell: 20 reports × 5
categories).

| Configuration | Evidence coverage | Units with an unchanged quotation | Unchanged-segment fraction |
|---|---:|---:|---:|
| MedGemma, decision-conditioned | 97/100 (97%) | 60/100 (60%) | 62/124 (50.0%) |
| MedGemma, independent-category | 68/100 (68%) | 34/100 (34%) | 36/107 (33.6%) |
| Mistral, decision-conditioned | 57/100 (57%) | 26/100 (26%) | 26/70 (37.1%) |
| Mistral, independent-category | 75/100 (75%) | 36/100 (36%) | 40/105 (38.1%) |

Changing the evidence schema moved the two models in opposite directions.
MedGemma's evidence coverage fell by 29 percentage points and its rate of
returning any unchanged quotation fell by 26 points when the decision was
withheld. Mistral showed the reverse pattern: coverage rose by 18 points and
unchanged-quotation rate rose by 10 points under the same schema change. The
resulting interaction — a 47-point difference-of-differences in coverage and
36 points in unchanged-quotation rate — is not attributable to either model
alone; it requires both factors to explain.

The direction was consistent across all five categories in both directions
of the crossing, not driven by one category's behavior.

## What this establishes, and what it does not

This result establishes that the evidence-generation schema is not a nuisance
factor safely ignored when comparing two models' explanations: on this
development sample, its effect size is comparable to, and opposite in sign
for, the two models under study. It follows directly that the original
decision-conditioned-only comparison cannot support a claim of the form
"model A grounds its evidence better than model B" — that comparison is
confounded with a schema choice that, if reversed, reverses which system
looks better.

This result does not establish which schema is preferable in general,
whether either model's grounded phrases are clinically relevant or
sufficient, or a population-level effect of any kind: the study is a
20-report purposive development crossing, one cell was single-shot
generation rather than a receipted multi-run estimate, and no hypothesis test
or confidence interval is claimed. It also does not bear on whether MedGemma
or Mistral is the better *classifier* — Core Agreement and Certainty-Adjusted
Agreement, the endpoints used for that separate question, are untouched by
this study.

## Contribution to the methodological framework

The practical value of this result is that it is a demonstrated instance of
exactly the failure mode the surrounding evaluation-surface framework was
built to prevent, rather than a hypothetical one. A framework that requires
every comparison to name its held-fixed and manipulated factors is a
reasonable precaution in the abstract; a framework that has already caught a
47-point, sign-reversing confound before it reached a published claim is a
validated one. We treat this factorial crossing as the primary evidentiary
basis, in this follow-up line of work, for treating evidence-generation
schema as a first-class experimental factor — coordinate with model family,
quantization, and interface — rather than an implementation detail folded
into "which model produced this explanation."

## Immediate implication for study sequencing

No full-cohort evidence run is justified by the present result. The 20-report
crossing was sufficient to detect and characterize the confound at far lower
computational and governance cost than scaling either arm to the full
1,894-report evaluation surface would have required, and scaling would not by
itself resolve the open relevance/sufficiency question that motivated the
sealed 20-case blinded review. That review, and a decision on which
evidence-generation schema (or an explicit, named mixture) to standardize on
for any future full-cohort run, are the two remaining gates before this line
of work could support a quantitative claim.
