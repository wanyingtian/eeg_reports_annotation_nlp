# MedGemma fixed-decision evidence extraction: 100-report result

**Status:** completed development-surface transport diagnostic; not required for
the current JBHI revision and not admitted as a held-out explanation-quality
result.

## What ran

The historical second call was run locally on the fixed MedGemma v1
classifications for all 100 reports in the declared Zoe development surface.
The model, GGUF revision, native instruction-turn template, historical
explanation prompt, grammar, sampling parameters, report order, and source
classification file were hash-frozen before inference.

The run made 100 explanation calls and zero classification calls. All 100
outputs were schema-valid, all 500 copied category decisions matched their
saved classifications, and no case was removed or repaired. The run took about
79 minutes and remained entirely local.

## Same-report descriptive result

The new MedGemma evidence and the already-saved historical-interface Mistral
evidence were audited on the same 100 reports under one deliberately narrow
reporting rule: a verified quotation must be an unchanged, nonblank substring
of the exact source report. Fuzzy and semantic matches were retained as
source-localization candidates rather than folded into the quotation count.

| Saved configured system | Substantive phrases | Exact quotations | Exact fraction | Category units with any exact quotation | Category units with all phrases exact |
|---|---:|---:|---:|---:|---:|
| MedGemma native v1, fixed decisions | 728 | 330 | 45.3% | 305 / 484 | 233 / 484 |
| Mistral historical interface, saved output | 295 | 53 | 18.0% | 48 / 245 | 27 / 245 |

MedGemma also produced far fewer empty or declared-no-evidence units: 16 of 500,
compared with 255 of 500 for the saved Mistral stream. Of MedGemma's 728
substantive phrases, 390 were locatable only after case, typography, or
whitespace normalization and eight remained unresolved by those conservative
stages. For Mistral, 187 of 295 were normalization candidates and 55 remained
unresolved.

## Interpretation

This is strong technical evidence that the thesis-originated two-call design
transports to the configured MedGemma system. MedGemma can preserve the fixed
classification and return a substantially denser set of source-locatable
phrases under the existing grammar.

It is not yet evidence that MedGemma explanations are clinically better or
more causally faithful. The systems differ in model and interface; the
extraction is conditioned on each system's own saved decisions; phrase counts
are dependent; and source presence does not establish relevance or entailment.
The coverage difference may reflect a more assertive extraction behavior as
well as stronger literal copying. A blinded author or clinical review of a
bounded stratified sample is the next informative step.

## Decision gate

The technical gate for a future full-cohort evidence run is passed: every
development output is lawful, decision-preserving, and usable for the exact
source audit. A full 1,894-report pass would nevertheless require roughly 25
hours on the current laptop and would mainly add prevalence precision. It
would not answer the remaining relevance/entailment question. The efficient
next step is therefore a frozen, source-first review sample; full-cohort
generation can follow only if that review shows publication value for the
follow-up methods paper.

The keyed phrases remain in governed storage. The repository retains only the
plan, method, aggregate counts, boundaries, and artifact hashes.
