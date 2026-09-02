# Reason traceability: historical reconstruction and cross-model audit

**Status:** completed experimental analysis for author review; no new study-model
inference, clinical validation, or manuscript admission.

## Outcome

The explanation machinery is still usable at report-component level. A single
typed adapter now accepts:

- the thesis-era Mistral reason and saved-polarity artifact;
- contemporary Mistral fixed-decision evidence under both saved interfaces;
- MedGemma fixed-decision evidence; and
- MedGemma's independent present, absent, and qualifying evidence schema.

The audit retains every saved output and produces a governed, keyed segment
ledger plus a text-free aggregate receipt. This is an operational extension of
the thesis framework, not a new explanation-quality claim.

## What the historical search established

Chris's author-uploaded evidence script is recoverable at commit
`d7cf1423fa98b0bed0bc55153550c640a18f63fc` (October 3, 2025). The 2,000-row
reason/polarity artifact was created earlier, on May 13, 2025. The script and
final thesis are not the same executable specification:

| Element | Public 2025 script | Final thesis description |
|---|---|---|
| Selected units | Mistral labels 3/4 | learned polarity labelled abnormal-supporting |
| Explanation split | semicolons | sentences |
| Fuzzy comparison | whole report | report sentences |
| Semantic comparison | whole report | report text |
| Fuzzy / cosine thresholds | 70 / 0.70 | 70% / 0.70 |

The learned-polarity selection resolves the submitted denominator exactly at
2,180. With that selection held fixed, the public-script replay gives
2,018/2,180 (92.57%). A deterministic reconstruction of the thesis prose gives
1,911/2,180 (87.66%). Neither gives the submitted 2,132/2,180 (97.80%), and no
threshold was changed to pursue that value. The submitted numerator therefore
remains a historical reported result rather than a reproduced result.

## Exact embedding lineage recovered without asking Chris

The historical code names `sentence-transformers/all-MiniLM-L6-v2` but does not
pin a revision. Upstream repository chronology makes
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf` the latest available revision before
the May 13 artifact. The local replay used
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Between those revisions only the
model card changed: weights, tokenizer, pooling modules, model configuration,
and sentence-transformer configuration are byte-identical. The weights SHA-256
is `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db`.

This recovers a functionally equivalent embedding surface. It does not prove
which upstream revision the unpinned historical environment resolved.

## The new conservative contract

The audit asks two separate questions:

1. **Verified quotation:** is the generated segment an unchanged substring of
   the exact report? Only this stage is released as a verbatim quote.
2. **Review candidate:** can case, whitespace, typography, fuzzy sentence
   matching, or MiniLM similarity locate plausible source material for an
   author to inspect? These stages are navigation aids, not factuality labels.

Both “at least one segment located” and “every segment located” are reported.
This matters because a multi-part explanation should not become wholly
traceable merely because one clause matches.

## Historical complete-surface finding

The 2,180 abnormal-supporting report-category units contain 3,115 semicolon
segments; four units have no usable segment. Under the strictest rule, 593
segments (19.04%) are unchanged quotations. Progressive review stages locate
3,045/3,115 segments (97.75%), leaving 70 unresolved. At unit level, 2,140 of
2,176 nonempty units contain at least one located segment, while 2,107 have
every segment located. Only 550 units contain an unchanged quotation and 201
have every segment unchanged.

These are not replacements for the submitted 97.8%. They expose why the
aggregation rule matters and give the authors a governed failure-review queue.

## Same contract on saved contemporary evidence

The common adapter was exercised on the already-produced first 20 development
reports. No Mistral or MedGemma generation was rerun.

| Saved evidence stream | Substantive segments | Unchanged quotes | Located for review | Unresolved |
|---|---:|---:|---:|---:|
| Mistral raw-completion, decision-conditioned | 51 | 10 (19.6%) | 49 (96.1%) | 2 |
| Mistral native-chat, decision-conditioned | 70 | 26 (37.1%) | 70 (100%) | 0 |
| MedGemma v2, decision-conditioned | 124 | 62 (50.0%) | 124 (100%) | 0 |
| MedGemma v2.1, independent evidence roles | 107 | 36 (33.6%) | 107 (100%) | 0 |

The v2.1 output contains 37 present, 54 absent, and 16 qualifying passages
across 68 of 100 category cells; the other 32 cells explicitly contain no
passage. This demonstrates that the thesis idea can be carried forward in a
more reviewable schema without feeding classifications into the evidence call.

The percentages above are transport and formatting diagnostics on 20
development reports. Different prompts and evidence roles mean they are not a
model ranking, a factuality comparison, or an evaluation-cohort estimate.

## Scientific use now available

- The historical polarity/correctness result can remain a recovered thesis
  analysis with its corrected counts and non-causal interpretation.
- The unreproduced 97.8% value can be described as submitted history and kept
  out of new quantitative claims.
- The strict quote layer can be applied identically to any saved Mistral or
  MedGemma evidence stream.
- Fuzzy and semantic candidates now form a governed review queue for failure
  analysis and future prompt design, never silent accepted evidence.
- Present, absent, and qualifying passages can be summarized by schema field,
  which makes the granular access useful without pretending to expose a
  model's hidden reasoning.

## Reproducibility and custody

Run `scripts/audit_cross_model_reason_traceability.py` with explicit saved
evidence, report database, and local MiniLM paths. The complete aggregate receipt
is `data/governed/analysis-runs/jbhi-cross-model-reason-traceability-20260902/aggregate-traceability.json`
(SHA-256 `92cb1ecf85f83f1bfd6af5f32587a268087fee2fc7575381f2d45fb10315d555`).
The keyed segment ledger remains governed and contains hashes, stages, and
scores but no report or reason text.

No full contemporary evidence run is required for the current author package.
A later full-cohort evidence generation would be a separately frozen compute
study, not a prerequisite for reporting the completed classification comparison.
