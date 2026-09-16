# MedGemma fixed-decision evidence extraction follow-up

**Status:** implementation prepared; the 100-report development execution must
be hash-frozen before inference. This is a follow-up methods study, not evidence
required for the current JBHI revision.

## Question

Can the thesis-originated second call—grammar-constrained extraction of source
phrases after classification—be transported to the completed MedGemma
configuration while preserving fixed decisions and exact source traceability?

## First executable stage

The first stage uses the already-declared 100-report Zoe development surface.
It does not access the 1,894 held-out evaluation reports.

- The completed MedGemma v1 classifications are immutable inputs.
- The historical explanation prompt and explanation grammar are unchanged.
- MedGemma uses its pinned local GGUF artifact and embedded instruction-turn
  template; no report leaves the laptop.
- The explanation runner performs zero classification calls and rejects any
  output whose copied decision differs from the saved classification.
- Every row is checkpointed in manifest order and can be resumed only under an
  identical execution contract.
- Keyed phrases remain in governed storage. Only text-free counts and hashes
  enter the repository.

The saved Mistral explanations for the same 100 reports are not regenerated.
Both streams are summarized under one primary rule: an accepted quotation is
an unchanged, nonblank substring of the exact source report. Fuzzy or semantic
matching is excluded from the primary comparison.

## Outcomes

The descriptive outcomes are schema validity, copied-decision agreement,
substantive phrase count, exact-quotation count, units with any/all exact
quotations, and category-specific exact-traceability counts. All denominators
are retained, including blank, fallback, rejected, and unmatched evidence.

These outcomes measure output lawfulness and source presence. They do not
measure clinical correctness, entailment, calibrated confidence, hidden
reasoning, or whether a quoted phrase caused the classification. Since the
configured systems differ in model and interface, the result is not a pure
base-weight comparison.

## Stopping rule and next gate

The development run is complete when all 100 manifest rows have valid receipts
or an unrecoverable execution error is documented. No threshold is optimized
and no prompt is revised in this stage.

A full held-out evidence run is warranted only if the development result is
technically usable: all 100 outputs are schema-valid, no copied decision drifts,
and the exact-quotation audit yields enough nonempty evidence to support a
meaningful author-review workload. The authors can then decide whether the
roughly one-day full-cohort compute cost serves a follow-up paper. It remains
unnecessary for the current JBHI classification-portability claim.

## Relationship to Chris's historical tools

The public 2025 alignment and factuality scripts remain preserved historical
comparators. Their fuzzy and semantic matches are useful for locating review
candidates, but they are not silently treated as verified quotations. The
primary audit uses the later, stricter contract already developed in this
repository: exact substrings are verified; normalized, fuzzy, and semantic
matches remain review aids.
