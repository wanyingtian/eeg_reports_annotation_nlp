# BERT input-fairness sensitivity

**Status:** completed, read-only post-submission sensitivity. No model was fit
or invoked, and no historical population or result was changed.

## Question

The reviewer asks whether BERT's 512-token input limit placed it at a material
disadvantage relative to models that received the complete report, especially
when a report's Impression, Interpretation, Conclusion, or Clinical
Correlation appeared late in the text.

## What was tested

The analysis used the exact locally cached `bert-base-uncased` tokenizer at
revision `86b5e0934494bd15c9632b12f734a8a67f723594`, with the historical
right-truncation rule and 512-token maximum. The eligible population was fixed
to the complete five-label Reference Annotator records: 1,395 Zoe and 499
Maria reports.

Every report exposed to BERT truncation was then removed from **all** saved
model surfaces—not just BERT—and category metrics were recomputed on the same
remaining reports. The five surfaces were fresh BoW+LR, fresh BERT+LR,
submitted Mistral, reproduced Mistral, and the independently frozen MedGemma
configuration. This is an exclusion sensitivity, not a new model run.

## Finding

- Zoe: 1 of 1,395 reports exceeded 512 BERT tokens (0.072%). Its recognized
  downstream heading began within the retained text.
- Maria: 3 of 499 reports exceeded 512 tokens (0.601%). All three contained a
  recognized downstream heading; two had at least one such heading beyond the
  retained boundary, and in one the first recognized `Interpretation` heading
  itself began beyond the boundary.
- Across both cohorts, 4 of 1,894 reports were truncation-exposed.

After excluding those four reports from every model, no category-level Core
Agreement ranking changed in either cohort. For BERT+LR, the largest absolute
change was 0.106 percentage points in Core Agreement and 0.310 points in F1.
Across all five model surfaces, the largest Core Agreement change was 0.187
points. The largest F1 change was 1.259 points in the rare Maria focal
epileptiform stratum for reproduced Mistral, where removing three reports has
a relatively visible denominator effect. BERT's zero or near-zero rare
epileptiform F1 findings were unchanged in interpretation.

## Reviewer-facing interpretation

The reviewer's concern is valid in mechanism: right truncation can remove a
late interpretive section, and it did so for two Maria reports. It affected,
however, only four held-out reports in the complete-case evaluation. A
same-population exclusion sensitivity changed neither model ordering nor the
study's central comparative conclusions. The revised paper should report both
facts rather than claiming that the input formats were identical.

## Boundaries

- This does not estimate how a counterfactual long-context BERT classifier
  would have labelled the four reports.
- Heading recognition is conservative; an unrecognized section may still be
  present.
- Small changes do not prove model equivalence.
- This analysis does not establish patient independence, clinical validity,
  or a causal model-family effect.
- The four exposed report keys and boundary details remain in governed
  storage; no report key or report text is reproduced here.

## Reproducibility receipt

- Frozen implementation commit:
  `76b34ded4aedc3da82b10ae4cff6267078a4123c`
- Preregistered plan SHA-256:
  `42e9e7031fca72348450261dfa3f352817d4dc6da718abe9e8cff2e512b17512`
- Governed aggregate SHA-256:
  `538509175ba3e1c39e6500f24400a6a3e60eb19a6e64abd73c86e49688f873d1`
- Governed case ledger SHA-256:
  `a45f28e02590e4011e33fec8023c40a4b50f8ea02b703a91b07fdde249c6927c`

The receipt records a dirty worktree because of a pre-existing, unrelated
`.gitignore` edit. The preregistered plan, analysis implementation, and tests
were committed before the governed result was inspected.
