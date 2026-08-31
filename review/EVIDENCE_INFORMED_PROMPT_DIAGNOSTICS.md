# Evidence-informed prompt diagnostics

This extends Chris Tian's thesis-derived framework rather than replacing it
with repeated accuracy-only prompt trials. Her *Explanation and Evaluation*
section distinguishes source traceability from explanation/label alignment,
uses baseline feature evidence, and describes rule-based and ClinicalBERT+LR
polarity checks. The *Prompt Tuning* appendix ties revisions to the original
100-case development set, literature-derived definitions and expert input.

The new diagnostic connects four already available information sources:

1. The reference annotation, preserving its role as an annotation rather than
   independently adjudicated truth.
2. Saved MedGemma v1/v2 and historical/native Mistral decisions on identical
   development report keys; disagreement is a review signal, not a vote.
3. The model's generated evidence and exact source passages, kept side by side.
4. Specific review questions about negation, category specificity, temporal
   context, version regressions and untraceable wording. These questions are
   prepared, not silently answered by another model or scored as ground truth.

## Literal source-span layer

`literal-source-span-v1` is an additive inspection. It leaves every historical
prompt, generated output, classification and frozen metric unchanged.

- Exact nonblank quotations get all matching source offsets and a source-text
  hash. Offsets are zero-based Unicode code points with exclusive ends.
- Blank reasons, malformed objects and mismatched copied decisions abstain in
  the verified-quotation view. Original responses remain available.
- Whitespace-only matches retain the model's original wording and separately
  locate exact source slices. They do not retroactively count as literal model
  quotations. Other normalization matches remain diagnostic hints only.
- Source presence does not prove semantic support, correct clinical
  interpretation or causal access to the model's reasoning.

This is **not the thesis's factuality score**: that instrument focuses on
abnormal supporting evidence and includes fuzzy and semantic matching. Nor is
it a rerun of the thesis's learned ClinicalBERT polarity classifier. Those
instruments and their training/producing artifacts must keep separate receipts.

## Bounded failure follow-up

The fixed first-20 evidence sample may miss important development errors.
The audit therefore records evidence coverage separately from classification
coverage and writes a manifest for **all focal false positives and all v1-to-v2
binary regressions**, retaining original manifest order. Already explained
cases are reused. The v2 follow-up is capped at three selected reports and
uses only the unchanged explanation prompt, grammar and frozen classifications.
No new classification, tuning, protected evaluation or clinical claim follows.
This error-enriched sample is a diagnostic, not a new performance denominator.

## Reproduction and data boundary

```bash
PYTHONPATH=src python scripts/audit_medgemma_v2_evidence.py \
  --run-dir /governed/path/to/completed-v2 \
  --comparison-run /governed/path/to/completed-mistral-followup \
  --output-dir /governed/path/to/new-audit \
  --acknowledge-governed-output
```

The original run's scientific manifest is verified before and after inspection.
Outputs must live outside that immutable bundle; an existing audit is never
overwritten. The keyed diagnostic packet includes source reports and is
governed. Its aggregate receipt contains no report keys or text. Source code and
synthetic tests may be shared; aggregate publication remains author-reviewed.
