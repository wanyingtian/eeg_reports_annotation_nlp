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

After a bounded follow-up using `run_fixed_classification_explanations.py`,
repeat the audit into a **new directory** with
`--targeted-evidence /governed/path/to/targeted/evidence.csv`. This verifies the
same model, prompt, grammar, runtime, offline environment and fixed predictions.
The original first-20 aggregate remains separate from the error-enriched
follow-up aggregate; the review packet may use both without pooling denominators.

## Existing-rule trace and category consistency

Add `--alignment-diagnostics` to that same audit command to apply
`saved-evidence-alignment-diagnostic-v1`. Use the pinned runtime with the
repository's baseline dependencies; the inherited alignment module imports
those packages, but this diagnostic does **not** instantiate any model.

This calls Chris's original `classify_reason_rule` unchanged, on the same
semicolon-joined reason-cell unit as `process_output.py`. It records the rule
branch: explicit abnormal wording, negative pattern, positive pattern, or an
unmatched string defaulting to negative. Matches of both polarities are exposed
without changing the original priority. A rule agreement is not validation:
the fallback can look like normal-supporting evidence without recognizing any
finding, and a negated subtype is not necessarily evidence of overall normality.
The thesis selected the **learned ClinicalBERT polarity classifier**, not this
rule heuristic; the diagnostic does not claim to reproduce that score.

Both explicit historical category-consistency instructions are checked against
all saved development classifications: any positive subtype with negative
overall abnormality, and all negative subtypes with positive overall abnormality.
These are instruction checks, not independently validated clinical laws. There
is no majority vote or automatic label repair: such a repair can erase a correct
overall call when an individual subtype was missed.

For source phrases still unmatched by the original audit, a separate optional
diagnostic searches after eliding only double-asterisk tokens from the source
and collapsing whitespace. It keeps case, numbers, words, negation and other
punctuation intact, returns exact source offsets, and leaves literal acceptance
unchanged. A candidate does not establish that these tokens are merely formatting
or that the passage entails the model's label. Phrase instances and distinct
report/phrase pairs are counted separately.

The fixed first-20 and error-enriched additional sample keep separate summaries.
The extended keyed output is `governed-alignment-diagnostics.json`; only its
aggregate counts enter the safe receipt. Source files are hash-bound and the
immutable study manifest is rechecked after assembly. No historical classifier,
prediction, prompt, sample denominator or factuality score is replaced.

## Next-version direction

Historical direction at the v2 diagnostic checkpoint (subsequently implemented
as the separately frozen v2.1; see the read-only interpretation below):

The next useful experiment is a **category-scoped evidence audit**, not another
focal keyword exclusion. Preserve the existing native interface, JSON grammar,
four-level labels and original comparison rows. A separately named audit should
ask which source passages support, qualify or contradict each category, without
requiring the model to justify a supplied decision. Compare those passages with
the already frozen decisions afterward. This tests the limitation of the current
decision-conditioned explanation question; it does not claim access to causal
reasoning or allow an auditing model to silently relabel cases.

Before generation, freeze the new task text, output schema, development keys,
complete-outcome reporting and bounded call budget. Use synthetic checks first;
the same 100 development cases remain available for exploratory work. No new
classification prompt has been selected, and no additional protected-cohort run
follows from these diagnostics. Any later classification change must retain
v1/v2 results and carry its own version and frozen development decision rule.

## Read-only v2.1 interpretation

The completed v2.1 run retains all three versions and the independent first-20
audit. The next operation is analysis of saved data, not new inference:

```bash
PYTHONPATH=src python scripts/audit_medgemma_v21.py \
  --run-dir /governed/path/to/completed-v21 \
  --output-dir /governed/path/to/new-interpretation \
  --acknowledge-governed-output
```

The script validates the complete producing manifest before and after reading,
rejects missing/duplicate/reordered keys and missing parents, and reproduces
all frozen confusion counts, paired repair/regression counts, exact four-level
changes and literal phrase counters. It exposes case exchanges hidden by
unchanged confusion totals. It checks both inherited category constraints
without forcing predictions to obey them.

Every independent phrase retains its category and role, original text,
source hash, literal offsets or explicitly unverified whitespace candidates.
Empty and invalid records retain their denominators; absent audits are not
imputed. A positive-evidence list with a negative classification is a review
flag, not automatically an error: normal-state qualifications or conflicting
source statements can explain it. Conversely, source-present text can have an
incorrectly assigned role. No audit output becomes a vote or a replacement
label, and no conditional-v2 versus independent-v2.1 quality ranking is made.

`governed-review-packet.json` contains report text and keys and must remain in
authorized storage. `interpretation-summary.json` contains aggregate counts and
receipt hashes only; publication remains a separate author-reviewed step.
The complete inference bundle is never modified by this tool.
