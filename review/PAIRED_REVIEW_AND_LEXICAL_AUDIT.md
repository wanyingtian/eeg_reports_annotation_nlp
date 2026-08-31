# Paired review preparation and lexical overlap audit

This is a post-hoc diagnostic of frozen outputs, not a new model configuration,
clinical adjudication, prompt-selection process or patient-linkage method.
It extends Chris's existing evaluation and evidence-review lineage. Model
predictions, cohorts and all prior favorable/unfavorable results remain unchanged.

## Frozen preparation rules (before inspecting audit outcomes)

- Use the complete 1,395 Zoe and 499 Maria native-v1 MedGemma/reference/reproduced
  Mistral key sets. Reject missing, extra, duplicate or invalid labels. Reconcile
  all repair/regression counts and model error totals with the frozen comparison.
- Per cohort and category: retain up to five MedGemma corrections, five new
  errors, five shared false negatives and five shared false positives. Add up
  to two shared-correct positive and two shared-correct negative controls.
- Deterministic SHA-256 ranking uses seed 20260718 and a saved private handle
  salt. This caps workload, not statistical inference. A report may contribute
  several category rows. Do not treat those rows as independent patients.
- Provide source-first local HTML and an editable CSV. Model/reference answers
  and saved Mistral explanation checks are initially collapsed, not formally
  blinded. No clinical interpretation is prefilled. Qualified reviewers should
  agree a rubric and record independent assessment before comparing answers.
- Keep all original Mistral explanations, including unmatched phrases and
  decision mismatches; the existing literal-source validator supplies offsets.
  MedGemma's protected run was classification-only. Do not invent, borrow from
  development, or newly generate corresponding MedGemma explanations.
- Exhaustively compare the three pairs of development/Zoe-evaluation/Maria-
  evaluation cohorts using casefolded Unicode-word five-shingle set Jaccard
  similarity >= 0.80. Require at least 20 distinct shingles in each report;
  count every skipped short-text pair. Use only an exact length-ratio upper
  bound to prune impossible matches. No approximate retrieval or label-based
  choice of threshold. The threshold is a diagnostic convention, not a
  validated classifier of duplicates, patients, semantic equivalence or leakage.
- No within-cohort scan, semantic-embedding run, clinical inference, automatic
  exclusion or recalculated primary metric. Near matches may be templates.
  Negative scans do not establish independence. A validated patient map is
  still required for patient-grouped inference.

## Execution and privacy

`scripts/prepare_comparison_review.py --help` describes the native runner.
Run `--dry-run` first. Outputs must be a dedicated ignored directory under
`data/governed/analysis-runs/`, with explicit governed-output acknowledgement.
No network, model invocation or source-run writes are used. Eclipse markers are
honored, the native 82-file bundle is checked before and after, and all directly
used inputs and implementation files are hashed. Cross-cohort scans checkpoint
separately; resumption requires unchanged inputs, policy and code. A completed
package is revalidated, not overwritten. Keep the entire directory together for
transfer; its completion receipt binds every output, including the private salt.

Case handles, source reports, keyed predictions, review worksheets and similarity
pairs remain governed. The source-first HTML is for local use only, never an
email attachment. Only an independently inspected aggregate receipt and prose
may enter the private paper workspace; no study aggregate is pushed to GitHub
by this workflow. The public collaboration surface contains code/tests/protocol.

## Remaining human inputs

Clinical co-authors define reviewer qualifications and a compact error rubric;
Chris can clarify historical annotation choices where cases expose ambiguity.
Review preparation does not require guessing either. Patient-grouped analysis
waits for a stable pseudonymous key plus validated meaning and completeness;
`Cluster code`, report hashes, textual similarity and physician identity must
not be substituted for patient identity. Existing `eeg-review compare` supports
cluster resampling once that map has been validated. Author decisions about
placement and release, and approved ethics wording, remain separate.
