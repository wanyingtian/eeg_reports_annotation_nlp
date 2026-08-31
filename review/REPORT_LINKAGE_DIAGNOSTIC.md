# Local report-linkage candidate diagnostic

Frozen before inspecting the matching outputs on 2026-08-31. This is an
author-requested, post-hoc investigation of missing linkage, not a new model
evaluation or an amendment to submitted results. It uses the existing code and
governed study snapshots. It is not an author-group request or circulation item.

## Question and allowed conclusions

Can existing report keys, full text and saved embeddings recover an overlooked
link or produce credible, source-supported candidates worth checking against an
upstream patient-hash mapping? The current snapshots have no confirmed patient
key. No text similarity, model confidence, physician identity, or transitive
chain of pair matches may be substituted for patient identity.

The input is the fixed first-100 Zoe development set plus the existing 1,395
Zoe and 499 Maria evaluation reports. Only keys and text are loaded: no model
predictions or reference category labels are used in matching or selection.
All existing evaluation outputs, cohort memberships and statistics stay fixed.

## Fixed inexpensive retrieval and bounded local review

1. Verify the saved BERT cache against its ordered full-text digest and source
   database, including the six subsequently excluded source rows. Join its
   vectors to the fixed 1,994 reports by report key and exact text. No new
   embedding inference or model download. This is the historical frozen
   `bert-base-uncased` CLS representation, truncated at 512 tokens, not a
   trained patient-linkage or sentence-similarity model.
2. Compare all within- and cross-cohort unordered report pairs using that
   cosine score, full-text word TF-IDF (1--2 grams) and character TF-IDF
   (3--5 character word-boundary grams). Numerical settings are in the versioned
   `linkage_diagnostic.POLICY`. Use whole reports for both TF-IDF views; no
   clinical-label-guided feature selection. Scores are not match probabilities.
3. Retain the union of each report's top neighbour under each method and the
   top five pairs per method in each of six cohort-pair strata. Retain exact
   normalized-text/key coincidences independently. Compute five-word shingle
   overlap for candidates, with 0.6 and 0.8 descriptive thresholds; these do
   not confer duplicate-report or patient status.
4. Review the top one pair per method/stratum and one deterministic hash-selected
   comparison control per stratum, at most 24 pairs. Controls are not assumed
   different-patient pairs. Use the existing MedGemma-27B Q2_K artifact locally
   with native chat, a new explicit evidence-review prompt and JSON grammar,
   temperature 0, seed 20260831, 512 generated tokens, 4096 context, full GPU
   offload and flash attention. This is not v1/v2/v5g classification inference.
   Do not truncate overlong report pairs silently; retain skips/failures.
5. Keep source passages, raw model outputs and exact substring offsets in
   governed storage. Model judgments remain unverified hypotheses even when
   quoted passages are verbatim. Retain contrary evidence and abstentions.
   Do not retry failures to obtain a favorable result. Interruption resumes
   from completed per-pair receipts; an interrupted uncommitted call may repeat.

No patient-key training or validation set is available. Consequently this run
cannot estimate patient-linkage precision/recall, establish the absence of
patient overlap, or declare the patient-linkage gap closed by filling every
record's nearest-neighbour slot. Any promising case needs an independent source
anchor. If none is available, report that result plainly and retain report-level
uncertainty. Additional text-similarity diagnostics may help characterize shared
templates without becoming patient-clustered inference.

## Execution and handling

`scripts/run_linkage_diagnostic.py` provides prepare, match, review, finalize,
status and all phases, and a no-write dry run. Sources, cache chunks, policy,
prompt, grammar, implementation and runtime are hash-bound. Stage and per-pair
outputs are atomic, verified on resume, and kept in a dedicated ignored directory
under `data/governed/analysis-runs/`. The source runs and diagnostic respect
eclipse markers. Explicit governed-output acknowledgement is required.

No network model calls, no weight redistribution, no public keyed outputs,
no email and no new circulation PDF. The current user's instruction supplies
authority for this bounded local investigation; it does not establish missing
patient facts or publication approval. Model artifacts resolve local-only;
offline environment settings and outbound-connect blocking apply to review.
The final source-linked HTML is local private review material, not an attachment.
