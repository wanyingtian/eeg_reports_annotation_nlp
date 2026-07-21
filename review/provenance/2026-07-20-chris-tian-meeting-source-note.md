# Historical execution and analysis layout — Chris Tian source note

## Record status

- Date: 2026-07-20
- Source: Wanying (Chris) Tian, speaking during a live meeting with Steven
  Bergner
- Recorder: contemporaneous summary prepared from the meeting transcript
- Evidentiary role: primary historical-process account, separate from the
  verification findings below
- Data boundary: this note contains no report text, report-level identifier, or
  patient information

This note preserves Chris's description of how the historical study materials
were arranged and used. It is not a claim that every file has already been
recovered or independently verified.

## Chris's account

### Published repository versus historical analysis workspace

The clean repository was prepared as a researcher-facing implementation of the
annotation pipeline. It included a small, schema-compatible report database so
that the pipeline could be exercised without releasing the governed study
corpora. It was not intended to reproduce the paper's figures from the released
sample, and the historical result-analysis and plotting code was not included.

The historical EEG-facility workspace had a different, work-in-progress folder
layout. Chris described three main material classes:

1. a `data` area containing databases with human annotations;
2. processed Mistral annotation results; and
3. processed baseline-model annotation results.

The result-analysis code was tailored to that historical folder layout and was
used to produce the paper's plots and tables. Chris intends to recover and
upload it. She cautioned that it may require path and layout adaptations before
it can run against the clean repository.

### Inference coverage and evaluation selection

The models did not inspect a human-annotation flag before inference. They tried
to annotate the reports in the requested order. Only the reports for which
human reference annotations were available were subsequently used in result
analysis.

Historical result analysis selected rows positionally. Chris recalled code
using slices such as `[0:500]` and `[1000:2000]`, rather than a stored
include/exclude column or a list of hashed report identifiers. These selection
slices belong to the analysis layer; they should not be written back into or
used to alter the authoritative databases or historical output files.

Chris's email and meeting account identify the Zoe selection as source rows
`[0:500]` plus `[1000:2000]`. Five selected cases were unusable because they
were corrupted, lacked annotation, or did not yield a readable LLM
explanation, leaving 1,495 annotated/usable reports. The first 100 reports were
the development set, leaving 1,395 reports for Zoe evaluation.

For Maria, the historical selection was `[0:500]`; one selected case was
unusable, leaving 499 evaluation reports.

### Cohorts and annotator copies

The LD and SG database files for a report-author cohort contain the same
reports but different human annotation columns. A model therefore needs one
inference pass for the Zoe reports and one for the Maria reports, not a separate
inference pass for each annotator copy.

Chris confirmed the historical role mapping:

- LD: Reference Annotator (RA)
- SG: Second Annotator (SA)

The report-author/cohort role (Zoe or Maria) and the human-annotator role (LD or
SG) are distinct and must remain separate in code, filenames, and manuscript
provenance.

### Preservation request

Chris did not recommend changing table headers or adding include/exclude fields
to the historical source files. The source databases and processed outputs work
within their historical schema and should be preserved exactly. Compatibility
work belongs in an adapter or analysis layer.

## Independent verification completed on the 2026-07-20 intake

The following checks were performed without printing report text or identifiers
and without modifying any source file:

- The two Maria baseline CSVs each contain 500 rows and have identical report
  identifier/text order.
- The two Zoe baseline CSVs each contain 1,900 rows and have identical report
  identifier/text order.
- Maria baseline rows match the 500-row Mistral classification workbook in the
  same order.
- Zoe baseline rows match Mistral classification rows `[100:2000]` exactly in
  identifier/text order. This explains the 1,900-row baseline exports: the
  first 100 Zoe development reports are absent from the external-evaluation
  inference outputs.
- Within the Zoe source slices, all 100 development rows have complete five-
  category annotations from both LD and SG. Of 1,400 evaluation candidates,
  exactly 1,395 have complete annotations from both annotators. The remaining
  five consist of three rows with no LD labels and complete SG labels, one with
  partial LD labels and complete SG labels, and one with neither annotator's
  labels. This reproduces the paper's development/evaluation counts without
  altering either database.
- The 500-row unselected middle Zoe block contains 499 complete LD rows but only
  41 complete SG rows. Those 41 additional SG annotations, offset by the one
  selected row lacking SG labels, explain why the full SG database contains
  1,540 complete rows despite its historical `1500` filename.
- Within Maria `[0:500]`, 499 rows have complete annotations from both LD and
  SG, and one has neither annotator's labels.
- For all four baseline CSVs and all five categories, the exported four-level
  class exactly follows the recorded probability and `epsilon = 0.1` rule:
  class 1 below 0.4, class 2 from 0.4 to below 0.5, class 3 from 0.5 to below
  0.6, and class 4 at or above 0.6.

The row-coordinate translation is therefore:

| Analysis surface | Historical Zoe evaluation candidates before unusable-case removal |
|---|---|
| 2,000-row Mistral output / source selection | `[100:500]` plus `[1000:2000]` |
| 1,900-row baseline output corresponding to source `[100:2000]` | `[0:400]` plus `[900:1900]` |

Both expressions select 1,400 candidate evaluation reports before removal of
the five unusable cases. This translation is a derived verification result; it
does not replace the pending historical result-analysis code as the
authoritative account of exclusion handling.

The observed annotation-completeness rule reconstructs exactly 1,395 Zoe
evaluation cases and 499 Maria cases. It is therefore a strong candidate for
the historical exclusion implementation, but it must remain labelled as a
reconstruction until the result-analysis source confirms whether readability
or corruption checks were also applied explicitly.

## Result-analysis source recovered after the meeting

Chris subsequently uploaded four Python files: `main.py`,
`result_preprocessing.py`, `analysis_functions.py`, and `plotting.py`. Exact
copies and their SHA-256 checksums are preserved under
`historical/result_analysis/2026-07-20/`.

The recovered `clean_ground_truth_by_index_range` implementation confirms the
selection mechanism described in the meeting. It extracts the seven columns
`Hashed ID`, `Report`, and the five four-level labels with positional half-open
slices, concatenates the slices, and calls `dropna()` across those columns.
`align_model_with_ground_truth` then filters and reorders each model output by
the retained `Hashed ID` values. This reproduces the observed 1,495 Zoe and 499
Maria annotation-complete totals without writing selection state into any
source file.

The uploaded `main.py` is a clean-repository/sample adaptation, not the exact
historical invocation: both author branches point to the Zoe ten-report sample,
use `(0, 10)`, and contain placeholder relative paths. The source therefore
confirms the selection and alignment functions, while the meeting/email account
still supplies the full-study ranges. A maintained adapter must set those paths
and coordinate systems explicitly and validate every expected row count.

An audit also found that the historical core-label conversion mutates model
data frames in place before a later nominal four-level kappa calculation. The
submitted results must be checked against immutable copies before that pathway
is relied upon; the original source remains unchanged.

## Remaining historical evidence to recover unchanged

- the exact full-study invocation/configuration and any notebooks not included
  in the recovered four-file result-analysis snapshot;
- any separate logic that classified the failure reason for the five Zoe and
  one Maria rows (the recovered selection function excludes them through
  annotation-completeness `dropna()`);
- the authoritative Mistral sheet/processed artifact used where classification
  and explanation exports disagree;
- raw Mistral outputs, prompts, grammars, run configuration, logs, and model
  receipt;
- baseline training artifacts, fold assignments, fold-level scores, random
  seeds, fitted estimators/vectorizer, and any threshold-development outputs;
  and
- any paper-table or figure intermediate files.

When recovered, these materials should first be frozen under governed storage
with checksums. Necessary path/schema changes should be implemented as a
documented adapter or copied analysis layer, never by cleaning or overwriting
the recovered originals.
