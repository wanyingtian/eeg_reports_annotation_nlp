# Mistral task-adaptation implementation checkpoint

Date: 2026-08-27

## Outcome

The thesis's proposed lightweight post-hoc certainty direction is now an
executable, development-only work package. It is not a study result and has not
been admitted as a fourth evidence layer.

The implementation makes the scientific separation explicit:

- the submitted and reproduced Mistral route continues to generate the four
  levels directly and is unchanged by default;
- the proposed route makes only a binary core decision (`1 = absent`,
  `4 = present`) under a separate grammar and records both token alternatives;
- the fixed 0.5 core boundary is then mapped to the four thesis levels with one
  symmetric certainty margin per category; and
- only that margin may be selected on the first 100 Zoe RA development reports.

This is closer to the thesis's stated future direction than treating the
historical ordinal labels as if they were calibrated probabilities.

## Prespecified development sequence

1. exact reproduced historical four-level Mistral;
2. binary-core Mistral with the historical 0.1 margin; and
3. the same binary-core predictions with per-category margins selected from
   0.1, 0.2, or 0.3.

The comparison retains every candidate score. It cannot move the core boundary
or continue searching after evaluation outcomes are seen. A category requires
80 valid pairs and at least five examples on each core side; otherwise it keeps
0.1 and is explicitly marked not fitted.

## Implemented confidence machinery

- exact 100-key manifest and prediction-key equality;
- producing prediction CSV checksum and run-receipt agreement;
- exact binary-mode, model, prompt, and grammar identities;
- missing and duplicate key rejection;
- unavailable probability accounting rather than truncated renormalization;
- leave-one-report-out threshold-selection diagnostics;
- 2,000 stratified report-bootstrap threshold-stability replicates;
- descriptive Wilson intervals with an explicit development-only warning; and
- aggregate-only adapter, candidate, fit, and run receipts.

No report text, report key, patient key, or case-level prediction is emitted by
the fitter. Patient independence is not inferred from report-level resampling.

## Verification

At this checkpoint:

- all 47 repository tests passed;
- lint and sample aggregate audit passed;
- both historical four-level and binary-core fake-model runs passed end to end;
- binary probability extraction fails closed unless both 1 and 4 alternatives
  are present;
- the historical route remains the default, and its classification prompt is
  byte-identical to commit `b8d3f01` (SHA-256
  `52198221d8330e9857b51a7ad99b017aa18836e1718b08dd0ae355820f5a5e69`); and
- the public plan validates as `ready_for_implementation: true` and
  `ready_for_evaluation: false`.

No Zoe or Maria development/evaluation inference was run for this work package,
and no MedGemma prediction or aggregate was used for selection.

## Author decision gates before real compute

The author group should confirm, before a development run:

1. that the binary-core interface is the intended operationalization of the
   thesis's proposed binary-token-probability route;
2. that the historical margin grid and the stated sparse-support fallback are
   acceptable;
3. the immutable first-100 Zoe RA manifest identity; and
4. that the three-stage development sequence is complete enough to freeze
   without open-ended prompt search.

After those confirmations, create a governed execution copy of the plan, add
the manifest path and checksum, run only the 100-report development surface,
review all favorable and unfavorable diagnostics, and freeze the adapter before
any protected evaluation outcome is inspected.
