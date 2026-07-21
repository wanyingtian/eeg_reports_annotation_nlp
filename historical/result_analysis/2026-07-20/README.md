# Historical result-analysis source — 2026-07-20 intake

These four Python files are an exact, byte-preserving copy of the result-analysis
code Wanying (Chris) Tian uploaded to the private `EEG-Vasily` Proton Drive
share on 2026-07-20. They are retained as historical evidence and are not the
canonical revision pipeline.

| File | SHA-256 |
|---|---|
| `analysis_functions.py` | `d9ed3a1c6dee34e53c82106df487847ba70a317cede784f310be49e01c2ff141` |
| `main.py` | `34f51b1ff0c4fa221f28b2155e915c4054eb1340d9e4a419fdedde8abf0284f1` |
| `plotting.py` | `f9ef5cb74132fb25de499e0943a49c41373cf8017e67cbc4d94528a78a9f5bec` |
| `result_preprocessing.py` | `43efd3d99f59388f0b79210be5472c4d8a2c3f8cf32856b27e40dbdd4b2d4688` |

The files contain no embedded report rows, report text, identifiers, credentials,
or model outputs. Their hard-coded paths point to files that are not included in
Git.

## What the source establishes

- `clean_ground_truth_by_index_range` applies one or more positional Python
  half-open slices, concatenates them, and then removes rows with a missing value
  in any of `Hashed ID`, `Report`, or the five four-level annotations.
- `align_model_with_ground_truth` selects predictions by `Hashed ID` and
  reindexes them into the retained reference order.
- `process_all_files` joins the two human-annotation databases, processed
  Mistral classifications, and baseline CSVs at the analysis layer. It does not
  alter a source database or historical model output.
- `main.py` is a ten-report, clean-repository adaptation with placeholder paths
  and `(0, 10)` ranges. It is not the frozen invocation that generated the
  submitted manuscript figures.

## Known execution hazards in the historical code

These findings are preserved as audit observations; the historical files above
must not be edited to fix them.

1. `main.py` calls `compute_kappa(models, core=True)` before
   `compute_kappa(models, core=False)`. The core conversion mutates the model
   data frames in place, so the later nominal four-level calculation may receive
   already-collapsed binary values.
2. Several paths, output directories, model names, and both author branches in
   `main.py` are placeholders for the ten-report public sample.
3. The chi-square helper assumes every contingency table is exactly 2x2; sparse
   categories require an explicit policy rather than relabeling a smaller table.
4. Plotting creates a relative output directory at import time, which makes
   results depend on the caller's working directory.
5. Exceptions while loading model files are printed and suppressed. A revision
   run must fail closed when an expected model input is absent or misaligned.

The maintained revision workflow belongs under `src/eeg_review/`, with explicit
inputs, immutable checksums, aggregate-only outputs, tests, and run manifests.
This snapshot remains the provenance bridge for validating that workflow.
