# Resumable study-compute workflow

This workflow reproduces the native study computations in a governed run
directory. It preserves the submitted-study outputs as the historical source
of record while generating a new, fully receipted execution on current
hardware. The run can be stopped between or during stages, resumed without
repeating completed work, and moved to a Linux/NVIDIA host through an approved
governed transfer channel.

The source material is the controlled thesis compute-environment snapshot. It
contains report text and pseudonymous case identifiers and must not be committed
to Git or transferred through an unapproved channel.

## Run layout

The supervisor creates one self-contained directory:

```text
<run>/
  inputs/       canonical, hashed input snapshots and the pinned model receipt
  products/     raw inference, processed predictions, baselines, and analyses
  cache/        restartable BERT embedding chunks
  logs/         one append-only log per stage and the supervisor log
  stages/       completion receipts with output SHA-256 values
  job.json      immutable study selections, source receipts, and privacy note
  state.json    current stage and stage status
```

The default local run used for the August 2026 reproduction is:

```text
data/governed/study-runs/jbhi-native-20260814
```

Everything below `data/governed/` is ignored by Git.

## Initialize and verify

From the repository root, install the complete environment and initialize the
run once:

```sh
uv sync --extra reports --extra baselines --extra llm
uv run --extra reports --extra baselines --extra llm python scripts/study_job.py init \
  --run-dir data/governed/study-runs/jbhi-native-20260814
```

Initialization copies only the selected columns and rows into canonical SQLite
and CSV snapshots. Each input receives a SHA-256 receipt in `job.json`. The Zoe
development set is rows `[0:100]`; the Zoe evaluation candidates are
`[100:500] + [1000:2000]`; and the Maria evaluation candidates are `[0:500]`.
Eligibility exclusions are applied downstream by identifier alignment, making
the observed 1,395- and 499-report analysis surfaces explicit rather than
silently deleting source rows.

Run the quick deterministic stages through the contemporary BoW calibration:

```sh
uv run --extra reports --extra baselines --extra llm python scripts/study_job.py run \
  --run-dir data/governed/study-runs/jbhi-native-20260814 \
  --stop-after baseline_bow_calibrate_maria
```

Each completed stage is skipped on every later invocation only when its required
files still match the hashes in its stage receipt.

## Launch, inspect, stop, and resume

Launch the remaining work independently of the terminal or Codex task:

```sh
uv run --extra reports --extra baselines --extra llm python scripts/study_job.py launch \
  --run-dir data/governed/study-runs/jbhi-native-20260814
```

On macOS the launcher places the supervisor in a new session and wraps it in
`caffeinate`, preventing system or disk sleep while the job is active. Closing
the terminal or finishing a Codex task does not stop it. Normal power loss,
reboot, or process termination leaves the latest atomic checkpoints intact.

Inspect progress at any time:

```sh
make study-status RUN_DIR=data/governed/study-runs/jbhi-native-20260814
```

For the LLM stages, status reports completed rows, mean observed inference time,
and an updated ETA. Detailed output is available with `status --json`; stage
logs are under `<run>/logs/`.

Stop cleanly before moving the run or shutting down the machine:

```sh
uv run python scripts/study_job.py stop \
  --run-dir data/governed/study-runs/jbhi-native-20260814
```

The LLM result is flushed atomically after every completed report, so at most
the report currently in progress is repeated. BERT embeddings are written as
validated batch chunks. Run `launch` again to resume from the last valid stage,
row, or embedding batch.

## Computed stages

The job covers all currently executable planned products:

1. cohort completeness and cross-cohort overlap audits;
2. historical Mistral evaluation, paired second-annotator comparisons, and
   bootstrap intervals;
3. historical Maria BoW/BERT evaluation and calibration;
4. fresh five-fold BoW+LR development, Zoe/Maria inference, evaluation, and
   calibration;
5. fresh 2,000-report Zoe and 500-report Maria Mistral inference, processing,
   evaluation, historical comparison, second-annotator comparison, and governed
   clinical error-review samples; and
6. fresh five-fold BERT+LR development with resumable embeddings, Zoe/Maria
   inference, evaluation, and calibration.

Five-fold out-of-fold estimates are emitted only when both core classes contain
at least five development examples. If support is smaller, the receipt reports
the limitation and no out-of-fold value is invented; a final model is still fit
on all valid development reports for the separately identified external-cohort
analysis.

The exact producing Zoe baseline row exports remain absent from the handover.
The submitted aggregate matrices and the available historical Zoe exports are
retained as evidence, but the workflow does not mislabel the latter as the
producing submitted rows.

## Transfer to Linux/NVIDIA

Stop the source job, then create a transfer inventory:

```sh
uv run python scripts/study_job.py manifest \
  --run-dir data/governed/study-runs/jbhi-native-20260814
```

Copy the entire run directory through an approved governed channel, preserving
relative paths. Verify every file against `transfer-manifest.json` before
resuming. Clone the exact repository revision recorded in `job.json`. Model
weights are not included in the run directory; the pinned receipt identifies
the exact public artifact and its SHA-256.

On a Linux/NVIDIA host, install `uv`, a compatible CUDA toolchain, and build the
LLM runtime with CUDA enabled:

```sh
CMAKE_ARGS="-DGGML_CUDA=on" uv sync \
  --extra reports --extra baselines --extra llm \
  --reinstall-package llama-cpp-python
```

Then run `launch` against the transferred directory. The launcher omits
`caffeinate` on Linux but otherwise uses the same detached supervisor and stage
receipts. If the Mac and Linux executions are intended as an implementation
comparison, initialize a separate run directory on Linux rather than reusing
Mac BERT caches or raw model output.

Compare independent raw Mistral outputs without exporting report text:

```sh
uv run python scripts/study_job.py compare \
  --left /approved/path/to/mac-run \
  --right /approved/path/to/linux-run \
  --output /approved/path/to/platform-comparison.json
```

The comparison reports shared rows, exact four-level classifications, binary
core decisions, and exact explanation-reason lists. It does not treat expected
cross-backend numerical variation as a change to the historical source of
record.
