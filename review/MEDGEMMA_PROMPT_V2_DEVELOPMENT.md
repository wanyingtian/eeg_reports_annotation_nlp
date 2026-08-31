# MedGemma native prompt v2: bounded focal-disambiguation development

The completed v2 remains unchanged. The separately prepared
[v2.1 upgrade](MEDGEMMA_V21_UPGRADE_PREPARATION.md) uses `--variant v21`, a new
run directory and its own frozen development rule; it has not run yet.

This is a **new, post-submission prompt version** within Chris's original
classification and evidence-extraction framework. It does not renumber Chris's
historical V1/V2 work or implement Vasily's external v5g. Steven authorized the
bounded local experiment in chat on 2026-08-30. No author-order decision is
implied, and no protected-cohort inference is included.

## Change and provenance

`historical-submitted` remains the pipeline default, byte-for-byte unchanged.
`medgemma-native-focal-disambiguation-v2` inserts four contextual clarifications
only in the focal-epileptiform definition. It distinguishes an observed focal
epileptiform finding from history/indication, sharp morphology alone, focal
non-epileptiform findings, and generalized discharges with regional prominence.
Explicit focal evidence remains admissible; these are not keyword bans.

The inherited prompt, examples, remaining definitions, confidence instructions,
native chat template, JSON grammar, model bytes, sampling, and runtime profile
otherwise remain unchanged. No weights are trained and no external inference
service is contacted.

The hypothesis comes from the completed native comparator's error pattern and
the focal-exclusion ideas in Vasily's supplied prompt-comparison report. Those
ideas are acknowledged, not claimed as newly invented. Chris's prompt and
versioned development method remain the framework parent; Steven's new version
and receipts record this implementation separately.

## Frozen workload and decision rule

The machine-readable plan is
`model-receipts/medgemma-native-focal-v2.development-plan.json`.

1. Generate one candidate on the exact original 100-report Zoe development
   manifest, comparing against the preserved native-MedGemma v1 predictions.
2. Describe all five labels, confusion counts, precision/recall/F1, binary and
   exact-four-level agreement, report-level intervals, latency and tokens.
3. Mark the narrow development rule met only if focal false positives decrease,
   focal false negatives do not increase, and no other label gains binary
   errors. Report the available positive/negative support; this small set
   cannot establish equivalence or reliable population-level safety.
4. Run Chris's existing evidence prompt and grammar on **the first 20 manifest
   positions**, with v2 classifications held fixed. Inspect copied decisions,
   empty reasons, fallback strings and exact text matches. This is not a paired
   MedGemma-v1 explanation comparison and does not select the prompt.
5. Stop after these 120 planned record-stage outputs. Preserve negative,
   unchanged, invalid and favorable outcomes. Interrupted uncommitted calls
   may be repeated on recovery; completed rows are never intentionally rerun.

The 100-case rule is frozen before this version's generation. Nevertheless,
**prior protected evaluation results informed the hypothesis**. Returning to
the old development set does not erase that knowledge. Any later rerun on the
same evaluation cohort must be labelled posthoc sensitivity, not a fresh
independent confirmation. This job cannot launch such a rerun automatically.

## Operation

Use the existing pinned local Python environment with `PYTHONPATH=src`. The
runner reuses the repository's `Stage`/`Supervisor` execution and the original
classification pipeline, with a named prompt option. It does not create a
parallel inference implementation.

```bash
python scripts/medgemma_prompt_v2.py prepare \
  --source-run /governed/path/to/native-development-v1 \
  --run-dir /governed/path/to/new-prompt-v2 \
  --public-status /private/path/to/prompt-v2-status.json
python scripts/medgemma_prompt_v2.py dry-run --run-dir /governed/path/to/new-prompt-v2
python scripts/medgemma_prompt_v2.py smoke --run-dir /governed/path/to/new-prompt-v2
python scripts/medgemma_prompt_v2.py launch --run-dir /governed/path/to/new-prompt-v2
python scripts/medgemma_prompt_v2.py status --run-dir /governed/path/to/new-prompt-v2
```

The live smoke computes the first three planned cases, stores their receipt,
and resumes those same outputs toward 100. Synthetic tests precede it. The
macOS LaunchAgent continues after chat completion and publishes aggregate-only
progress. `pause`/`resume` operate at checkpoints; a forced shutdown may repeat
the one uncommitted case. Runtime/source/input drift and scientific failures
stop for review rather than mixing configurations. `watch` can also be run
directly on an authorized host. A host/runtime migration requires an explicit
receipt; do not silently edit `job.json` to bypass its identity checks.

The self-contained run bundle includes inputs, prompts, predictions, analysis,
source/runtime identities and hashes. The final scientific manifest excludes
mutable launcher logs and state; these remain operational records. All report
keys, report text and extracted passages stay governed. No manuscript, PDF,
email or public performance claim is automatically produced.
