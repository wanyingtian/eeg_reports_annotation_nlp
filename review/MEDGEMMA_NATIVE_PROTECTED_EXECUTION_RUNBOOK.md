# MedGemma native-interface protected evaluation runbook

## Purpose and evidence boundary

This runbook prepares and executes the protected evaluation stage of the separately
preregistered MedGemma native-interface sensitivity. It does not alter or replace the
completed matched-historical Q2 comparator. The native configuration was selected once
on the frozen 100-report Zoe development set using a result-blind structural rule. It is
not reselected on Zoe or Maria evaluation outcomes.

The protected stage is intentionally dormant until a documentary authorization receipt
passes the fail-closed validator. Authorship, possession of de-identified data, and an
intention to revise the manuscript are not treated as substitutes for the applicable
approved-study record or written confirmation by the principal investigator or an
authorized data custodian.

## Frozen public controls

- Scientific plan:
  `review/model-receipts/medgemma-native-protected-comparator.preregistered.json`
- Tier plan:
  `review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json`
- Authorization template:
  `review/model-receipts/medgemma-native-protected-authorization.template.json`
- Result-blind development freeze:
  `review/model-receipts/medgemma-native-interface-development.freeze.json`
- Development result:
  `review/model-receipts/medgemma-native-interface-development.result.json`
- Promoted Apple Metal runtime:
  `review/model-receipts/medgemma-metal-runtime-amendment.promoted.json`

The plan fixes the Q2_K model artifact, native GGUF chat template, task-message bytes,
historical prompt and grammar, deterministic sampling, and the exact 1,395-report Zoe
and 499-report Maria populations. The development run is a frozen parent and is not run
again as part of protected evaluation.

## What resolves the execution gate

Populate a copy of the authorization template in governed administrative storage. The
receipt records only the source and scope of the confirmation; it does not attempt to
interpret legal or ethics sufficiency. It must identify one of these sources:

1. an applicable approved-study record;
2. written confirmation by the principal investigator; or
3. written confirmation by an authorized data custodian.

The confirmation must explicitly cover the already transferred de-identified reports,
the new post-submission model inference and aggregate analysis, secondary use, and the
two exact cohorts. Keyed outputs remain governed. Patient-grouped inference remains
unavailable unless a stable patient key and its semantics are separately confirmed.

Validate the receipt without touching study data:

```bash
python scripts/check_medgemma_native_protected_authorization.py \
  --authorization "$AUTHORIZATION_RECEIPT" \
  --output "$PUBLIC_SAFE_UNLOCK_RECEIPT"
```

A pending or incomplete receipt exits nonzero and lists exact blockers.

## Prepare the governed, transferable run

The preparation command validates authorization before resolving either the source or
destination run path. It then selects exact complete-reference keys, copies the three
paired comparator surfaces, writes native-chat commands, and records hashes in a
self-contained governed bundle.

```bash
python scripts/prepare_medgemma_study.py \
  --plan review/model-receipts/medgemma-native-protected-comparator.preregistered.json \
  --source-run "$GOVERNED_SOURCE_RUN" \
  --output-dir "$GOVERNED_NATIVE_RUN" \
  --runtime-amendment review/model-receipts/medgemma-metal-runtime-amendment.promoted.json \
  --authorization "$AUTHORIZATION_RECEIPT" \
  --acknowledge-governed-output
```

The output contains report-level data, pseudonymous keys, and predictions. Transfer it
only through an approved governed channel. Preserve relative paths and validate every
entry in `transfer-manifest.json`. Model weights are not included or redistributed.

## Result-blind dry run and launch

Dry-run validation checks the plan, authorization binding, source-plan hash, exact cohort
counts, input and manifest hashes, native-chat command identity, runtime amendment, and a
clean repository revision.

```bash
python scripts/run_tiered_medgemma_study.py dry-run \
  --run-dir "$GOVERNED_NATIVE_RUN" \
  --tier-plan review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json \
  --authorization "$AUTHORIZATION_RECEIPT" \
  --public-status-output "$PUBLIC_SAFE_STATUS"
```

Launch only after the dry run succeeds:

```bash
python scripts/run_tiered_medgemma_study.py launch \
  --run-dir "$GOVERNED_NATIVE_RUN" \
  --tier-plan review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json \
  --authorization "$AUTHORIZATION_RECEIPT" \
  --public-status-output "$PUBLIC_SAFE_STATUS"
```

The supervisor checkpoints every report and can be interrupted and relaunched with the
same command. On macOS it runs under the system's sleep-prevention service. A launch
receipt binds the clean repository revision, tier-plan hash, and authorization-receipt
hash. The public-safe status contains only operational counts, validity/degeneracy
signals, timing, tokens, and an updated estimate; it contains no report text or key,
reference label, keyed prediction, or partial performance metric.

## Mission control and unattended recovery

Mission control now preserves the same authorization receipt through status refreshes,
detached watching, and its one permitted orphan-recovery attempt. Adopt the live runner
and then launch the watcher:

```bash
python scripts/medgemma_mission_control.py adopt \
  --run-dir "$GOVERNED_NATIVE_RUN" \
  --compute-repo "$COMPUTE_REPOSITORY" \
  --tier-plan review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json \
  --authorization "$AUTHORIZATION_RECEIPT" \
  --public-status "$PUBLIC_SAFE_STATUS" \
  --public-heartbeat "$PUBLIC_SAFE_HEARTBEAT"

python scripts/medgemma_mission_control.py launch-watch \
  --run-dir "$GOVERNED_NATIVE_RUN" \
  --compute-repo "$COMPUTE_REPOSITORY" \
  --tier-plan review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json \
  --authorization "$AUTHORIZATION_RECEIPT" \
  --public-status "$PUBLIC_SAFE_STATUS" \
  --public-heartbeat "$PUBLIC_SAFE_HEARTBEAT"
```

Mission control does not restart a failed or deliberately stopped study. It recovers only
a run whose state remains `running` after its matching supervisor disappears, and only
within the configured attempt limit. On completion it writes a final governed transfer
manifest binding the authorization hash and every retained artifact.

## Schedule and stopping rules

The frozen 100-report development run observed a mean of 17.52 seconds per report. At
that rate, 1,894 protected reports require about 9.2 hours of inference on the recorded
24 GB Apple-silicon system, plus preparation and final analysis. The first cross-cohort
operational view is expected after about 22 minutes (50 Zoe and 25 Maria reports).

Later tiers reach approximately 2.1, 4.6, 7.2, and 9.2 cumulative hours. No partial
accuracy or agreement statistic is computed. An invalid structured output, duplicate
key, or single-pattern degeneracy at the first gate stops progression. Such a stop may
support an operational repair, but it cannot support a result-driven prompt, template,
quantization, seed, or cohort change.

After exact final key coverage, the runner processes both cohorts and performs the frozen
same-case evaluations against submitted Mistral, reproduced Mistral, and the second
annotator. It retains null, unfavorable, invalid, and unfinished outcomes. Aggregate
interpretation, manuscript placement, and release remain author-group decisions.

## Produce the author-review evidence candidate

After mission control records completion and writes `final-transfer-manifest.json`, create
the public-safe aggregate candidate:

```bash
python scripts/finalize_medgemma_native_protected_result.py \
  --run-dir "$GOVERNED_NATIVE_RUN" \
  --study-plan review/model-receipts/medgemma-native-protected-comparator.preregistered.json \
  --tier-plan review/model-receipts/medgemma-native-protected-tiered-execution.preregistered.json \
  --authorization "$AUTHORIZATION_RECEIPT" \
  --output "$PUBLIC_SAFE_RESULT_CANDIDATE"
```

The finalizer validates authorization before opening the run, then requires exact final
coverage, zero invalid or duplicate outputs, the frozen model direction, 2,000
report-bootstrap replicates with seed `20260718`, Holm adjustment, exact source hashes,
and complete final-manifest coverage. It emits all 90 prespecified label/effect claim rows
across both cohorts and three comparators. Directional language is derived only from the
paired interval and is labeled as an author-review candidate. The output contains no case
content or identifiers and does not admit itself to the manuscript.
