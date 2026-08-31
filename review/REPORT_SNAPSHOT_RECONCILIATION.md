# Report reconciliation is not patient linkage

Two distinct questions must stay separate:

1. **Is this the same report in two exports?** Join on the existing report hash
   and require exact source text. The supported `Hashed ID` and
   `Hashed_ReportURN` names are aliases, not different identifier namespaces.
   Missing keys, duplicates and changed text must remain visible. Do not use
   row position, silently normalize text or fall back to a nearest neighbour.
2. **Did two different reports come from the same patient?** A report-level join
   does not answer this. Use an independently documented, stable patient key
   linked to each report key. Neither semantic similarity nor a perfect
   report-export join creates that information.

`scripts/audit_report_snapshot_join.py` verifies the repository example and the
prepared study cohorts against both original annotator copies. It reads only
report keys and text, inventories table/column names, checks unique non-null
keys, opens SQLite read-only, and binds the audit to file and implementation
hashes. It rejects changed receipts and reports aggregate counts only. The
completed receipt remains under the ignored governed diagnostic directory.

`scripts/audit_linkage_anchors.py` separately checks for opaque hex/UUID tokens
and explicitly labelled patient keys in those formats. It does not search for
names, dates of birth, health numbers or external identities. Even a repeated
token is only a candidate until its semantics are confirmed. No findings means
only that these specific formats were absent, not that every possible patient
link has been disproved.

Both audits operate on the verified preparation stage of the
[bounded matching diagnostic](REPORT_LINKAGE_DIAGNOSTIC.md), without changing
that frozen experiment, its cohort or its outputs. Example commands:

```bash
.venv/bin/python scripts/audit_report_snapshot_join.py \
  --run-dir data/governed/analysis-runs/<diagnostic> \
  --snapshot-dir data/governed/<author-snapshot>/data \
  --acknowledge-governed-output
.venv/bin/python scripts/audit_linkage_anchors.py \
  --run-dir data/governed/analysis-runs/<diagnostic> \
  --acknowledge-governed-output
```

Start any remaining historical question with the people who produced or hold
the original export: was its patient hash retained separately, and is it stable
across report authors and development/evaluation sets? A two-column governed
report-to-patient map plus its provenance may suffice. No patient names or
identifying clinical details are needed. Involve an upstream custodian only if
that export must actually be retrieved there.

Once validated, the existing paired cluster-bootstrap analysis can use the
saved predictions without another model run. Report-level comparisons remain
usable in the meantime, with their repeated-patient uncertainty limitation.
Do not reinterpret unique reports as unique patients or alter historical
splits silently if a subsequently recovered map reveals patient overlap.
