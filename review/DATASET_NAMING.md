# Dataset naming and role semantics

This note records what the clean pipeline actually does. It prevents cohort,
report-author, annotator, and filesystem names from being conflated during
reproduction.

## Current pipeline contract

- `--dataset-path` selects the SQLite database that is opened.
- `--dataset-id` is a run label. If omitted, it defaults to the database
  filename without its extension.
- `--model` independently selects the model registry entry.
- `--num-reports` independently controls the requested total number of report
  rows, after excluding report hashes already present in a resumed output.

The dataset ID is used to discover prior runs and construct output names:

```text
raw_{dataset_id}_{model}_{num_reports}_v{version}_run{run}.csv
config_{dataset_id}_{model}_v{version}.json
```

Consequently, `zoe`, `maria`, and `sample` have no hidden loading semantics in
the current code. Renaming a database changes the default label but not its
contents. Supplying an explicit stable `--dataset-id` avoids accidental naming
changes and makes resume behavior predictable.

## Historical `--author` behavior

Before commit `8a46ccb` (26 August 2025), the CLI exposed
`--author {zoe,maria}` with help text `Report author`. That value selected one
of two separately configured database paths and was embedded in result
filenames. The commit replaced this coupled switch with the general
`--dataset-path` / `--dataset-id` contract.

Thus Chris's recollection is supported by the Git history: `author` originally
meant the neurologist/report-author cohort selector, not the RA/SA or LD/SG
annotation identity. The precise human-role mapping still belongs in the
annotation/provenance record and should be confirmed by Chris and the clinical
team.

## Reproduction rules

1. Do not rename or edit an authoritative database. Freeze it and record its
   checksum.
2. Use an explicit cohort/split ID, for example `zoe-development`,
   `zoe-evaluation`, or `maria-evaluation`.
3. Use a fresh output directory for the exact submitted reproduction so the
   interactive resume/version logic cannot mix conditions.
4. Preserve database row order or recover the original selection manifest.
   The historical SQLite query has no explicit `ORDER BY`, so a filename and
   row count alone do not prove that the same first N reports were processed.
5. Record report-author, reference annotator, secondary annotator, and clinical
   reviewer as separate provenance fields; do not encode those roles only in a
   filename.
