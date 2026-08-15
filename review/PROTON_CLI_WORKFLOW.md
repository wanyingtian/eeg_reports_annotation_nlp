# Proton Drive controlled private handover workflow

The private handover share between Steven and Chris is available through the
existing local `proton-drive` CLI. It is not a project-wide resource and is not
Vasily's research store. Use this path before opening a browser session:

```text
/Users/sbergner/.local/bin/proton-drive
```

The historically named private remote root is:

```text
/my-files/EEG-Vasily
```

## Read-only inventory

The general `-j` option follows the subcommand rather than preceding it:

```bash
proton-drive filesystem list -j /my-files/EEG-Vasily
proton-drive filesystem list -j /my-files/EEG-Vasily/data
proton-drive filesystem list -j /my-files/EEG-Vasily/baseline_annotation_results
proton-drive filesystem list -j /my-files/EEG-Vasily/mistral_annotations_results
proton-drive filesystem list -j /my-files/EEG-Vasily/result_analysis_code
```

The JSON inventory exposes the remote filename, claimed size, claimed
modification time, and claimed SHA-1 digest. Do not print file contents.

Before any upload, verify the root sharing status. The intended boundary is
Steven as owner and Chris as the only editor, with editor re-sharing disabled
and no public link. Do not infer authorization for another author or system
from the folder name.

## Controlled download

Downloads belong only under the ignored `data/governed/` tree. Set restrictive
permissions first and use an explicit destination; never download governed
material into the repository root or another tracked path.

```bash
umask 077
proton-drive filesystem download \
  /my-files/EEG-Vasily/result_analysis_code \
  /absolute/path/to/repository/data/governed/proton-YYYY-MM-DD
```

After transfer, compare local file size and digest with the remote claimed
metadata. Preserve a private intake receipt under the governed destination.
Only aggregate, non-identifying provenance conclusions may enter Git.

## Controlled return of compute products

Return bundles belong in a new dated child folder. Prefer the complete resumable
run plus its per-file transfer manifest over a hand-selected output subset.
Include a readme, an exact code archive, a top-level checksum file, and any
author-facing PDF. Do not duplicate public model weights when a pinned snapshot,
size, and SHA-256 are sufficient.

```bash
proton-drive filesystem create-folder \
  /my-files/EEG-Vasily jbhi-native-reproduction-YYYYMMDD

proton-drive filesystem upload -t \
  /governed/export/README.md \
  /governed/export/SHA256SUMS \
  /governed/export/code-revision.tar.gz \
  /governed/export/discussion-supplement.pdf \
  /governed/path/to/resumable-run \
  /my-files/EEG-Vasily/jbhi-native-reproduction-YYYYMMDD
```

After upload, list the remote folder and stage-receipt directory, compare
claimed sizes/digests, and round-trip download into a new ignored governed
directory. Verify the top-level `SHA256SUMS` and every file named by the run's
`transfer-manifest.json`. A successful upload message alone is not a content
integrity receipt.

## Current verified state

On 2026-07-20 after the Monday handover, the CLI inventory and the frozen local
intake matched byte-for-byte for fourteen artifacts: four annotator databases,
four baseline inference CSVs, two Mistral result workbooks, and four historical
analysis scripts. The last scientific uploads were the four analysis scripts.
No later child artifact was present at the time of revalidation.

This receipt establishes transfer completeness only. It does not establish
that the uploaded Zoe baseline CSVs are the exact report-level artifacts that
produced the submitted Zoe matrices; that version-provenance question remains
open.

On 2026-08-15 the completed laptop reproduction was returned under
`/my-files/EEG-Vasily/jbhi-native-reproduction-20260815`. The upload contained
445 items (22.86 MiB), including the complete 40-stage resumable run, exact code
archive at `8fcc465`, updated author PDF, readme, and checksums. A full
round-trip download verified all four top-level SHA-256 entries and all 380
files in the run transfer manifest. See
[`PROTON_RETURN_RECEIPT_2026-08-15.md`](PROTON_RETURN_RECEIPT_2026-08-15.md).
