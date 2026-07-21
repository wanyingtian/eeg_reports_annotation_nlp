# Proton Drive controlled intake workflow

The project share is available through the existing local `proton-drive` CLI.
Use this path before opening a browser session:

```text
/Users/sbergner/.local/bin/proton-drive
```

The governed remote root is:

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
