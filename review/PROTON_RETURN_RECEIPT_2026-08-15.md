# Private Proton return receipt - 2026-08-15

## Boundary

This receipt covers the private Proton Drive share between Steven Bergner and
Wanying (Chris) Tian. The remote root's historical name is `EEG-Vasily`, but the
share is not a project-wide resource and is not Vasily Vakorin's research
store. At verification time Chris was the only editor, editor re-sharing was
disabled, there were no pending invitations, and no public share was enabled.

The returned run contains governed report inputs, pseudonymous identifiers,
report-derived outputs, caches, and clinical-error preparation. It must not be
shared onward merely because another person is an author or collaborator.

## Remote destination

```text
/my-files/EEG-Vasily/jbhi-native-reproduction-20260815
```

The upload completed with 445 items and 22.86 MiB. The destination contains:

- `jbhi-native-20260814/`: complete resumable run, including 40 stage receipts
  and the 380-file transfer manifest;
- `eeg_reports_annotation_nlp-8fcc465.tar.gz`: exact public-safe code snapshot;
- `jbhi-revision-discussion-supplement-2026-08-15.pdf`;
- `README.md`; and
- `SHA256SUMS`.

The 5.13 GB public Mistral GGUF is not duplicated. The readme identifies its
registry, snapshot, filename, byte size, and SHA-256.

## Top-level SHA-256 receipt

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `README.md` | 3,536 | `5fbf4b2b358ad7040d11a96780190a682a47648fbeaf42241f13721b51deef7d` |
| `eeg_reports_annotation_nlp-8fcc465.tar.gz` | 957,067 | `8b17f11b3c9ffd5bc80a54da771c6a751b6c95bb284cc0785755c5c69a161aed` |
| `jbhi-revision-discussion-supplement-2026-08-15.pdf` | 275,954 | `07cb0964788fb5d70a7304b3ca82e0270083587eb197658db7d5da203fe6bb8b` |
| `jbhi-native-20260814/transfer-manifest.json` | 94,381 | `78d55b303694810e849d6e07e5e0366604cb40be30441397846b6b6b09fb6191` |

## Round-trip verification

The complete remote folder was downloaded into a new restrictive-permission,
Git-ignored governed directory. Verification results were:

- all four top-level SHA-256 entries: pass;
- all 380 files named by `transfer-manifest.json`: pass;
- missing/mismatched run files: zero;
- remote stage receipts: 40;
- round-tripped `state.json`: `completed`, 40 stages.

Remote claimed SHA-1 values for the readme, checksum file, code archive, and PDF
also matched the local files. The round-trip SHA-256 verification is the
stronger content-integrity check.

## Reproduction use

Chris can download the folder, verify the checksums, inspect the completed
products, and preserve it on her authorized system. The code archive supplies
the exact final toolchain revision. The original job-start revision remains in
`job.json`; the run documentation explains the reconciled revision history.

For a platform comparison, initialize a separate Linux/NVIDIA run rather than
overwriting the completed macOS directory. The returned bundle is a private
scientific handover, not a public artifact release.
