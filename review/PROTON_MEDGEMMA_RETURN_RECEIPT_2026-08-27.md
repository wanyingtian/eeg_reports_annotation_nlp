# Private Proton MedGemma feasibility return receipt - 2026-08-27

## Boundary

This receipt covers the private Proton Drive share between Steven Bergner and
Wanying (Chris) Tian. The remote root's historical name is `EEG-Vasily`, but the
share is not a project-wide resource and is not Vasily Vakorin's research
store. At verification time Chris remained its only editor, editor re-sharing
was disabled, and there were no pending invitations.

## Remote destination

```text
/my-files/EEG-Vasily/jbhi-medgemma-feasibility-20260827
```

Seven files were uploaded (9.43 KiB total): a readme, a top-level checksum
file, Q2 preload/runtime/classification receipts, and Q4 preload/runtime-attempt
receipts. They contain no EEG report text or source identifier. Public model
weights were not duplicated.

## Artifact identities

- Q2_K: 10,503,437,312 bytes, SHA-256
  `b137aac80f2bcb1c1ed35bfe13387bc496eb18898d5f46425687604f0f714481`.
- Q4_K_S: 15,673,773,056 bytes, SHA-256
  `1ad12d20c9e2ef61f74c0e952de589c93cb3dce17750f1fbfe0db4921616a5b1`.
- Distribution revision:
  `unsloth/medgemma-27b-text-it-GGUF@334fbf6811c963d223f6ac107a459347353f068d`.
- Code revision: `75e18bb93637d72faff1859e07c9afae507af9f1` on
  `review/jbhi-02463-revision-toolchain`.

## Round-trip verification

The complete remote folder was downloaded into a new restrictive-permission
temporary directory. Every artifact named by `SHA256SUMS` passed SHA-256
verification. The temporary verification copy was then moved to the local
Trash; the governed source receipts remain in the repo-adjacent submission
area.

## Interpretation

Q2 downloaded, loaded, obeyed its embedded chat template plus a trivial GBNF
constraint, and completed a one-report classification-only compatibility test
against the preserved submitted prompt and grammar. That single row matched
all five reference decisions; it is a runtime smoke test, not a study estimate.

Q4 downloaded and passed checksum verification. It loaded on the 24 GB Mac,
but the tiny inference probe left insufficient memory headroom and was stopped
without retaining output. Q2 is the safe laptop development path; Q4 should be
run on a roomier Linux/NVIDIA host.

Neither artifact is yet identified as Vasily's exact producing file, and the
receipts do not reproduce v5g. Exact comparison still requires the producing
prompt, grammar, wrapper, runtime receipt, keyed predictions, and cohort
manifests.
