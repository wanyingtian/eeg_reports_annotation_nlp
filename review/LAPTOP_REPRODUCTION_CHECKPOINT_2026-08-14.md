# Laptop reproduction checkpoint - 2026-08-14

## Conclusion

The preserved study framework is executable on Steven's Apple-silicon laptop
with the exact submitted Mistral GGUF artifact. A fresh ten-report Zoe run
reproduced all 50 binary/core decisions and 49 of 50 four-level classifications
from Chris's preserved 2,000-report Mistral workbook. This is strong evidence
that the local environment can support the revision work, but it is not a claim
of byte-identical historical inference. The preserved historical predictions
remain the authoritative source for the submitted figures.

No governed report text, report identifier, or case-level result from this
validation is committed. The raw run and receipt remain under the ignored
`data/governed/` boundary.

## Input and transfer state

- A fresh read-only Proton Drive inventory on 2026-08-14 found the same fourteen
  July handover artifacts and no newer child artifact:
  - four Zoe/Maria LD/SG SQLite databases;
  - four BoW/BERT inference CSVs;
  - two processed Mistral workbooks; and
  - four historical result-analysis scripts.
- The frozen local copies still match the remote artifacts recorded in the
  controlled intake receipt.
- The exact report-level Zoe BoW/BERT prediction version that produced the
  submitted Zoe matrices has not appeared. The submitted aggregate matrices
  remain auditable, while paired Zoe baseline inference and calibration remain
  version-blocked.

## Model identity

The laptop cache already contained the submitted model; no manual download was
required:

- repository: `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`;
- file: `mistral-7b-instruct-v0.2.Q5_K_M.gguf`;
- Hugging Face snapshot: `3a6fbf4a41a1d52e415a4958cde6856d34b2db93`;
- size: 5,131,409,696 bytes; and
- SHA-256: `b85cdd596ddd76f3194047b9108a73c74d77ba04bef49255a50fc0cfbda83d32`.

The checksum was recomputed locally before the validation run. The public
receipt remains at `review/model-receipts/submitted-mistral.json`.

## Fresh bounded run

The 2026-08-14 validation used the maintained `pipeline.py`, the exact Zoe LD
database, the exact GGUF above, the recorded classification and explanation
prompts and grammars, temperature 0, top-k 40, top-p 0.95, a 4,096-token
context, and 30 GPU layers.

| Check | Result |
|---|---:|
| Reports aligned by preserved hashed report key | 10/10 |
| Binary/core classification cells matching history | 50/50 |
| Exact four-level classification cells matching history | 49/50 |
| Explanation decision cells matching history | 49/50 |
| Exact extracted reason strings matching history | 32/50 |

The single four-level difference was generalized non-epileptiform activity on
source row index 6: historical level 4 versus current level 3. Both levels are
core-positive, so the binary/core decision is unchanged. Explanation phrases
are more sensitive than the classification decisions and must not be claimed
as a byte-identical regeneration of the historical explanation workbook.

The ten reports completed in 350.49 seconds, or approximately 102.7 reports per
hour. At that observed rate, sequential classification-plus-explanation runs
would take approximately:

| Surface | Estimated time |
|---|---:|
| Maria 500 reports | 4.9 hours |
| Submitted 1,495-report Zoe surface | 14.6 hours |
| Full 2,000-report Zoe source surface | 19.5 hours |

These are scheduling estimates from a small slice, not performance guarantees.

## Aggregate reproduction state

The independent aggregate verification remains stronger than the fresh slice:

- the historical selections reproduce Zoe development N=100, Zoe evaluation
  N=1,395, and Maria evaluation N=499;
- the uploaded row-level artifacts reproduce 161/200 displayed main comparison
  cells, including every human/Mistral cell and every Maria baseline cell;
- the twenty baseline confusion matrices embedded in the paper source
  reproduce 100/100 submitted baseline core-table cells, 20/20 baseline
  certainty accuracies, and 40/40 RA-to-baseline kappa values; and
- the only row-level numerical provenance gap remains the exact producing Zoe
  baseline exports.

On 2026-08-14, `make verify` passed lint, 13 tests, and the sample audit. The
companion paper workflow again passed matrix recovery and structural
verification of both the nine-page baseline reconstruction and the initial
revision layer.

## Interpretation and next gate

The laptop is capable of running the paper's native model and of supporting a
full governed rerun. The current evidence is sufficient to begin planning the
reviewer analyses on top of the preserved historical prediction surfaces. It
is not yet sufficient to replace those surfaces with a newly generated run or
to call the current environment an exact historical replica.

Before reviewer-driven manuscript edits are merged into an authoritative
baseline, the authors should still confirm:

1. the submission-time Overleaf supplement/reference state;
2. whether the producing Zoe baseline rows can be recovered or must remain a
   disclosed historical artifact gap;
3. the stable patient-key semantics required for clustered intervals and
   leakage checks;
4. H18-02728 ethics, waiver/consent, de-identification, and secondary-use
   wording; and
5. revision leadership, clinical review ownership, and the approved execution
   environment through Vasily.

Per the current coordination direction, reproduction outside BDH is permitted;
all interaction with BDH or BDH staff must be led by Vasily. This checkpoint
requires no BDH interaction.
