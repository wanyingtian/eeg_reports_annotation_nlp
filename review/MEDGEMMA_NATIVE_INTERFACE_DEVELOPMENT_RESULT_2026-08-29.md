# MedGemma native-interface development checkpoint

> Author-working, aggregate-only record. This is not protected-cohort evidence,
> a manuscript claim, or a replacement for the completed matched-historical Q2
> comparator.

## What ran

The single preregistered native-interface candidate ran once on the frozen
100-report Zoe development manifest. It used the exact Q2_K model artifact,
historical classification prompt and grammar, deterministic sampling, and the
GGUF's embedded chat template. Only interface serialization changed. The run
completed 100/100 unique keys in 1,724.5 seconds on Apple Metal, produced zero
invalid outputs, and yielded 16 distinct full five-label patterns.

Before any reference labels were examined, the structural rule selected the
candidate for freeze. The immutable public-safe freeze receipt is
`review/model-receipts/medgemma-native-interface-development.freeze.json`.
The complete keyed output and execution receipt remain in governed storage.

## Development-only result

After the freeze, the existing evaluator was applied to the development set and
the same 100 cases from the completed matched-historical Q2 configuration. All
differences below are native interface minus matched historical interface.

| Label | Native core accuracy (95% report-bootstrap CI) | Paired difference (95% CI) | Holm-adjusted exact McNemar p |
|---|---:|---:|---:|
| Focal epileptiform | 0.99 (0.97, 1.00) | +0.24 (+0.16, +0.32) | 0.00000119 |
| Generalized epileptiform | 0.97 (0.94, 1.00) | +0.14 (+0.07, +0.22) | 0.00415 |
| Focal non-epileptiform | 0.86 (0.78, 0.92) | +0.11 (+0.04, +0.18) | 0.0295 |
| Generalized non-epileptiform | 0.91 (0.85, 0.96) | +0.10 (+0.03, +0.18) | 0.0425 |
| Abnormality | 0.89 (0.83, 0.95) | +0.12 (+0.05, +0.20) | 0.0251 |

The descriptive development evidence is consistent with a substantial interface
effect: the model-native chat serialization reduced the matched-interface
overcalling pattern on these cases. It does not establish evaluation-cohort
performance or justify selecting a different prompt, template, quantization, or
seed. The small positive counts for the epileptiform labels also make their
estimates intrinsically imprecise despite high accuracy.

## Preserved evidence layers

1. **Independent matched-historical Q2:** completed on the full planned surface;
   unfavorable primary comparator evidence remains unchanged.
2. **Steven-led native-interface Q2 sensitivity:** completed and frozen on the
   100-report development surface only; protected evaluation is governance-
   locked.
3. **Vasily v5g:** externally reported configuration awaiting exact producing-
   bundle intake; it is not inferred from either independent configuration.

## Protected-evaluation stop

The available record confirms historical REB application H18-02728 but does not
contain the protocol/amendment scope or an authorized custodian statement that
establishes coverage for this new post-submission protected-cohort execution.
The workflow therefore stopped after development. Unlock requires either the
applicable approved-study language or written PI/data-custodian confirmation
that the frozen comparator may run on the already transferred de-identified
evaluation reports. Interpretation, manuscript placement, and aggregate release
remain separate author-group decisions.

## Traceability

- Producing code: `cc2c01dc0a019836404246098b598b6e4208f820`
- Configuration freeze: `ec5b9822125d257a92d9d2b79fed413c8bb7947a`
- Model SHA-256: `b137aac80f2bcb1c1ed35bfe13387bc496eb18898d5f46425687604f0f714481`
- Output SHA-256: `f799362d9f6a22523c0ae8240b8ebd71a4cfe7a0187e5d65bd1ec051841bf633`
- Development evaluation receipt SHA-256: `8ae13d2f878d723c637260b05c284b96077721283cae445c38443eddd4407815`
- Paired comparison receipt SHA-256: `83c5f51ea1d1f2f580527b3322f265299a1c85a0ab471df5f66a6050151d7d50`

Bootstrap intervals used 2,000 report-level replicates with seed `20260718`.
No patient key was available, so none of these intervals establishes patient
independence.
