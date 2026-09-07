# MedGemma focal-epileptiform error-signature audit

Status: complete read-only exploratory audit; no model inference or fitting.
The target was selected after the aggregate focal-epileptiform trade-off was
known. Seven report-language cues were frozen before their case-level results
were inspected. This is an explanatory diagnostic, not a new confirmatory
model comparison or a clinical adjudication.

## Question and value

The full held-out comparison showed a useful but clinically relevant trade-off
for Zoe focal epileptiform activity: MedGemma reduced misses from 2 to 0 while
increasing false positives from 16 to 36. This audit asks whether those
additional false positives concentrate in simple, predeclared text contexts
that would justify a narrow prompt repair.

The answer is **no simple lexical mechanism dominates**. Focal slowing is more
common among the additional MedGemma false positives than among cases both
systems correctly call negative, but occurs at nearly the same rate among
false positives shared by both systems. Negated epileptiform language and a
history/indication mentioning seizures are common throughout the negative
population and are not enriched among MedGemma-only false positives. This
does not support an automatic v3 prompt change based on one keyword rule.

## Fixed analysis

- Population: the exact 1,395 Zoe and 499 Maria held-out Reference Annotator
  report keys used by the completed comparison.
- Prediction parents: reproduced Mistral and frozen native-interface MedGemma
  v1, joined to the same report keys.
- Target: focal epileptiform activity, with levels 1--2 treated as absent and
  levels 3--4 as present.
- Method: classify every paired case into one of eight directional transition
  groups, then count seven frozen, same-segment lexical contexts.
- Privacy: only aggregate counts and proportions are public-safe. The keyed
  ledger and matched source segments remain governed.

The exact reference-key join excludes 605 prediction-only rows from the saved
2,000-row Zoe Mistral file and one prediction-only row from the saved 500-row
Maria Mistral file. No reference key is missing from either prediction parent.
This is selection by the frozen evaluation manifest, not a new exclusion.

## Paired error anatomy

| Reference/model relationship | Zoe (n=1,395) | Maria (n=499) |
|---|---:|---:|
| Both correct negative | 1,304 | 457 |
| MedGemma-only false positive | 22 | 3 |
| Mistral-only false positive | 2 | 5 |
| Both false positive | 14 | 2 |
| Both true positive | 51 | 26 |
| MedGemma-only true positive | 2 | 3 |
| Mistral-only true positive | 0 | 1 |
| Both false negative | 0 | 2 |

For Zoe, the 22 MedGemma-only false positives and 14 shared false positives
reproduce MedGemma's total of 36. The 2 Mistral-only false positives and 14
shared false positives reproduce Mistral's total of 16. MedGemma also recovers
both true positives missed by Mistral and loses none, reproducing the 0-versus-2
false-negative result. This makes the trade-off case-paired rather than merely
a comparison of marginal totals.

Maria is directionally different and much smaller: MedGemma has three
model-only false positives but corrects five Mistral-only false positives; it
recovers three Mistral-missed positives while missing one Mistral-detected
positive. The small discordant groups do not support a stable cue narrative.

## Frozen text-context results for Zoe

Percentages are within each transition group. A report may contain several
cues, so rows do not sum to 100%.

| Same-segment context | MedGemma-only FP (n=22) | Shared FP (n=14) | Both correct negative (n=1,304) |
|---|---:|---:|---:|
| Negation + epileptiform term | 10 (45.5%) | 1 (7.1%) | 903 (69.2%) |
| History/indication + seizure term | 9 (40.9%) | 9 (64.3%) | 654 (50.2%) |
| Focal term + slowing/attenuation | 6 (27.3%) | 4 (28.6%) | 162 (12.4%) |
| Focal + epileptiform wording | 2 (9.1%) | 3 (21.4%) | 322 (24.7%) |
| Generalized/bilateral + epileptiform term | 1 (4.5%) | 3 (21.4%) | 81 (6.2%) |
| Uncertain + epileptiform wording | 0 (0.0%) | 0 (0.0%) | 18 (1.4%) |
| Benign/artifact + sharp term | 0 (0.0%) | 0 (0.0%) | 4 (0.3%) |

The most plausible registered context, focal slowing/attenuation, appears in
only 6 of 22 additional MedGemma false positives. Its nearly identical rate in
the 14 shared false positives (27.3% versus 28.6%) suggests an EEG-language
ambiguity or reference-disagreement review target rather than a uniquely
MedGemma-specific lexical trigger. Conversely, simple negation is less common
among MedGemma-only false positives than among correctly negative reports.
The remaining cases therefore need source-linked clinical review; expanding a
keyword list would not establish why the configured system made the call.

## Consequence for the revision

This analysis strengthens the discussion without adding another model row:

1. The focal trade-off is real, exactly keyed and concentrated in 22 additional
   Zoe false positives, not a bookkeeping artifact.
2. The trade-off is not explained by a single obvious formatting, history or
   negation cue under the frozen rules.
3. A further evaluation-informed prompt version should not be tuned on these
   protected reports and then presented as an unbiased evaluation.
4. The next high-value step is qualified review of the already prepared,
   source-resolvable clinical packet. A future prompt hypothesis can be tested
   on new or separately reserved data if that review supports one.

The audit does not establish clinical truth, causal mechanism, patient
independence, a base-model-family effect or a general model ranking.

## Reproducibility

- Frozen plan: `model-receipts/medgemma-focal-error-signature.preregistered.json`
- Required correction: `model-receipts/medgemma-focal-error-signature.execution-amendment.json`
- Corrected producing revision: `7a89eebc12ee86dc8adcde8056e71a7285a93e3b`
- Plan SHA-256: `a27bdf3295c877fe4712a075672b554ba6db38586d15a26f08a7871f43f1b5ec`
- Aggregate SHA-256: `8f8bbe8d9edd52c007538dab87cf094c390775e64e12bb655f2ea4773ff055be`
- Governed ledger SHA-256: `f621a94d3238a49e6c25876c506109c34e51cd45755673b3e4ea6bd4631bd941`
- Run receipt SHA-256: `38a55278ddf849a79a20e221eab6ca6af52cd789f60ee3281cd661161fe423cd`

The first execution is retained at
`data/governed/analysis-runs/jbhi-focal-error-signature-20260907/` but is
invalid because it treated level 2 as binary present. The immutable amendment
records that defect. Only the corrected `-v2` output may be interpreted. The
receipt records the unrelated pre-existing `.gitignore` edit as a dirty
worktree; all inputs and outputs remain hash-bound.
