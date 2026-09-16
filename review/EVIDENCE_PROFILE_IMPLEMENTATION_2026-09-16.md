# Evidence profiles: implemented compatibility contract

**Status:** implemented and verified for future runs. This does not modify any
completed output, frozen protocol or current JBHI manuscript result.

## Why two profile axes are required

The completed 2 × 2 development study showed that evidence-generation behavior
depends on both the configured model and the evidence interface. Separately,
the traceability ladder showed that locating source text does not by itself
establish that the text supports a decision.

The toolchain therefore records two independent choices:

1. **Generation profile:** what the model is asked to return and whether a
   frozen classification enters that request.
2. **Claim profile:** which audit stages are accepted and what language those
   stages authorize.

A model name never selects either profile implicitly.

## Generation profiles

| Profile | Decision supplied? | Returned structure | Lineage |
|---|---:|---|---|
| `.../decision-conditioned/v1` | yes | copied decision plus reasons | Chris's thesis classify-then-extract pathway |
| `.../independent-category/v1` | no | present, absent and qualifying passages | independent category-evidence extension |

The first profile remains the original scholarly lineage. The second is an
additional module, not its replacement.

## Claim profiles

Five claim profiles are registered: original-thesis source support, unchanged
quotation, normalized provenance, retrieval-assisted review and adjudicated
decision support. Each names its matcher stages, aggregation, suitable claim
language, prohibited claims and whether independent human adjudication is
required.

The original-thesis profile is deliberately compatible only with the
decision-conditioned generation pathway. The other provenance/review profiles
can evaluate outputs from either generation pathway. The adjudicated profile
cannot be selected without its receipt stating that human review is required.

## Enforcement

`src/eeg_review/evidence_profiles.py` is the typed registry. It rejects unknown
or incompatible selections and emits a receipt containing separate hashes for
the generation and claim profiles. The committed public-safe catalog is checked
against that registry by `make evidence-profile-verify`, so a profile cannot be
silently redefined after results have cited it.

Historical runners remain unchanged because their hashes are already bound by
completed protocols. New experiments can adopt the profile-selection receipt
without rewriting or invalidating those records.

The completed factorial study has a separate post-analysis profile map. It
binds the result receipt to the catalog and records that both evidence schemas
were assessed under the unchanged-quotation claim profile. It does not retrofit
or rewrite any historical execution receipt.
