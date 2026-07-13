# `romda.tools`

Dimensionality-reduction building blocks and the reservoir-computing core. These are
pure tools (no forecaster); combine them with the models in
[`romda.models.data_driven`](models.md) to build reduced-order models.

## Projector interface

::: romda.tools.autoencoders.Projector

## POD and spectral POD

::: romda.tools.autoencoders.POD

::: romda.tools.autoencoders.SPOD

## Functional API

::: romda.tools.pod_spod.snapshot_pod

::: romda.tools.pod_spod.snapshot_pod_randomized

::: romda.tools.pod_spod.spod_sieber

::: romda.tools.pod_spod.spod_towne

## Echo State Network core

::: romda.tools.esn_core.EchoStateNetwork

## References

- Sirovich (1987). Turbulence and the dynamics of coherent structures.
  *Quart. Appl. Math.*, XLV(3), 561–590 — snapshot POD.
- Halko, Martinsson & Tropp (2011). Finding structure with randomness.
  *SIAM Review*, 53(2), 217–288 — randomized solver.
- Sieber, Paschereit & Oberleithner (2016). Spectral proper orthogonal decomposition.
  *J. Fluid Mech.*, 792, 798–828 — filtered-correlation SPOD.
- Towne, Schmidt & Colonius (2018). Spectral proper orthogonal decomposition and its
  relationship to dynamic mode decomposition and resolvent analysis.
  *J. Fluid Mech.*, 847, 821–867 — Welch-CSD SPOD.
- Mendez et al. (2023). *Data-Driven Fluid Mechanics.* Cambridge University Press —
  notation conventions.
- Racca & Magri (2021). Robust optimization and validation of echo state networks for
  learning chaotic dynamics. *Neural Networks*, 142, 252–268 — ESN training and
  validation strategy.
