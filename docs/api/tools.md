# Tools (POD / SPOD / ESN core)

## Summary

Building blocks reused across the data-driven models: `EchoStateNetwork`
(the reservoir-computing engine mixed into `ESN_model`) comes from the external
[`echostatenetwork`](https://github.com/andreanovoa/EchoStateNetwork) package
(re-exported by `romda.models.data_driven`); the `Projector` hierarchy
(`POD`/`SPOD`) and the standalone decomposition functions (`pod_utils.py`)
live in `src/models/data_driven/autoencoders/` and are documented below.
Import everything from `romda.models.data_driven` (or its `autoencoders` subpackage).

------

::: echostatenetwork.EchoStateNetwork

<figure markdown>
  ![ESN open-loop and closed-loop configurations](../figs/DA/ESN-open-close-schematics.png){ width="700" }
  <figcaption>Open-loop (training/washout) vs. closed-loop (forecasting) reservoir
  configurations.</figcaption>
</figure>

## Modal decompositions

::: romda.models.data_driven.autoencoders.Projector

::: romda.models.data_driven.autoencoders.POD

::: romda.models.data_driven.autoencoders.SPOD

::: romda.models.data_driven.autoencoders.pod_utils.snapshot_pod

::: romda.models.data_driven.autoencoders.pod_utils.snapshot_pod_randomized

::: romda.models.data_driven.autoencoders.pod_utils.spod_sieber

::: romda.models.data_driven.autoencoders.pod_utils.spod_towne

::: romda.models.data_driven.autoencoders.pod_utils.spod_towne_reconstruct

::: romda.models.data_driven.autoencoders.pod_utils.print_spod_towne_summary

::: romda.models.data_driven.autoencoders.pod_utils.energy_fraction

