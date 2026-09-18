# Echo State Network model

`ESN_model` combines a discrete-time forecast model with the reservoir core provided by
[`echostatenetwork`](https://andreanovoa.github.io/EchoStateNetwork/). The main configuration
parameters control the reservoir size, spectral radius, input scaling, and washout.

::: romda.models.data_driven.esn.ESN_model

<figure markdown>
  ![ESN open-loop and closed-loop configurations](../figs/DA/ESN-open-close-schematics.png){ width="700" }
  <figcaption>Open-loop training and washout versus closed-loop forecasting.</figcaption>
</figure>
