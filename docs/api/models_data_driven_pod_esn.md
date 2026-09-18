# POD-ESN model

`POD_ESN` combines Proper Orthogonal Decomposition with an Echo State Network. It provides
a reduced-order forecast model for high-dimensional fields, with sensor placement available
through the projector interface.

::: romda.models.data_driven.pod_esn.POD_ESN

<figure markdown>
  ![POD-ESN data assimilation pipeline](../figs/DA/POD-ESN_DA.png){ width="700" }
  <figcaption>State estimation with a POD-ESN reduced-order model.</figcaption>
</figure>

<figure markdown>
  ![POD-ESN state and parameter estimation](../figs/DA/POD-ESN-SPE.png){ width="700" }
  <figcaption>Joint state and parameter estimation with a POD-ESN.</figcaption>
</figure>
