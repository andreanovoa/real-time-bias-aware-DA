# `romda.data_assimilation`

Ensemble filters. All filters are constructed with the observation operator `M` and
applied as callables on the augmented forecast ensemble
`Af = [φ; α; y]` (state, parameters, observables).

::: romda.data_assimilation.Filter

::: romda.data_assimilation.EnKF

::: romda.data_assimilation.EnSRKF

::: romda.data_assimilation.rBA_EnKF

## References

- Evensen (2009). *Data Assimilation: The Ensemble Kalman Filter.* Springer. — EnKF and EnSRKF.
- Nóvoa, Racca & Magri (2023). Inferring unknown unknowns: Regularized bias-aware
  ensemble Kalman filter. *Comput. Methods Appl. Mech. Eng.*, 418, 116502.
  [DOI: 10.1016/j.cma.2023.116502](https://doi.org/10.1016/j.cma.2023.116502) — r-EnKF.
- Nóvoa, Racca & Magri (2024). [**Erratum**](../2023_CMAME_Erratum.pdf) to the above —
  the implementation follows the corrected Eqs. (1a)–(1b).
