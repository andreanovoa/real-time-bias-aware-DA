# Physical models

## Summary

**Package:** [`dynamodels`](https://github.com/andreanovoa/dynamodels) — physical models live in
`dynamodels.physical` and are re-exported as `romda.models.physical`. All use `IVPIntegrator`
(`scipy.integrate.solve_ivp`) except `KS`, which uses a discrete ETDRK4 map.

Governing equations, figures and quickstart examples for each model are on the
[`dynamodels` docs site](https://andreanovoa.github.io/dynamodels/) — follow the links below.

| Class | Dim | Key parameters | Integrator | Docs |
|---|---|---|---|---|
| `VdP` | 2 | `beta`, `zeta`, `kappa`, `law`, `omega` | IVP | [Van der Pol](https://andreanovoa.github.io/dynamodels/models/van_der_pol/) |
| `Lorenz63` | 3 | `rho`, `sigma`, `beta` | IVP | [Lorenz 63](https://andreanovoa.github.io/dynamodels/models/lorenz63/) |
| `Lorenz96` | Nx | `F`, `Nx` | IVP | [Lorenz 96](https://andreanovoa.github.io/dynamodels/models/lorenz96/) |
| `KS` | Nx | `nu`, `L`, `Nx` | Discrete (ETDRK4) | [Kuramoto–Sivashinsky](https://andreanovoa.github.io/dynamodels/models/ks/) |
| `Rijke` | 2Nm+Nc | `beta`, `tau`, `C1`, `C2`, `kappa` | IVP | [Rijke tube](https://andreanovoa.github.io/dynamodels/models/rijke/) |
| `Annular` | 4 | `omega`, `nu`, `c2beta`, `kappa`, `epsilon` | IVP | [Annular combustor](https://andreanovoa.github.io/dynamodels/models/annular/) |
