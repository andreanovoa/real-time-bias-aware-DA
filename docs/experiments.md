# Running physical experiments

The public repository includes physical-model and thermoacoustic data-assimilation
experiments. Their YAML configurations are in `configs/twin_da/` and `configs/tai_da/`.
Run them from the repository root after installing `romda` in editable mode:

```bash
pip install -e ".[dev,notebooks]" --use-pep517
```

## Run a configuration

Use `romda.experiments.run_experiment` from Python:

```python
from romda.experiments import run_experiment

filter_ensemble, truth, metrics = run_experiment(
    "configs/tai_da/rijke.yml",
    seed=1,
)
print(metrics)
```

The same runner is available from the command line:

```bash
python -m romda.experiments configs/twin_da/lorenz63.yml --seed 3
python -m romda.experiments configs/tai_da/rijke.yml --seed 1
python -m romda.experiments configs/tai_da/annular.yml --set truth.ER=0.5375
```

The `--seed` option redraws the initial ensemble while keeping the physical truth fixed.
Use `--set key=value` for temporary dotted-key overrides. Numbers, booleans, lists, and
`null` are parsed as JSON. The final output line contains the experiment metrics as JSON.

The annular case downloads its experimental data on first use. Generated data and results
are written under `results/`.

## Available cases

| Configuration | Experiment |
|---|---|
| `configs/twin_da/lorenz63.yml` | Lorenz 63 twin |
| `configs/twin_da/lorenz96.yml` | Lorenz 96 twin |
| `configs/tai_da/vdp.yml` | Van der Pol twin |
| `configs/tai_da/rijke.yml` | Rijke-tube twin |
| `configs/tai_da/annular.yml` | Annular experimental data |

The Python API returns the filtered ensemble, truth observations, and a metrics dictionary,
which can be used to create custom plots or further analysis.
