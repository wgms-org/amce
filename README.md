# README

## Installation

```sh
mamba env create --file environment.yaml
# mamba env update --file environment.yaml --prune
conda activate ggmc
```

## Structure

- `data/` – _Not version controlled_
  - `input/`: Input data
  - `output/`: Output data
- `tests/`: Pytest test suite. Mostly empty and out of date.
- `ggmc/`
    - `functions.py`: Principal initial workflow functions
    - `helpers.py`: Helper functions
    - `kriging.py`: Kriging related functions
    - `creation.py`: Principal creation workflow functions
    - `propagation.py`: Error propagation functions
- `environment.yaml`: Conda environment file
- `ggmc_workflow.py`: Workflow script
