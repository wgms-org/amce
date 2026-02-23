# README

## Installation

```sh
mamba env create --file environment.yaml
# mamba env update --file environment.yaml --prune
conda activate amce
```

## Structure

- `data/` – _Not version controlled_
  - `input/`: Input data
  - `output/`: Output data
- `tests/`: Pytest test suite. Mostly empty and out of date.
- `amce/`
- `environment.yaml`: Conda environment file
- `workflow.py`: Workflow script
