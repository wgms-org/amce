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
- `publish/`: Final output data. _Not version controlled_
- `tests/`: Pytest test suite. Mostly empty and out of date.
- `amce/`
  - `functions.py`: Principal initial workflow functions
  - `helpers.py`: Helper functions
  - `kriging.py`: Kriging related functions
  - `creation.py`: Principal creation workflow functions
  - `propagation.py`: Error propagation functions
  - `publish.py`: Produce final data and figures
- `environment.yaml`: Conda environment file
- `workflow.py`: Workflow script
- `constants.py`: Workflow constants
