# README

## Installation

```sh
mamba env create --file environment.yaml
# mamba env update --file environment.yaml --prune
conda activate ggmc
```

## Directory structure

The home directory is structured as such:

- `/data` is split into 3 subdirectories:
  - `/input`: the inputted data files for each step of `/ggmc` as copied from the NAS
  - `/output`: the outputted data files for each step *as produced* by the new workflow script (`ggmc_workflow.ipynb`) (as well as during a full pytest); i.e., all the functions currently output to locations in this directory
    - `/tests` contains the files for the pytests
        - `conftest.py` is structured to load the validate data and check against the outputted data from a workflow run
        - `test_functions.py` and `test_helpers.py` are structured as locations for the main/helper function tests, respectively
        - **Note**: in order to prioritize the efforts to provide an initial refactor all workflow scripts, tests was deprioritized.

  - `/ggmc` contains the python functions, split into:
    - `functions.py`: contains the principal functions from each step of first main part of the original workflow (`mb_data_crunching_local`)
    - `helpers.py`: the helper functions used within the main functions
    - `kriging.py`: a specific set of kriging related functions separated in the original workflow and maintained as separate functions for organization purposes
    - `creation.py`: contains the principal functions from each step of second main part of the original workflow (`version_1.5_LongPeriod_files_creation_ESSD`)
    - `propagation.py` and `propagation_ram.py`: these functions were originally split off separately in the main worklow and were maintained as such; the only difference is the `_ram.py` version has `.values` integrated as specific locations that make it necessary in the 2nd main part of the workflow (i.e., instead of debugging inputs files to work with the standard version, the `_ram.py` was maintained to work with scripts from `version_1.5_LongPeriod_files_creation_ESSD`); this makes it potential priority item for code refactoring
- `environment.yaml`: Conda environment file
- `ggmc_workflow.ipynb` is the centralized workflow script


## Refactoring Notes, Reflections, and Important Details

- At the outset, I attempted to conform to formatting requests described in the brief. After working through the first sections of the workflow, however, it became apparent that concentrating on these aspects of a refactor would hinder the more substantive goal towards a single workflow script with all functions extracted and collected in respective function files.
    - As such and after our discussions, I shifted the remaining work-time to concentrate on a refactor with the main goal of reaching a central workflow script from which a user can align inputs/outputs then work towards validation as desired.


- Here are collected notes from the process:
    - From the first 2 sections of `mb_data_crunching_local`:
        - the output 'FOG_coord_2025-01.csv' from 1_glacier_change_data passes the pytest, but the input 'FOG_coord_2025-01.csv' found in 'fog-2025-01' of '2_Kriging_spatial_anomalies' does not match this file
        - 'ba'/'bs'/'bw' and 'unc' files starting from 1_glacier_change_data
        - The 2024-01 version appeared to be the output that matches her original data_prep_spt_anomfunction. These do not match the input files in "2_Kriging_spatial_anomalies", from the in_data
    directory within the fog-2025-01 sub-directory
    - Given the disorganized structure of the repo, I made the best effort to align input files from the scripts as they were originally written in the code (i.e., using the file names leftover in the scripts as a guide for selecting the files for input at each step).
        - For each file required as an input in each step, I copied the matched files by name verbatim from the original repo to the new repo or pointed to a matching file already in the `input` directory.
        - If there were any substantive changes required to the code for the sake functionality (i.e., so that the supplied input files would run **somehow** via the original scripts), I marked the changes with `!!`. Otherwise, I attemped **not** to alter the analytical aspects of the code as much as possible so that it could be matched to its original source code as necessary during further development.
        - Similarly, although I cleaned up many of the original unnecessary comments, various comments from the original script remain in the final code (and can simply be deleted as you would like).
        - I also attempted to leave any helpful annotations and/or descriptive text (specifically function synopses), and copied these over verbatim when they existed.
        - All inputs were situated in the corresponding `inputs` directories for both main sections (`mb_data_crunching_local` and `version_1.5_LongPeriod_files_creation_ESSD`, which I labeled `Creation_Workflow` in a unified directory). If you're unsure of where an input file originates, simply search the original repo for files with matching names—this is what I was forced to do at times given the erratic nature of the file organization; the matching files I found from the original repo were the ones I copied to the new repo (attempting to match every input file to each corresponding workflow step, filling missing items as required from whatever locations I found them in the old repo, as these locations sometimes appeared incorrect according to the original organizational schema).
        - I.e., TL;DR if you don't know where an input file comes from, simply search for it by file name in the original repo to match what I did to put everything together.
    - Memory usage and Runtime:
        - Refactoring original proceeded on a VM, but section `4_Kriging_regional_mass_balance` started to show errors when debugging such that I needed to move computational environments from a VM to my local laptop for greater flexibility with memory swap from the local SSD (as well as having a much faster single clock processor speed).
        - Transferring the workflow to my local laptop allowed me to observe memory spikes in this section upwards of 30GB+. Consider this requirement carefully when running/rerunning the workflow.

## Potential Next Steps?
- Debug differences between `propagation.py` and `propagation_ram.py` according to the preferred structure of input datasets.
- Align inputs/outputs, make a full run on a system with enough RAM, then make use of the existing PyTest structure to test inputs/outputs.
