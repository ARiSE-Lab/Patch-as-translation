# Patching as Translation: The Data and the Metaphor ([ASE'20](https://conf.researchr.org/details/ase-2020/ase-2020-papers/51/Patching-as-Translation-The-Data-and-the-Metaphor))

This repository is for the replication materials and data of ASE'20 paper: "Patching as translation: The Data and the Metaphor". See the [paper](https://arxiv.org/abs/2008.10707) for details 

The code is divided into two parts: Models and Analysis. Analysis contains the code for the results in section 4.1 & 4.2. Models contains the code for section 4.3 and section 5. Please refer to the README.md in each directory for instructions.

## Data
  - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3830095.svg)](https://doi.org/10.5281/zenodo.3830095)

## Reproducing Analysis part
1. Download the data.
2. Open ```code/real-fix-analysis.py``` and check line 284-307. The comment indicates how to reproduce the paper results for each section.
3. Run the script
```
python3 code/real-fix-analysis.py
```

## Reproducing Model part
All hyper-parameter configurations can be set in `config.yml`, which currently contains the default settings used in the papers. You can set the context and/or edit enhancements there too, under `data` -> `add_context` and `edits`.

### Training
To train our models from scratch, run `train_model.py` with the (provided) data file and vocabulary as arguments. This will run through the specified number of epochs (see `config.yml`) and validate the model's performance on held-out data every epoch, after which it writes both a log entry (in `log.txt`) and stores the latest model (under `models/`). Use the optional `-s|--suffix` flag to specify a descriptive name for the log and model locations, such as `-s edits-context`.

During training, the model periodically (every `print_freq` minibatches, see `config.yml`) prints its metrics to track progress. Once every 5 print steps, it produces an example, which consists of the buggy line and the predicted (teacher-forced) repair tokens. When edit-based repairing, the buggy line is annotated with the real pointers (`^` for start and `$` for end) and the predicted ones (`>` and `<` respectively). When validating, the model additionally beam-searches on every sample and prints/logs the corresponding top-K accuracy at the end. It may be worthwhile to reduce the config's `beam_size` just for training (from the default 25 to e.g. 5) to speed up the held-out pass.

### Testing
To generate the top-K (beam searched) patches for all test data, run `evaluate_model.py`, again with the same parameters. This runner creates an output file, named `results.txt`, again with optional suffix, with all the produced patches; it is particularly useful because it translates the model-produced edit (in terms of pointer and tokens) combined with the bug into the corresponding patch. For convenience, it prints the top-generated patch to console while producing patches.

The results file is formatted as follows: each bug is preceded by a whiteline, followed by the tab-separated tokens of that bug. Then, the `beam_size` top patches follow, each starting with the probability of that patch, as a percentage rounded to two decimals, followed by `: ` and then the tab-separated patch tokens.
