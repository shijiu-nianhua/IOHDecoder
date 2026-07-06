# IOHDecoder Code Overview

This document explains the role of each source file under `release/iohdecoder/`.
The release package keeps only the paper method: HMF-style historical encoding,
future-query decoding, lower-tail Event-KL training, teacher-cache loading, and
clinical evaluation.

## Top-level Files

### `__init__.py`

Package metadata for `iohdecoder`.

- Defines the release package version.
- Does not contain model or training logic.

### `engine.py`

Training and evaluation orchestration.

Main responsibilities:

- set random seeds and resolve the target device;
- build the `FutureQueryIOHDecoder` model from a YAML config;
- create train/validation/test DataLoaders;
- build the MSE + Event-KL objective;
- load split-aligned teacher predictions during training;
- run the training loop, validation loop, checkpoint saving, and test evaluation;
- write predictions and metrics to the configured result directory.

Important entry points:

- `build_model(config)`
- `train_experiment(config_path)`
- `evaluate_experiment(config_path)`

## `data/`

### `data/__init__.py`

Exports the dataset and DataLoader helpers.

Public exports:

- `IOHDataset`
- `get_ioh_loader`
- `get_ioh_loaders`

### `data/dataloader.py`

Dataset parsing and DataLoader construction.

Main responsibilities:

- read preprocessed split pickle files;
- parse MAP/SBP dynamic history signals and future MAP targets;
- apply training-set normalization statistics;
- convert variable or serialized sequence values into fixed-length arrays;
- provide batches with the keys expected by the model:
  - `index`
  - `x_dyn`
  - `y_true`
- construct PyTorch DataLoaders for `train`, `val`, and `test` splits.

The paper protocol uses ART_MBP and ART_SBP from `x_dyn`.

## `model/`

### `model/__init__.py`

Public model exports.

Public exports:

- `HMFEncoder`
- `FutureQueryDecoderConfig`
- `FutureQueryIOHDecoder`

### `model/hmf_encoder.py`

HMF-style historical encoder used by IOHDecoder.

Main components:

- `MovingAverageDecomposition`: separates the normalized history into seasonal
  and trend components using moving-average decomposition.
- `HMFPatchEncoder`: embeds history patches and encodes them with Transformer
  layers.
- `HMFEncoder`: combines seasonal and trend patch encoders to produce
  historical memory tokens for the future-query decoder.

This file intentionally keeps only the encoder portion needed by IOHDecoder.
The original HMF shared forecasting head is not included in the release method.

### `model/future_query.py`

The main IOHDecoder model.

Main components:

- `FutureQueryDecoderConfig`: dataclass holding common model hyperparameters.
- `FutureQueryIOHDecoder`: HMF encoder plus future-query cross-attention
  decoder.

Model flow:

1. Select MAP/SBP channels from `batch["x_dyn"]`.
2. Encode the historical sequence with `HMFEncoder`.
3. Create one learnable query for each future horizon.
4. Let future queries attend to the HMF memory.
5. Predict future MAP residuals anchored at the last observed MAP.

At inference time, this file is the core deployed model. It does not load
Chronos-2 or teacher-cache files.

## `losses/`

### `losses/__init__.py`

Exports the release training objective.

Public exports:

- `TAEDConfig`
- `TeacherAdvantagedEventDistillationLoss`

### `losses/taed.py`

MSE plus lower-tail event-space distillation.

Despite the historical class name, the release keeps the paper's main Event-KL
path rather than the older diagnostic variants.

Main responsibilities:

- compute the MAP regression loss, usually MSE;
- convert student and teacher future MAP trajectories back to raw mmHg scale;
- project trajectories into a soft sustained-IOH event trace;
- compute Bernoulli KL between teacher q10 event trace and student event trace;
- return a dictionary of total loss and logging terms used by `engine.py`.

The teacher prediction is used only during training. If no teacher prediction is
provided, the loss falls back to the regression term.

## `metrics/`

### `metrics/__init__.py`

Exports the clinical evaluator.

Public export:

- `ClinicalEvaluator`

### `metrics/clinical.py`

Regression and sustained-IOH event metrics.

Main responsibilities:

- compute MSE, MAE, and RMSE on future MAP trajectories;
- convert each predicted trajectory into a sustained-hypotension event decision;
- compute event-level AUROC, F1, Accuracy, Recall, Precision, and Specificity;
- provide NumPy fallbacks for common sklearn metrics when `scikit-learn` is not
  installed.

The event rule is based on the same threshold-duration definition used in the
paper.

## `teachers/`

### `teachers/__init__.py`

Exports the teacher-cache utility.

Public export:

- `SplitTeacherCache`

### `teachers/cache.py`

Split-aligned teacher prediction cache loader.

Main responsibilities:

- locate teacher prediction arrays for `train`, `val`, and `test` splits;
- support common cache layouts such as `split/pred_q10.npy` or
  `split_pred_q10.npy`;
- optionally align predictions by dataset indices from `indices.npy`;
- return teacher predictions for the current batch indices as PyTorch tensors.

This file does not run Chronos-2. It only reads teacher forecasts that have
already been generated offline.

## `utils/`

### `utils/__init__.py`

Exports common YAML and path helpers.

Public exports:

- `load_yaml_config`
- `project_abs_path`
- `resolve_dataset_cache_paths`

### `utils/io.py`

Small IO helpers used across the release package.

Main responsibilities:

- read YAML config files;
- resolve relative paths against the project root;
- resolve the preprocessed data cache path and normalization statistics path
  from a data config.

## Typical Runtime Flow

The main training/evaluation path connects the files as follows:

1. `engine.py` reads an experiment config through `utils/io.py`.
2. `data/dataloader.py` loads split data and yields model batches.
3. `model/future_query.py` builds predictions using `model/hmf_encoder.py`.
4. `teachers/cache.py` provides offline q10 teacher forecasts during training.
5. `losses/taed.py` combines MAP regression with Event-KL distillation.
6. `metrics/clinical.py` evaluates both MAP regression and sustained-IOH events.
