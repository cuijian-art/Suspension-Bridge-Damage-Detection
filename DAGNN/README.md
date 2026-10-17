# DAGNN Cross-Excitation Transfer Learning

This directory contains the code for vibration-response transfer learning across three excitation domains of a suspension bridge:

1. `HammerExcitation`
2. `ShakerExcitation`
3. `MovingTrolleyExcitation`

## Files

| File | Purpose |
|---|---|
| `train.py` | Trains one directed source-to-target transfer task |
| `train_cross_excitation.py` | Trains all six directed transfer tasks among the three excitation domains |
| `train_ablation.py` | Runs the full model and four ablation variants |
| `data_loader.py` | Loads vibration signals, creates training and validation splits, and applies normalization |
| `models.py` | Defines the one-dimensional CNN, instance graph, multi-receptive-field graph network, and domain discriminator |
| `losses.py` | Defines the multi-kernel maximum mean discrepancy loss |
| `trainer.py` | Implements training, validation, best-checkpoint selection, and result export |
| `plotting.py` | Generates training curves, confusion matrices, and t-SNE figures |
| `cli.py` | Defines shared command-line arguments and logging configuration |

## Data Organization

All three excitation domains must contain the same damage-state subdirectories. The class names and number of classes are inferred automatically from the directory names.

```text
data/
├── HammerExcitation/
│   ├── DamageState01/
│   │   ├── sample_01.mat
│   │   └── sample_02.mat
│   └── DamageState02/
├── ShakerExcitation/
│   ├── DamageState01/
│   └── DamageState02/
└── MovingTrolleyExcitation/
    ├── DamageState01/
    └── DamageState02/
```

Supported file formats are `.mat`, `.npy`, `.npz`, `.csv`, and `.txt`. Continuous signals are divided into samples according to `window_size` and `stride`.

## Single Transfer Task

```powershell
python train.py `
  --data_dir ./data `
  --source HammerExcitation `
  --target ShakerExcitation
```

The `--source` and `--target` arguments can be set to any two different excitation domains.

## Six Directed Transfer Tasks

```powershell
python train_cross_excitation.py --data_dir ./data
```

The following tasks are executed sequentially:

- `HammerExcitation → ShakerExcitation`
- `HammerExcitation → MovingTrolleyExcitation`
- `ShakerExcitation → HammerExcitation`
- `ShakerExcitation → MovingTrolleyExcitation`
- `MovingTrolleyExcitation → HammerExcitation`
- `MovingTrolleyExcitation → ShakerExcitation`

## Ablation Study

```powershell
python train_ablation.py `
  --data_dir ./data `
  --source HammerExcitation `
  --target MovingTrolleyExcitation
```

The default ablation study includes:

- `Full DAGNN`
- `w/o ADV`
- `w/o MMD`
- `w/o GCN`
- `w/o ADV & MMD`

To run selected variants only:

```powershell
python train_ablation.py --data_dir ./data --only full_dagnn without_adv
```

## Main Arguments

| Argument | Default | Description |
|---|---:|---|
| `--window_size` | `1024` | Number of time points in each vibration sample |
| `--stride` | `1024` | Step size between consecutive signal windows |
| `--normalization` | `mean-std` | Sample-wise normalization method |
| `--signal_key` | Empty | Signal variable name in a `.mat` or `.npz` file; a numeric array is selected automatically when omitted |
| `--channel_index` | `0` | Sensor-channel index selected from a two-dimensional array |
| `--validation_ratio` | `0.2` | Proportion assigned to validation |
| `--batch_size` | `64` | Mini-batch size |
| `--epochs` | `50` | Number of training epochs |
| `--warmup_epochs` | `1` | Number of source-only classification epochs |
| `--learning_rate` | `0.001` | Initial learning rate |
| `--lambda_adv` | `1.0` | Weight of the domain-adversarial loss |
| `--lambda_mmd` | `1.0` | Weight of the MMD loss |
| `--edge_dropout` | `0.5` | Graph-edge dropout probability during training |
| `--seed` | `42` | Model-training random seed |
| `--split_seed` | `42` | Data-splitting random seed |
| `--device` | `cuda:0` | Training device |

Display all available arguments with:

```powershell
python train.py --help
python train_cross_excitation.py --help
python train_ablation.py --help
```

## Outputs

Outputs are stored in `./checkpoint` by default. Each task has an independent directory containing:

- `best_model.pth`: model weights from the highest target-domain validation accuracy
- `configuration.json`: training configuration
- `summary.json`: task summary and best accuracy
- `training_history.csv`: epoch-level training metrics
- Training curves, adaptation curves, confusion matrix, and t-SNE figures

Use `--output_dir` to select another output location.
