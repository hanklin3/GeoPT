# 2D Reynolds Cavitation Experiment Log

## Overview
Complete training and evaluation pipeline for 2D Reynolds cavitation contact-plane dynamics using Transolver architecture with per-node velocity features.

---

## Commands Executed

### 1. Full Dataset Generation
```bash
cd /home/hanklin/GeoPT
bash ./scripts/synthetic/generate_reynolds_cavitation_2d.sh
```

**Script Location**: `scripts/synthetic/generate_reynolds_cavitation_2d.sh`

**Output Folder**: `data/reynolds_cavitation_2d_npys/`

### 2. Small Preview Dataset Generation
```bash
cd /home/hanklin/GeoPT
bash ./scripts/synthetic/preview_reynolds_cavitation_2d.sh
```

**Script Location**: `scripts/synthetic/preview_reynolds_cavitation_2d.sh`

**Output Folder**: `data/reynolds_cavitation_2d_preview/`

### 3. Full Training (200 epochs)
```bash
cd /home/hanklin/GeoPT
source ~/anaconda3/etc/profile.d/conda.sh
conda activate geoPt
bash ./scripts/synthetic/train_reynolds_cavitation_2d.sh
```

**Script Location**: `scripts/synthetic/train_reynolds_cavitation_2d.sh`

**Configuration**:
- Training samples: 64
- Test samples: 16
To generate the trained-model preview outputs for the test set, run:

```bash
cd /home/hanklin/GeoPT
bash ./scripts/synthetic/test_reynolds_cavitation_2d_preview.sh
```

- Epochs: 200
- Space dimension: 3 (x, z, h coordinates)
2. **Review preview and inference outputs**: `results/reynolds_cavitation_2d_transolver/test_full_previews/plots/` contains the 10 preview PNGs (`cross_section_*.png`, `phase_map_*.png`, `raw_pressure_*.png`, `density_*.png`, `velocity_gradient_*.png`, `flow_*.png`, `top_down_pressure_*.png`). The metrics live in `results/reynolds_cavitation_2d_transolver/test_full_previews/metrics/`, including [per_sample_metrics.md](/home/hanklin/GeoPT/results/reynolds_cavitation_2d_transolver/test_full_previews/metrics/per_sample_metrics.md) and [reynolds_cavitation_2d_transolver_test_full_mesh_per_sample.csv](/home/hanklin/GeoPT/results/reynolds_cavitation_2d_transolver/test_full_previews/metrics/reynolds_cavitation_2d_transolver_test_full_mesh_per_sample.csv). The top-level [results/reynolds_cavitation_2d_transolver/](results/reynolds_cavitation_2d_transolver) folder also contains the model-side comparison PNG files (`gt_*.png`, `pred_*.png`, `error_*.png`).
- Hidden dimension: 256
- Attention heads: 8
- Transformer layers: 6
- Batch size: 2
- Learning rate: 0.001

**Execution Time**: ~2 minutes 11 seconds (~131 seconds)

**Status**: ✅ **COMPLETED SUCCESSFULLY**
- Training loss converged: 0.0707
- Validation loss converged: 0.0652
- Convergence reached by epoch ~180
- Metrics logged to: `training_logs/reynolds_cavitation_2d_transolver_metrics.csv`

---

### 4. Model Inference / Eval on the Test Set
```bash
cd /home/hanklin/GeoPT
source ~/anaconda3/etc/profile.d/conda.sh
conda activate geoPt
bash ./scripts/synthetic/test_reynolds_cavitation_2d_preview10.sh
```

**Script Location**: `scripts/synthetic/test_reynolds_cavitation_2d_preview10.sh`

**Configuration**:
- Test samples: 10
- Visualization samples: 11 (ensures all 10 are visualized)
- Model checkpoint: Loaded from full training
- Evaluation mode: eval=1 (no training)

**Status**: ✅ **COMPLETED SUCCESSFULLY**

**Test Metrics** (subsampled to 2,048 points):
- Relative error: 0.0614
- MSE: 0.0536
- MAE: 0.1408

**Test Metrics** (full mesh, 12,288 points):
- Relative error: 0.0606
- MSE: 0.0513
- MAE: 0.1366

---

## Generated Artifacts

### A. Training Outputs

| File | Location | Description |
|------|----------|-------------|
| Model checkpoint | `checkpoints/reynolds_cavitation_2d_transolver.pt` | Trained Transolver model (2,930,677 parameters) |
| Metrics CSV | `training_logs/reynolds_cavitation_2d_transolver_metrics.csv` | Per-epoch train/val metrics (200 rows) |
| Training logs | `training_logs/reynolds_cavitation_2d_transolver.log` | Detailed training output |

### B. Test Preview Plots (70 total)

**Location**: `results/reynolds_cavitation_2d_transolver/test_full_previews/plots/`

**Structure**: 10 test samples × 7 plot types each

**Plot Types** (per test sample):

1. **cross_section_XXXX.png** - Film geometry and vapor region at centerline
   - Shows contact-plane film thickness profile with vapor region
   - Side-view vertical cross-section

2. **phase_map_XXXX.png** - Vapor fraction field (contact-plane view)
   - Top-down spatial distribution of vapor phase
   - Useful for understanding cavitation patterns

3. **raw_pressure_XXXX.png** - Pressure field (cross-section)
   - Pressure magnitude across vertical cross-section
   - Side-view heat map

4. **density_XXXX.png** - Mixture density field
   - Density distribution showing liquid-vapor interface
   - Side-view heat map

5. **velocity_gradient_XXXX.png** - Shear stress proxy
   - Velocity gradient magnitude (related to shear stress)
   - Side-view heat map

6. **flow_XXXX.png** - Velocity vectors with speed magnitude
   - Quiver plot of velocity field
   - Speed magnitude as background heat map
   - Side-view, shows fluid motion patterns

7. **top_down_pressure_XXXX.png** - Pressure and liquid-air boundary
   - Pressure field in contact plane (top-down)
   - Overlaid with liquid-air interface contour
   - Paper-publication style visualization

**Sample Numbering**: 0001 to 0010 (10 test samples)

**File Count**: 70 PNG images total

---

## Data Pipeline

### Input Data
- **Source**: 2D Reynolds cavitation synthetic data with per-node velocity features
- **Generated via**: `data_preprocess/ReynoldsCavitation2D_process.py`
- **Format**:
  - `x_*.npy`: shape (N, 4) → [x_coord, z_coord, film_height, u_node_velocity]
  - `y_*.npy`: shape (N, 5) → [pressure, vapor_fraction, density, film_height, shear_proxy]
  - `cond_*.npy`: shape (5,) → [wall_speed, viscosity, vapor_pressure, h_min, h_max]

### Data Loader
- **Class**: `ReynoldsCavitation2D` in `data_provider/data_loader.py`
- **Processing**:
  - Splits x into: `pos` (first 3 cols: x, z, h) and `fx` (remaining cols: u_node velocity)
  - Subsamples to 2,048 points per sample (for training/validation)
  - Full 12,288 points available for full-mesh evaluation
  - Expands global conditions to per-node features via `reynolds_cavitation` direction function

### Model Architecture
- **Type**: Transolver (Transformer with Operator Layer)
- **Input**: Concatenated [space coords (3) + velocity feature (1) + expanded conditions (5)] = 8 features
- **Preprocess**: MLP(8 → 512 → 256)
- **Core**: 6 Transformer blocks (256 hidden dim, 8 attention heads)
- **Output**: Linear(256 → 5) for physics quantities
- **Total Parameters**: 2,930,677

---

## Key Implementation Details

### Per-Node Velocity Feature (u_node)
- Computed from thin-film Reynolds approximation in `build_sample()` function
- Represents depth-averaged velocity in the film
- Saved as 4th column in `x_*.npy` coordinates
- Loader treats as feature (fx) rather than spatial coordinate
- Enables model to learn velocity-dependent cavitation dynamics

### Condition Expansion
- Global conditions (5 parameters) expanded to per-node features
- Done via `reynolds_cavitation.direction()` function in `exp/dynamics_config.py`
- Each node receives full condition vector concatenated with local velocity
- Allows model to learn spatially-varying physics under different operating conditions

### Training Strategy
- **Optimizer**: AdamW with learning rate 0.001
- **Batch size**: 2
- **Loss**: L2 regression loss on 5 physics quantities
- **Scheduling**: No explicit scheduler (constant LR)
- **Convergence**: Stable after ~180 epochs (out of 200)

---

## Validation Checklist

- ✅ Dataset loading verified (64 train, 16 test samples)
- ✅ Model architecture aligned (space_dim=3, fun_dim=5)
- ✅ Training completed without shape mismatches
- ✅ Loss curves show convergence
- ✅ Test evaluation metrics computed
- ✅ Preview plots generated successfully (70 PNG images)
- ✅ All 10 test samples visualized with 7 plot types each

---

## How to Review Results

1. **View training metrics**:
   ```bash
   cat training_logs/reynolds_cavitation_2d_transolver_metrics.csv | head -20
   ```


2. Trained-Model Inference on Test Data
   ```bash
   cd /home/hanklin/GeoPT
   bash ./scripts/synthetic/test_reynolds_cavitation_2d_preview.sh
   ```

   **Script Location**: `scripts/synthetic/test_reynolds_cavitation_2d_preview.sh`

   **Purpose**: Run the trained Transolver checkpoint on the test split and save eval/preview outputs


3. Ground-Truth Test Visualization
   ```bash
   cd /home/hanklin/GeoPT
   bash scripts/synthetic/visualize_test_data.sh
   ```

   **Script Location**: `scripts/synthetic/visualize_test_data.sh`

   **Purpose**: Visualize already-generated test NPY files directly (no model predictions)



4. **Review preview and inference outputs**: `results/reynolds_cavitation_2d_transolver/test_full_previews/plots/` contains the 10 preview PNGs (`cross_section_*.png`, `phase_map_*.png`, `raw_pressure_*.png`, `density_*.png`, `velocity_gradient_*.png`, `flow_*.png`, `top_down_pressure_*.png`). The metrics live in `results/reynolds_cavitation_2d_transolver/test_full_previews/metrics/`, including [per_sample_metrics.md](/home/hanklin/GeoPT/results/reynolds_cavitation_2d_transolver/test_full_previews/metrics/per_sample_metrics.md) and [reynolds_cavitation_2d_transolver_test_full_mesh_per_sample.csv](/home/hanklin/GeoPT/results/reynolds_cavitation_2d_transolver/test_full_previews/metrics/reynolds_cavitation_2d_transolver_test_full_mesh_per_sample.csv). The top-level [results/reynolds_cavitation_2d_transolver/](results/reynolds_cavitation_2d_transolver) folder also contains the model-side comparison PNG files (`gt_*.png`, `pred_*.png`, `error_*.png`).

---

## File Structure Reference

```
/home/hanklin/GeoPT/
├── data/
│   ├── reynolds_cavitation_npys/
│   ├── reynolds_cavitation_2d_npys/
├── checkpoints/
│   └── reynolds_cavitation_2d_transolver.pt (trained model)
├── training_logs/
│   ├── reynolds_cavitation_2d_transolver_metrics.csv
│   └── reynolds_cavitation_2d_transolver.log
├── results/
│   └── reynolds_cavitation_2d_transolver/
│       ├── gt_*.png / pred_*.png / error_*.png
│       └── test_full_previews/
│           ├── plots/
│           │   ├── cross_section_0001.png ... 0010.png
│           │   ├── phase_map_0001.png ... 0010.png
│           │   ├── raw_pressure_0001.png ... 0010.png
│           │   ├── density_0001.png ... 0010.png
│           │   ├── velocity_gradient_0001.png ... 0010.png
│           │   ├── flow_0001.png ... 0010.png
│           │   └── top_down_pressure_0001.png ... 0010.png
│           └── metrics/
│               ├── per_sample_metrics.md
│               └── reynolds_cavitation_2d_transolver_test_full_mesh_per_sample.csv
├── data_preprocess/
│   └── ReynoldsCavitation2D_process.py
├── data_provider/
│   └── data_loader.py
├── exp/
│   └── steady_cond.py
├── utils/
│   └── visual.py
└── scripts/synthetic/
    ├── train_reynolds_cavitation_2d.sh
    ├── test_reynolds_cavitation_2d_preview10.sh
    └── preview_reynolds_cavitation_2d.sh
```

---

## Summary

Successfully trained a Transolver model on 2D Reynolds cavitation contact-plane dynamics with per-node velocity features. The model converged in 200 epochs with stable test performance (rel_err ~6.1%). Generated comprehensive preview visualizations for 10 test samples showing the model's learned physics predictions across 7 different plot types (cross-section geometry, phase field, pressure, density, shear, flow, and top-down views).

All outputs are ready for review and further analysis.

**Generated**: May 10, 2026
**Status**: ✅ Complete
