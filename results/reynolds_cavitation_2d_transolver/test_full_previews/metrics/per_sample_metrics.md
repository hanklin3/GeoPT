# Reynolds2D Per-Sample Metrics

- CSV: `reynolds_cavitation_2d_transolver_test_full_mesh_per_sample.csv`
- Samples: 10
- Mean relative L2 prediction error: 0.0605649
- Mean predicted relative conservation residual L1: 0.758346
- Mean ground-truth relative conservation residual L1: 0.708356

## Conservation Form

The residual checks the density-aware Reynolds equation used by the synthetic generator:

```text
R = div((rho h^3 / (12 mu)) grad p) - (V / 2) d(rho h) / dx
```

where:

- `p` is pressure
- `rho` is mixture density
- `h` is film height from the input test geometry
- `mu` is viscosity from `cond_*.npy`
- `V` is wall speed from `cond_*.npy`

## Residual Metrics

For each full-mesh test sample, the report computes the residual on the interior grid points:

```text
residual_l1 = mean(abs(R))
residual_l2 = sqrt(mean(R^2))
residual_linf = max(abs(R))
residual_relative_l1 = mean(abs(R)) / (mean(abs(div_flux)) + mean(abs(source)) + eps)
```

Both predicted and ground-truth residuals are included in the CSV as `pred_*` and `true_*` columns.

## Per-Sample Table

| Sample | Points | Rel L2 | MSE | MAE | Pred Rel Residual L1 | True Rel Residual L1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12288 | 0.0914713 | 0.0482375 | 0.131544 | 0.773384 | 0.726794 |
| 2 | 12288 | 0.0412651 | 0.0665895 | 0.152835 | 0.736886 | 0.783978 |
| 3 | 12288 | 0.0393553 | 0.0723869 | 0.146329 | 0.778618 | 0.488257 |
| 4 | 12288 | 0.044302 | 0.0418392 | 0.13834 | 0.776196 | 0.822679 |
| 5 | 12288 | 0.0505723 | 0.0696502 | 0.162612 | 0.748514 | 0.559726 |
| 6 | 12288 | 0.102205 | 0.0377331 | 0.116555 | 0.73911 | 0.777613 |
| 7 | 12288 | 0.0736723 | 0.0385214 | 0.1308 | 0.759705 | 0.76175 |
| 8 | 12288 | 0.0652325 | 0.064057 | 0.149518 | 0.727681 | 0.673918 |
| 9 | 12288 | 0.0484585 | 0.0402106 | 0.124138 | 0.780685 | 0.766111 |
| 10 | 12288 | 0.0491149 | 0.033942 | 0.112882 | 0.762678 | 0.722733 |

## Channel Error Table

| Sample | Pressure Rel L2 | Vapor Rel L2 | Density Rel L2 | Film Height Rel L2 | Shear Rel L2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.114564 | 0.814056 | 0.37412 | 0.0600566 | 0.020925 |
| 2 | 0.240137 | 0.908443 | 0.283966 | 0.1428 | 0.0253975 |
| 3 | 0.22107 | 0.900882 | 0.309008 | 0.164872 | 0.0225184 |
| 4 | 0.161715 | 0.859605 | 0.2939 | 0.127803 | 0.0170017 |
| 5 | 0.161492 | 0.877154 | 0.29739 | 0.137308 | 0.0345411 |
| 6 | 0.114665 | 0.850736 | 0.309492 | 0.0912989 | 0.0275422 |
| 7 | 0.106026 | 0.839527 | 0.283712 | 0.0726358 | 0.0345645 |
| 8 | 0.164893 | 0.839381 | 0.331382 | 0.0970939 | 0.0384067 |
| 9 | 0.118027 | 0.869501 | 0.311508 | 0.0632909 | 0.0158887 |
| 10 | 0.130211 | 0.86856 | 0.279061 | 0.0620665 | 0.0138223 |
