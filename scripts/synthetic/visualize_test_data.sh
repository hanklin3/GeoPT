#!/bin/bash
# Visualize already-generated test NPY files directly (no model predictions)

cd "$(dirname "$0")/../.." || exit

python visualize_test_data.py \
  --data_dir ./data/reynolds_cavitation_2d_npys/test \
  --save_name ./data/reynolds_cavitation_2d_npys \
  --test_range 0 10 \
  --nx 128 \
  --nz 96
