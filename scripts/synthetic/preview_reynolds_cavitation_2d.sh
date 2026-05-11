#!/bin/bash
# Generate a small 2D Reynolds cavitation preview dataset for quick inspection.

cd "$(dirname "$0")/../.." || exit

python ./data_preprocess/ReynoldsCavitation2D_process.py \
--outdir ./data/reynolds_cavitation_2d_preview \
--ntrain 0 \
--ntest 10 \
--nx 128 \
--nz 96 \
--seed 0 \
--plot_examples 6
