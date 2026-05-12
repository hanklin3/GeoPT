#!/bin/bash
# Generate a small 3D Reynolds cavitation preview dataset with slice/projection plots.

cd "$(dirname "$0")/../.." || exit

python ./data_preprocess/ReynoldsCavitation3D_process.py \
--outdir ./data/reynolds_cavitation_3d_preview \
--ntrain 0 \
--ntest 8 \
--nx 56 \
--ny 40 \
--nz 14 \
--geometry wavy \
--seed 0 \
--plot_examples 4