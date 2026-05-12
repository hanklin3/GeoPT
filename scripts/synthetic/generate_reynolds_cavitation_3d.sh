#!/bin/bash
# Generate the full 3D Reynolds cavitation dataset.

cd "$(dirname "$0")/../.." || exit

python ./data_preprocess/ReynoldsCavitation3D_process.py \
--outdir ./data/reynolds_cavitation_3d_npys \
--ntrain 64 \
--ntest 16 \
--nx 56 \
--ny 40 \
--nz 14 \
--geometry wavy \
--seed 0 \
--plot_examples 4
