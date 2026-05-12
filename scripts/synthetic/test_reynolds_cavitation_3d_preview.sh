PYTHON=${PYTHON:-python}

$PYTHON run.py \
--gpu 0 \
--data_path ./data/reynolds_cavitation_3d_npys \
--loader ReynoldsCavitation3D \
--task steady_cond \
--dynamics reynolds_cavitation \
--geotype unstructured \
--space_dim 3 \
--fun_dim 5 \
--out_dim 5 \
--normalize 1 \
--model Transolver \
--n_hidden 256 \
--n_heads 8 \
--n_layers 6 \
--mlp_ratio 2 \
--slice_num 32 \
--ntrain 64 \
--ntest 16 \
--batch-size 1 \
--epochs 1 \
--eval 1 \
--vis_num 5 \
--save_name reynolds_cavitation_3d_transolver
