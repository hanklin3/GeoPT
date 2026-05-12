import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tempfile
import warnings
from data_preprocess.ReynoldsCavitation2D_process import build_sample, save_diagnostics as reynolds_save_diagnostics
from data_preprocess.ReynoldsCavitation3D_process import save_diagnostics as reynolds3d_save_diagnostics

warnings.filterwarnings('ignore')


def _save_figure(fig, output_dir, base_name):
    fig.savefig(
        os.path.join(output_dir, f"{base_name}.png"),
        bbox_inches='tight',
        pad_inches=0,
    )


def visual(x, y, out, args, id):
    if args.geotype == 'structured_2D':
        visual_structured_2d(x, y, out, args, id)
    if args.geotype == 'unstructured' and x.shape[-1] == 2:
        visual_unstructured_2d(x, y, out, args, id)
    if args.geotype == 'unstructured' and x.shape[-1] == 3:
        visual_unstructured_3d(x, y, out, args, id)


def visual_unstructured_3d(x, y, out, args, id, channel=0):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter3D(x[0, :, 0].detach().cpu().numpy(), x[0, :, 1].detach().cpu().numpy(),
                           x[0, :, 2].detach().cpu().numpy(), c=y[0, :, channel].detach().cpu().numpy(),
                           cmap='coolwarm',
                           s=50)#, vmin=0.0, vmax=0.06)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Value')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    _save_figure(fig, os.path.join('./results/' + args.save_name + '/'), f"gt_{id}")
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter3D(x[0, :, 0].detach().cpu().numpy(), x[0, :, 1].detach().cpu().numpy(),
                           x[0, :, 2].detach().cpu().numpy(), c=out[0, :, channel].detach().cpu().numpy(),
                           cmap='coolwarm',
                           s=50)#, vmin=0.0, vmax=0.06)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Value')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    _save_figure(fig, os.path.join('./results/' + args.save_name + '/'), f"pred_{id}")
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter3D(x[0, :, 0].detach().cpu().numpy(), x[0, :, 1].detach().cpu().numpy(),
                           x[0, :, 2].detach().cpu().numpy(),
                           c=(y[0, :, channel] - out[0, :, channel]).detach().cpu().numpy(),
                           cmap='coolwarm', s=50)#, vmin=-0.02, vmax=0.02)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Value')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    _save_figure(fig, os.path.join('./results/' + args.save_name + '/'), f"error_{id}")
    plt.close()


def visual_unstructured_2d(x, y, out, args, id):
    plt.axis('off')
    plt.scatter(x=x[0, :, 0].detach().cpu().numpy(), y=x[0, :, 1].detach().cpu().numpy(),
                c=y[0, :].detach().cpu().numpy(), cmap='coolwarm')
    plt.colorbar()
    _save_figure(plt.gcf(), os.path.join('./results/' + args.save_name + '/'), f"gt_{id}")
    plt.close()

    plt.axis('off')
    plt.scatter(x=x[0, :, 0].detach().cpu().numpy(), y=x[0, :, 1].detach().cpu().numpy(),
                c=out[0, :].detach().cpu().numpy(), cmap='coolwarm')
    plt.colorbar()
    _save_figure(plt.gcf(), os.path.join('./results/' + args.save_name + '/'), f"pred_{id}")
    plt.close()

    plt.axis('off')
    plt.scatter(x=x[0, :, 0].detach().cpu().numpy(), y=x[0, :, 1].detach().cpu().numpy(),
                c=((y[0, :] - out[0, :])).detach().cpu().numpy(), cmap='coolwarm')
    plt.colorbar()
    _save_figure(plt.gcf(), os.path.join('./results/' + args.save_name + '/'), f"error_{id}")
    plt.close()


def visual_structured_1d(x, y, out, args, id):
    pass


def visual_structured_2d(x, y, out, args, id):
    if args.vis_bound is not None:
        space_x_min = args.vis_bound[0]
        space_x_max = args.vis_bound[1]
        space_y_min = args.vis_bound[2]
        space_y_max = args.vis_bound[3]
    else:
        space_x_min = 0
        space_x_max = args.shapelist[0]
        space_y_min = 0
        space_y_max = args.shapelist[1]
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   np.zeros([args.shapelist[0], args.shapelist[1]])[space_x_min: space_x_max, space_y_min: space_y_max],
                   shading='auto',
                   edgecolors='black', linewidths=0.1)
    plt.colorbar()
    _save_figure(plt.gcf(), os.path.join('./results/' + args.save_name + '/'), f"input_{id}")
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   out[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm')
    plt.colorbar()
    _save_figure(plt.gcf(), os.path.join('./results/' + args.save_name + '/'), f"pred_{id}")
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   y[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm')
    plt.colorbar()
    _save_figure(plt.gcf(), os.path.join('./results/' + args.save_name + '/'), f"gt_{id}")
    plt.close()
    plt.axis('off')
    plt.pcolormesh(x[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   x[0, :, 1].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   out[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy() - \
                   y[0, :, 0].reshape(args.shapelist[0], args.shapelist[1])[space_x_min: space_x_max,
                   space_y_min: space_y_max].detach().cpu().numpy(),
                   shading='auto', cmap='coolwarm')
    plt.colorbar()
    _save_figure(plt.gcf(), os.path.join('./results/' + args.save_name + '/'), f"error_{id}")
    plt.close()


def visual_structured_3d(x, y, out, args, id):
    pass


def _reconstruct_reynolds2d_meta(pos_np, args, id, split):
    xs = np.unique(pos_np[:, 0])
    zs = np.unique(pos_np[:, 1])
    nx, nz = len(xs), len(zs)
    if nx * nz != pos_np.shape[0]:
        return None

    target_h = pos_np[:, 2].reshape(nx, nz)
    base_seed = getattr(args, 'seed', 0)
    split_seed = base_seed + 1 if split.startswith('test') else base_seed
    requested_smooth_top = getattr(args, 'smooth_top', None)
    if requested_smooth_top is None:
        smooth_top_candidates = [False, True]
    else:
        requested_smooth_top = bool(requested_smooth_top)
        smooth_top_candidates = [requested_smooth_top, not requested_smooth_top]

    best_meta = None
    best_error = float('inf')
    for smooth_top in smooth_top_candidates:
        rng = np.random.default_rng(split_seed)
        meta = None
        for sample_index in range(id):
            _, _, _, meta = build_sample(rng, nx, nz, sample_index, smooth_top=smooth_top)
        if meta is None:
            continue
        error = float(np.max(np.abs(target_h - meta['h'])))
        if error < best_error:
            best_error = error
            best_meta = meta

    return best_meta


def _save_reynolds_diagnostics_flat(output_dir, split, id, meta):
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        reynolds_save_diagnostics(tmp_dir, split, id, meta)
        generated_dir = os.path.join(tmp_dir, 'plots', split)
        for filename in os.listdir(generated_dir):
            shutil.copy2(os.path.join(generated_dir, filename), os.path.join(output_dir, filename))


def visual_reynolds_cavitation_2d_preview(pos, fx, y, args, id, split='test'):
    """Save preview-style Reynolds2D diagnostics from a test batch."""
    save_dir = os.path.join('./results', args.save_name, f'{split}_previews', 'plots')
    os.makedirs(save_dir, exist_ok=True)
    if id == 1:
        print(f"Saving Reynolds2D predicted preview plots to: {os.path.abspath(save_dir)}")

    pos_np = pos[0].detach().cpu().numpy()
    meta = _reconstruct_reynolds2d_meta(pos_np, args, id, split)
    if meta is None:
        return

    # Convert model predictions from flat to 2D grids
    nx, nz = meta['h'].shape
    y_np = y[0].detach().cpu().numpy()  # Shape: (N, 5)
    p_pred = y_np[:, 0].reshape(nx, nz)  # Pressure
    alpha_pred = y_np[:, 1].reshape(nx, nz)  # Vapor fraction
    rho_pred = y_np[:, 2].reshape(nx, nz)  # Density
    shear_pred = y_np[:, 4].reshape(nx, nz)  # Shear proxy

    # Draw model fields on the actual input geometry. The wall profiles must come
    # from the test sample, not from the predicted height channel.
    meta['p_raw'] = p_pred
    meta['alpha'] = alpha_pred
    meta['rho'] = rho_pred
    meta['shear'] = shear_pred

    _save_reynolds_diagnostics_flat(save_dir, split, id, meta)


def _infer_reynolds3d_grid(pos_np):
    xs = np.unique(pos_np[:, 0])
    ys = np.unique(pos_np[:, 1])
    nx, ny = len(xs), len(ys)
    if nx == 0 or ny == 0 or pos_np.shape[0] % (nx * ny) != 0:
        return None
    nz = pos_np.shape[0] // (nx * ny)
    if nx * ny * nz != pos_np.shape[0]:
        return None
    return xs, ys, nx, ny, nz


def _reynolds3d_meta_from_field(pos_np, field_np, cond_np=None):
    grid = _infer_reynolds3d_grid(pos_np)
    if grid is None:
        return None
    xs, ys, nx, ny, nz = grid
    coords = pos_np.reshape(nx, ny, nz, 3)
    field = field_np.reshape(nx, ny, nz, -1)
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    eta = np.linspace(0.0, 1.0, nz)

    bottom = coords[:, :, 0, 2]
    top = coords[:, :, -1, 2]
    h = np.maximum(top - bottom, 1e-6)
    u = field[:, :, :, 0]
    v = field[:, :, :, 1]
    p3 = field[:, :, :, 2]
    alpha3 = np.clip(field[:, :, :, 3], 0.0, 1.0)
    rho3 = field[:, :, :, 4]
    speed = np.sqrt(u ** 2 + v ** 2)

    if cond_np is None:
        cond_flat = np.zeros(5, dtype=np.float64)
    else:
        cond_flat = np.asarray(cond_np).reshape(-1)
    surface_flag = int(round(float(cond_flat[0]))) if cond_flat.size > 0 else 1
    wall_speed_x = float(cond_flat[1]) if cond_flat.size > 1 else 1.0
    wall_speed_y = 0.0
    viscosity = float(cond_flat[2]) if cond_flat.size > 2 else 0.05

    p_mid = p3[:, :, nz // 2]
    alpha_2d = np.max(alpha3, axis=2)
    rho_2d = np.mean(rho3, axis=2)
    du_deta = np.gradient(u, eta, axis=2, edge_order=2)
    dv_deta = np.gradient(v, eta, axis=2, edge_order=2)
    shear = np.sqrt(du_deta ** 2 + dv_deta ** 2) / h[:, :, None]

    return {
        "x": xs,
        "y": ys,
        "z": eta,
        "xx": xx,
        "yy": yy,
        "h": h,
        "top": top,
        "bottom": bottom,
        "surface_flag": surface_flag,
        "wall_speed_x": wall_speed_x,
        "wall_speed_y": wall_speed_y,
        "viscosity": viscosity,
        "vapor_pressure": float(np.nanmin(p_mid)),
        "p_raw": p_mid,
        "p_reynolds_raw": p_mid,
        "p": p_mid,
        "alpha": alpha_2d,
        "alpha3": alpha3,
        "rho": rho_2d,
        "rho3": rho3,
        "u": u,
        "v": v,
        "speed": speed,
        "shear": shear,
    }


def visual_reynolds_cavitation_3d_preview(pos, y, out, args, id, split='test', cond=None):
    """Save Reynolds3D diagnostics using the same plotter as dataset generation."""
    output_root = os.path.join('./results', args.save_name, f'{split}_previews')
    os.makedirs(output_root, exist_ok=True)
    if id == 1:
        print(f"Saving Reynolds3D preview plots to: {os.path.abspath(output_root)}")

    pos_np = pos[0].detach().cpu().numpy()
    y_np = y[0].detach().cpu().numpy()
    out_np = out[0].detach().cpu().numpy()
    cond_np = cond[0].detach().cpu().numpy() if cond is not None else None
    true_meta = _reynolds3d_meta_from_field(pos_np, y_np, cond_np)
    pred_meta = _reynolds3d_meta_from_field(pos_np, out_np, cond_np)
    if true_meta is None or pred_meta is None:
        return
    reynolds3d_save_diagnostics(output_root, 'truth', id, true_meta)
    reynolds3d_save_diagnostics(output_root, 'prediction', id, pred_meta)
