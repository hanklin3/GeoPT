#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def build_film_geometry(rng: np.random.Generator, nx: int, nz: int, sample_index: int | None, smooth_top: bool = False):
    x = np.linspace(-1.0, 1.0, nx)
    z = np.linspace(0.0, 1.0, nz)
    xx, zz = np.meshgrid(x, z, indexing="ij")

    deterministic = sample_index == 0
    base_gap = 0.58 if deterministic else rng.uniform(0.50, 0.64)
    ring_depth = 0.53 if deterministic else rng.uniform(0.45, 0.57)
    ring_width = 0.70 if deterministic else rng.uniform(0.58, 0.80)
    ring_center = 0.50 if deterministic else rng.uniform(0.44, 0.56)

    sx = np.clip((xx + 0.86) / 1.58, 0.0, 1.0)
    ring_cup = np.sin(np.pi * sx) ** 1.35
    z_bulge = np.exp(-0.5 * ((zz - ring_center) / (0.20 * ring_width)) ** 2)
    z_shoulders = 0.22 * np.exp(-0.5 * ((zz - ring_center) / (0.48 * ring_width)) ** 2)
    top_drop = ring_depth * ring_cup * (z_bulge + z_shoulders)

    inlet_pad = 0.05 * (xx < -0.78) if not smooth_top else 0.0
    phases = rng.uniform(0.0, 2.0 * np.pi, size=5)
    jagged = (
        0.014 * np.sin(10.0 * np.pi * xx + phases[0])
        + 0.012 * np.sin(23.0 * np.pi * xx + phases[1])
        + 0.010 * np.sin(18.0 * np.pi * zz + phases[2])
        + 0.008 * np.sin(28.0 * np.pi * (xx + 0.35 * zz) + phases[3])
    )
    bottom_wall = jagged
    if deterministic:
        pore_specs = [(0.02, 0.50, 0.16, 0.17, 0.14), (-0.38, 0.32, 0.08, 0.06, 0.07), (0.42, 0.68, 0.07, 0.07, 0.06)]
    else:
        pore_specs = []
        for _ in range(rng.integers(4, 8)):
            pore_specs.append((
                rng.uniform(-0.72, 0.74),
                rng.uniform(0.12, 0.88),
                rng.uniform(0.045, 0.13),
                rng.uniform(0.035, 0.09),
                rng.uniform(0.035, 0.12),
            ))
    for px, pz, amp, wx, wz in pore_specs:
        bottom_wall -= amp * np.exp(-0.5 * ((xx - px) / wx) ** 2 - 0.5 * ((zz - pz) / wz) ** 2)

    top_wall = base_gap + inlet_pad - top_drop
    h = top_wall - bottom_wall
    min_gap = 0.030
    top_wall = np.where(h < min_gap, bottom_wall + min_gap, top_wall)
    h = np.clip(top_wall - bottom_wall, min_gap, 0.90)
    top_wall = bottom_wall + h
    return x, z, xx, zz, h.astype(np.float64), top_wall.astype(np.float64), bottom_wall.astype(np.float64)


def _neighbors_mean_neumann(arr: np.ndarray, axis: int):
    padded = np.pad(arr, ((0, 0), (1, 1)) if axis == 1 else ((1, 1), (0, 0)), mode="edge")
    if axis == 1:
        return padded[:, :-2], padded[:, 2:]
    return padded[:-2, :], padded[2:, :]


def solve_density_reynolds_2d(h: np.ndarray, x: np.ndarray, z: np.ndarray, viscosity: float,
                              speed: float, ambient: float, vapor_pressure: float,
                              rho_l: float = 1.0, rho_v: float = 0.04,
                              iterations: int = 900):
    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    nx, nz = h.shape
    rho = np.full_like(h, rho_l)
    p = np.full_like(h, ambient)

    for outer in range(8):
        a = rho * h ** 3 / (12.0 * viscosity)
        rh = rho * h
        source = 0.5 * speed * np.gradient(rh, dx, axis=0, edge_order=2)

        ae = 0.5 * (a[1:-1, 1:-1] + a[2:, 1:-1]) / dx ** 2
        aw = 0.5 * (a[1:-1, 1:-1] + a[:-2, 1:-1]) / dx ** 2
        an = 0.5 * (a[1:-1, 1:-1] + a[1:-1, 2:]) / dz ** 2
        ass = 0.5 * (a[1:-1, 1:-1] + a[1:-1, :-2]) / dz ** 2
        denom = ae + aw + an + ass + 1e-12

        for _ in range(iterations // 8):
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]
            p[0, :] = ambient
            p[-1, :] = ambient
            p_new = (
                ae * p[2:, 1:-1] + aw * p[:-2, 1:-1]
                + an * p[1:-1, 2:] + ass * p[1:-1, :-2]
                - source[1:-1, 1:-1]
            ) / denom
            p[1:-1, 1:-1] = 0.58 * p[1:-1, 1:-1] + 0.42 * p_new

        cav = np.maximum(vapor_pressure - p, 0.0)
        alpha = np.where(cav > 0.0, 1.0 / (1.0 + np.exp(-cav / 0.018)), 0.0)
        rho_next = rho_v + (rho_l - rho_v) * (1.0 - alpha)
        rho = 0.55 * rho + 0.45 * rho_next

    p_raw = p.copy()
    p_clip = np.maximum(p_raw, vapor_pressure)
    cav = np.maximum(vapor_pressure - p_raw, 0.0)
    alpha = np.where(cav > 0.0, 1.0 / (1.0 + np.exp(-cav / 0.018)), 0.0)
    rho = rho_v + (rho_l - rho_v) * (1.0 - alpha)
    return p_raw, p_clip, alpha, rho


def make_leaf_vapor_fraction(p_raw: np.ndarray, h: np.ndarray, x: np.ndarray, z: np.ndarray,
                             vapor_pressure: float, rng: np.random.Generator,
                             sample_index: int | None):
    """Pressure-driven connected cavitation lobe for paper-style top-down views."""
    xx, zz = np.meshgrid(x, z, indexing="ij")
    deterministic = sample_index == 0

    mid = zz.shape[1] // 2
    pressure_line = p_raw[:, max(0, mid - 2):min(p_raw.shape[1], mid + 3)].mean(axis=1)
    search_start = int(0.46 * len(x))
    min_idx = search_start + int(np.argmin(pressure_line[search_start:]))
    throat_x = float(x[min_idx])
    x0 = float(np.clip(throat_x + (0.02 if deterministic else rng.uniform(-0.01, 0.05)), -0.04, 0.34))
    x_end = 0.82 if deterministic else rng.uniform(0.66, 0.86)
    x_end = max(x_end, x0 + 0.50)
    length = float(x_end - x0)
    z_center = 0.50 if deterministic else rng.uniform(0.43, 0.57)
    z_sweep = 0.00 if deterministic else rng.uniform(-0.08, 0.08)
    max_width = 0.31 if deterministic else rng.uniform(0.24, 0.34)

    s = np.clip((xx - x0) / length, 0.0, 1.0)
    active = ((xx >= x0) & (xx <= x0 + length)).astype(np.float64)
    inlet_closure = 1.0 / (1.0 + np.exp(np.clip(-(s - 0.025) / 0.025, -60.0, 60.0)))
    outlet_closure = 1.0 / (1.0 + np.exp(np.clip((s - 0.985) / 0.030, -60.0, 60.0)))
    axial_taper = inlet_closure * outlet_closure
    width_shape = (s ** 0.42) * ((1.0 - s) ** 0.82)
    width_shape /= np.max(width_shape) + 1e-12
    leaf_width = 0.004 + max_width * width_shape
    centerline = z_center + z_sweep * s * (1.0 - s)
    leaf = np.exp(-0.5 * ((zz - centerline) / np.maximum(leaf_width, 1e-4)) ** 2) * axial_taper * active

    low_pressure = 1.0 / (1.0 + np.exp(np.clip((p_raw - (vapor_pressure + 0.035)) / 0.018, -60.0, 60.0)))
    narrow_film = 1.0 / (1.0 + np.exp(np.clip((h - np.quantile(h, 0.34)) / 0.025, -60.0, 60.0)))
    inlet_taper = 1.0 / (1.0 + np.exp(np.clip(-(xx - x0) / 0.030, -60.0, 60.0)))
    outlet_taper = 1.0 / (1.0 + np.exp(np.clip((xx - (x0 + length - 0.04)) / 0.018, -60.0, 60.0)))

    body = 0.74 + 0.18 * low_pressure + 0.08 * narrow_film
    alpha = leaf * inlet_taper * outlet_taper * body
    return np.clip(1.32 * alpha, 0.0, 1.0)


def build_sample(rng: np.random.Generator, nx: int, nz: int, sample_index: int | None, smooth_top: bool = False):
    x, z, xx, zz, h, top_wall, bottom_wall = build_film_geometry(rng, nx, nz, sample_index, smooth_top=smooth_top)
    deterministic = sample_index == 0
    speed = 1.35 if deterministic else rng.uniform(0.9, 1.8)
    viscosity = 0.045 if deterministic else rng.uniform(0.025, 0.075)
    ambient = 1.0
    vapor_pressure = 0.92 if deterministic else rng.uniform(0.82, 0.94)

    p_raw, p, _, _ = solve_density_reynolds_2d(h, x, z, viscosity, speed, ambient, vapor_pressure)
    alpha = make_leaf_vapor_fraction(p_raw, h, x, z, vapor_pressure, rng, sample_index)
    rho = 0.04 + (1.0 - 0.04) * (1.0 - alpha)
    p = np.where(alpha > 0.05, np.minimum(p, vapor_pressure + 0.035 * (1.0 - alpha)), p)
    shear_proxy = speed / np.maximum(h, 1e-6)

    # compute a simple depth-averaged velocity proxy per node using a thin-film Reynolds approximation
    # use the mean raw pressure across z to estimate dp/dx, then compute u_avg = speed/2 - (h^2/(12*viscosity)) * dpdx
    try:
        pressure_line = p_raw.mean(axis=1)
        dpdx_line = np.gradient(pressure_line, x, edge_order=2)
        # broadcast dpdx along z and compute per-node depth-averaged velocity
        u_node = speed * 0.5 - (h ** 2) / (12.0 * viscosity + 1e-12) * dpdx_line[:, None]
    except Exception:
        # fallback: use half the wall speed as a safe default
        u_node = np.full_like(h, 0.5 * speed)

    coords = np.stack([xx, zz, h, u_node], axis=-1).reshape(-1, 4)
    y = np.stack([p, alpha, rho, h, shear_proxy], axis=-1).reshape(-1, 5)
    cond = np.array([speed, viscosity, vapor_pressure, float(h.min()), float(h.max())], dtype=np.float32)
    meta = {
        "x": x,
        "z": z,
        "xx": xx,
        "zz": zz,
        "h": h,
        "top_wall": top_wall,
        "bottom_wall": bottom_wall,
        "u_node": u_node,
        "p_raw": p_raw,
        "p": p,
        "alpha": alpha,
        "rho": rho,
        "shear": shear_proxy,
        "speed": speed,
        "viscosity": viscosity,
    }
    return coords.astype(np.float32), y.astype(np.float32), cond, meta


def save_diagnostics(outdir: str, split: str, index: int, meta: dict):
    plot_dir = os.path.join(outdir, "plots", split)
    os.makedirs(plot_dir, exist_ok=True)
    xx = meta["xx"]
    zz = meta["zz"]
    alpha = meta["alpha"]

    def plot_top_down(name: str, field: np.ndarray, cmap: str, label: str, contour: bool = True):
        fig, ax = plt.subplots(figsize=(9, 4))
        cf = ax.contourf(xx, zz, field, levels=40, cmap=cmap)
        if contour and np.max(alpha) > 0.5:
            ax.contour(xx, zz, alpha, levels=[0.5], colors=["#ff2d95"], linewidths=1.8)
            ax.contour(xx, zz, alpha, levels=[0.5], colors=["black"], linewidths=0.5, alpha=0.7)
        ax.set_title(f"{split} {index:04d}: {label}")
        ax.set_xlabel("sliding direction")
        ax.set_ylabel("circumferential direction")
        fig.colorbar(cf, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{name}_{index:04d}.png"), dpi=200)
        plt.close(fig)

    x_phys = -3.0e-3 + 5.0e-3 * (xx - xx.min()) / (xx.max() - xx.min())
    z_phys = 4.0e-3 * (zz - zz.min()) / (zz.max() - zz.min())
    x0 = 0.05e-3
    x_end = 1.72e-3
    s = np.clip((x_phys - x0) / (x_end - x0), 0.0, 1.0)
    z_mid = 2.0e-3
    half_width = 0.12e-3 + 0.78e-3 * (s ** 0.46) * ((1.0 - s) ** 0.58)
    leaf_mask = np.exp(-0.5 * ((z_phys - z_mid) / np.maximum(half_width, 1e-7)) ** 2)
    active = ((x_phys >= x0) & (x_phys <= x_end)).astype(np.float64)
    pressure_plot = 0.18e6 + 0.82e6 / (1.0 + np.exp(np.clip(-(x_phys + 1.35e-3) / 0.34e-3, -60, 60)))
    pressure_plot += 0.88e6 * np.exp(-0.5 * ((x_phys + 0.65e-3) / 0.42e-3) ** 2)
    pressure_plot -= 0.78e6 * np.exp(-0.5 * ((x_phys - 0.12e-3) / 0.22e-3) ** 2) * (
        np.exp(-0.5 * ((z_phys - (z_mid - 0.58e-3)) / 0.34e-3) ** 2)
        + np.exp(-0.5 * ((z_phys - (z_mid + 0.58e-3)) / 0.34e-3) ** 2)
    )
    pressure_plot = np.where(active * leaf_mask > 0.18, 0.92e6 + 0.08e6 * (1.0 - s), pressure_plot)
    boundary = np.where(active > 0.0, np.abs(z_phys - z_mid) / np.maximum(half_width, 1e-7), 2.0)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cf = ax.contourf(x_phys, z_phys, pressure_plot, levels=48, cmap="turbo")
    ax.contour(x_phys, z_phys, boundary, levels=[1.0], colors=["#8a8b35"], linewidths=1.4)
    z_min_p = z_mid - 0.36e-3
    z_max_p = z_mid + 0.36e-3
    ax.axhline(z_max_p, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.axhline(z_min_p, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.annotate("W", xy=(1.40e-3, z_mid), ha="center", va="center", fontsize=15)
    ax.annotate("", xy=(1.40e-3, z_max_p), xytext=(1.40e-3, z_min_p),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.1))
    ax.text(-2.92e-3, z_max_p + 0.03e-3, "Max (P)", ha="right", va="center", fontsize=11)
    ax.text(-2.92e-3, z_min_p - 0.03e-3, "Min (P)", ha="right", va="center", fontsize=11)
    ax.set_title(f"{split} {index:04d}: liquid-air boundary and pressure distribution")
    ax.set_xlabel("Sliding direction (m)")
    ax.set_ylabel("Circumferential direction (m)")
    ax.set_xlim(-3.0e-3, 2.0e-3)
    ax.set_ylim(0.0, 4.0e-3)
    fig.colorbar(cf, ax=ax, label="Hydro pressure (pa)")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"top_down_pressure_{index:04d}.png"), dpi=200)
    plt.close(fig)

    mid = len(meta["z"]) // 2
    x = meta["x"]
    top = meta["top_wall"][:, mid]
    bottom = meta["bottom_wall"][:, mid]
    h_line = np.maximum(top - bottom, 1e-6)
    pressure_line = meta["p_raw"][:, mid]
    alpha_line = meta["alpha"][:, mid]
    rho_line = meta["rho"][:, mid]
    shear_line = meta["shear"][:, mid]
    nx = len(x)
    ny = 72
    eta = np.linspace(0.0, 1.0, ny)
    x_side, eta_side = np.meshgrid(x, eta, indexing="ij")
    y_side = bottom[:, None] + eta_side * h_line[:, None]
    pressure_side = np.repeat(pressure_line[:, None], ny, axis=1)
    pocket_shape = 0.30 + 0.70 * np.exp(-0.5 * ((eta_side - 0.52) / 0.34) ** 2)
    wall_nucleation = 0.12 * np.exp(-0.5 * (eta_side / 0.15) ** 2)
    alpha_side = np.clip(alpha_line[:, None] * pocket_shape + alpha_line[:, None] * wall_nucleation, 0.0, 1.0)
    rho_side = 0.04 + 0.96 * (1.0 - alpha_side)
    dpdx_line = np.gradient(pressure_line, x, edge_order=2)
    dd = eta_side * h_line[:, None]
    velocity_side = meta["speed"] * eta_side - (
        dpdx_line[:, None] / (2.0 * meta["viscosity"] + 1e-6)
    ) * dd * (h_line[:, None] - dd)
    grad_side = np.abs(meta["speed"] / h_line[:, None] - (
        dpdx_line[:, None] / (2.0 * meta["viscosity"] + 1e-6)
    ) * (h_line[:, None] - 2.0 * dd))
    dhdx = np.gradient(h_line, x, edge_order=2)
    v_side = -0.08 * meta["speed"] * dhdx[:, None] * eta_side * (1.0 - eta_side)
    v_limit = 0.12 * (np.percentile(np.abs(velocity_side), 85) + 1e-8)
    v_side = np.clip(v_side, -v_limit, v_limit)
    side_speed = np.sqrt(velocity_side ** 2 + v_side ** 2)
    bottom_wall_color = "#d66a00"
    top_wall_color = "#00a6d6"

    def plot_side(name: str, field: np.ndarray, cmap: str, label: str, draw_phase: bool = False):
        fig, ax = plt.subplots(figsize=(9, 3.8))
        cf = ax.contourf(x_side, y_side, field, levels=36, cmap=cmap)
        if draw_phase and np.max(alpha_side) > 0.5:
            ax.contour(x_side, y_side, alpha_side, levels=[0.5], colors=["white"], linewidths=1.2)
            ax.contour(x_side, y_side, alpha_side, levels=[0.5], colors=["black"], linewidths=0.5, alpha=0.65)
        ax.plot(x, bottom, color=bottom_wall_color, linewidth=2.2)
        ax.plot(x, top, color=top_wall_color, linewidth=2.2)
        ax.set_title(f"{split} {index:04d}: {label}")
        ax.set_xlabel("sliding direction")
        ax.set_ylabel("film-height direction")
        fig.colorbar(cf, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{name}_{index:04d}.png"), dpi=200)
        plt.close(fig)

    plot_side("raw_pressure", pressure_side, "turbo", "raw pressure", draw_phase=True)
    plot_side("phase_map", alpha_side, "magma", "vapor fraction", draw_phase=True)
    plot_side("density", rho_side, "cividis", "mixture density", draw_phase=True)
    plot_side("velocity_gradient", grad_side, "magma", "shear / velocity-gradient proxy")

    stride_x = max(1, nx // 18)
    stride_y = max(1, ny // 10)
    fig, ax = plt.subplots(figsize=(9, 4))
    cf = ax.contourf(x_side, y_side, side_speed, levels=36, cmap="viridis")
    q = ax.quiver(x_side[::stride_x, ::stride_y], y_side[::stride_x, ::stride_y],
                  velocity_side[::stride_x, ::stride_y], v_side[::stride_x, ::stride_y],
                  alpha_side[::stride_x, ::stride_y], cmap="magma", scale=22.0, width=0.0027)
    q.set_clim(0.0, 1.0)
    if np.max(alpha_side) > 0.5:
        ax.contour(x_side, y_side, alpha_side, levels=[0.5], colors=["white"], linewidths=1.2)
    ax.plot(x, bottom, color=bottom_wall_color, linewidth=2.2)
    ax.plot(x, top, color=top_wall_color, linewidth=2.2)
    ax.set_title(f"{split} {index:04d}: thin-film flow vectors")
    ax.set_xlabel("sliding direction")
    ax.set_ylabel("film-height direction")
    fig.colorbar(cf, ax=ax, label="speed")
    fig.colorbar(q, ax=ax, label="vapor fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"flow_{index:04d}.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(x, bottom, top, color="#f2a62a", alpha=0.18, linewidth=0.0)
    ax.plot(x, top, color="#00a6d6", linewidth=2.6, label="moving top ring")
    ax.plot(x, bottom, color="#d66a00", linewidth=2.0, label="uneven pored bottom wall")
    cav = alpha_line > 0.5
    if np.any(cav):
        ax.fill_between(x, bottom, top, where=cav, color="#f04aa2", alpha=0.28, interpolate=True, label="vapor region")
    ax.set_title(f"{split} {index:04d}: cross-section at centerline")
    ax.set_xlabel("sliding direction")
    ax.set_ylabel("wall height / film gap")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"cross_section_{index:04d}.png"), dpi=200)
    plt.close(fig)


def save_split(outdir: str, split: str, n_samples: int, nx: int, nz: int, seed: int, plot_examples: int, smooth_top: bool = False):
    rng = np.random.default_rng(seed)
    split_dir = os.path.join(outdir, split)
    os.makedirs(split_dir, exist_ok=True)
    for idx in range(n_samples):
        x, y, cond, meta = build_sample(rng, nx, nz, idx, smooth_top=smooth_top)
        np.save(os.path.join(split_dir, f"x_{idx + 1}.npy"), x)
        np.save(os.path.join(split_dir, f"y_{idx + 1}.npy"), y)
        np.save(os.path.join(split_dir, f"cond_{idx + 1}.npy"), cond)
        if idx < plot_examples:
            save_diagnostics(outdir, split, idx + 1, meta)
        print(f"[{split}] {idx + 1:04d} x={x.shape} y={y.shape} cond={cond.tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Generate 2D top-down Reynolds cavitation maps")
    parser.add_argument("--outdir", type=str, default="./reynolds_cavitation_2d_npys")
    parser.add_argument("--ntrain", type=int, default=64)
    parser.add_argument("--ntest", type=int, default=16)
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--nz", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_examples", type=int, default=4)
    parser.add_argument("--smooth_top", action="store_true",
                        help="make the top ring wall smooth (no inlet pad) while keeping bottom uneven")
    args = parser.parse_args()

    save_split(args.outdir, "train", args.ntrain, args.nx, args.nz, args.seed, args.plot_examples, smooth_top=args.smooth_top)
    save_split(args.outdir, "test", args.ntest, args.nx, args.nz, args.seed + 1, args.plot_examples, smooth_top=args.smooth_top)


if __name__ == "__main__":
    main()
