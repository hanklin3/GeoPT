#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def build_surface_geometry(rng: np.random.Generator, nx: int, ny: int, sample_index: int | None,
                           geometry: str = "wavy"):
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")

    deterministic = sample_index == 0
    if geometry == "flat":
        surface_flag = 0
    elif geometry == "wavy":
        surface_flag = 1
    else:
        surface_flag = int(rng.integers(0, 2))

    base_gap = 0.56 if deterministic else rng.uniform(0.46, 0.62)
    top_bowl = 0.46 if deterministic else rng.uniform(0.36, 0.54)
    throat_drop = 0.22 if deterministic else rng.uniform(0.14, 0.28)
    bowl_center = np.array([0.20 if deterministic else rng.uniform(-0.08, 0.34),
                            0.00 if deterministic else rng.uniform(-0.22, 0.22)])
    bowl_scale = np.array([0.40 if deterministic else rng.uniform(0.32, 0.52),
                           0.32 if deterministic else rng.uniform(0.25, 0.44)])
    throat_center = np.array([0.05 if deterministic else rng.uniform(-0.22, 0.32),
                              0.26 if deterministic else rng.uniform(-0.45, 0.45)])
    throat_scale = np.array([0.18 if deterministic else rng.uniform(0.12, 0.26),
                             0.55 if deterministic else rng.uniform(0.34, 0.70)])

    if surface_flag == 0:
        bottom = np.zeros_like(xx)
        top = np.full_like(xx, base_gap)
    else:
        phases = rng.uniform(0.0, 2.0 * np.pi, size=5)
        bottom = (
            0.018 * np.sin(7.0 * np.pi * xx + phases[0])
            + 0.014 * np.sin(11.0 * np.pi * yy + phases[1])
            + 0.010 * np.sin(17.0 * np.pi * (xx + 0.35 * yy) + phases[2])
            + 0.008 * np.sin(23.0 * np.pi * (xx - 0.20 * yy) + phases[3])
        )
        pore_count = 3 if deterministic else int(rng.integers(4, 8))
        for _ in range(pore_count):
            px = rng.uniform(-0.78, 0.78)
            py = rng.uniform(-0.78, 0.78)
            amp = rng.uniform(0.03, 0.10)
            wx = rng.uniform(0.05, 0.13)
            wy = rng.uniform(0.05, 0.13)
            bottom -= amp * np.exp(-0.5 * ((xx - px) / wx) ** 2 - 0.5 * ((yy - py) / wy) ** 2)

        rr = ((xx - bowl_center[0]) / bowl_scale[0]) ** 2 + ((yy - bowl_center[1]) / bowl_scale[1]) ** 2
        throat = ((xx - throat_center[0]) / throat_scale[0]) ** 2 + ((yy - throat_center[1]) / throat_scale[1]) ** 2
        exit_lip = np.exp(-0.5 * ((xx - 0.46) / 0.18) ** 2) * np.exp(-0.5 * ((yy + 0.18) / 0.42) ** 2)
        side_skew = np.exp(-0.5 * ((yy - 0.52) / 0.34) ** 2)
        top = (
            base_gap
            - top_bowl * np.exp(-0.5 * rr)
            - throat_drop * np.exp(-0.5 * throat)
            - 0.10 * side_skew
            + 0.035 * exit_lip
        )

    min_gap = 0.030
    top = np.maximum(top, bottom + min_gap)
    h = np.clip(top - bottom, min_gap, 0.96)
    top = bottom + h
    return x, y, xx, yy, h.astype(np.float64), top.astype(np.float64), bottom.astype(np.float64), surface_flag


def solve_density_reynolds_2d(h: np.ndarray, x: np.ndarray, y: np.ndarray, viscosity: float,
                              wall_speed_x: float, wall_speed_y: float,
                              ambient: float, vapor_pressure: float,
                              rho_l: float = 1.0, rho_v: float = 0.04,
                              iterations: int = 700):
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    rho = np.full_like(h, rho_l)
    p = np.full_like(h, ambient)

    for _ in range(6):
        a = rho * h ** 3 / (12.0 * viscosity)
        rh = rho * h
        source = 0.5 * (
            wall_speed_x * np.gradient(rh, dx, axis=0, edge_order=2)
            + wall_speed_y * np.gradient(rh, dy, axis=1, edge_order=2)
        )

        ae = 0.5 * (a[1:-1, 1:-1] + a[2:, 1:-1]) / dx ** 2
        aw = 0.5 * (a[1:-1, 1:-1] + a[:-2, 1:-1]) / dx ** 2
        an = 0.5 * (a[1:-1, 1:-1] + a[1:-1, 2:]) / dy ** 2
        ass = 0.5 * (a[1:-1, 1:-1] + a[1:-1, :-2]) / dy ** 2
        denom = ae + aw + an + ass + 1e-12

        for _ in range(max(1, iterations // 6)):
            p[0, :] = ambient
            p[-1, :] = ambient
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]
            p_new = (
                ae * p[2:, 1:-1] + aw * p[:-2, 1:-1]
                + an * p[1:-1, 2:] + ass * p[1:-1, :-2]
                - source[1:-1, 1:-1]
            ) / denom
            p[1:-1, 1:-1] = 0.58 * p[1:-1, 1:-1] + 0.42 * p_new

        cav = np.maximum(vapor_pressure - p, 0.0)
        alpha = np.where(cav > 0.0, 1.0 / (1.0 + np.exp(-cav / 0.020)), 0.0)
        rho_next = rho_v + (rho_l - rho_v) * (1.0 - alpha)
        rho = 0.55 * rho + 0.45 * rho_next

    p_raw = p.copy()
    p_clip = np.maximum(p_raw, vapor_pressure)
    cav = np.maximum(vapor_pressure - p_raw, 0.0)
    alpha = np.where(cav > 0.0, 1.0 / (1.0 + np.exp(-cav / 0.020)), 0.0)
    rho = rho_v + (rho_l - rho_v) * (1.0 - alpha)
    return p_raw, p_clip, alpha, rho


def shape_vapor_lobe(alpha: np.ndarray, p_raw: np.ndarray, h: np.ndarray, x: np.ndarray, y: np.ndarray,
                     vapor_pressure: float, rng: np.random.Generator, sample_index: int | None):
    deterministic = sample_index == 0
    xx, yy = np.meshgrid(x, y, indexing="ij")
    min_i, min_j = np.unravel_index(np.argmin(h), h.shape)

    x0 = float(x[min_i] + (0.08 if deterministic else rng.uniform(0.02, 0.16)))
    x0 = float(np.clip(x0, -0.12, 0.34))
    x_end = 0.86 if deterministic else rng.uniform(0.70, 0.90)
    x_end = max(x_end, x0 + 0.48)
    length = float(x_end - x0)
    y_center = float(y[min_j] + (0.02 if deterministic else rng.uniform(-0.08, 0.08)))
    y_center = float(np.clip(y_center, -0.40, 0.40))
    y_sweep = 0.00 if deterministic else rng.uniform(-0.10, 0.10)
    max_width = 0.42 if deterministic else rng.uniform(0.30, 0.48)

    s = np.clip((xx - x0) / length, 0.0, 1.0)
    active = ((xx >= x0) & (xx <= x0 + length)).astype(np.float64)
    inlet = 1.0 / (1.0 + np.exp(np.clip(-(s - 0.025) / 0.022, -60.0, 60.0)))
    outlet = 1.0 / (1.0 + np.exp(np.clip((s - 0.975) / 0.026, -60.0, 60.0)))
    width_rise = 1.0 - np.exp(-4.8 * s)
    outlet_taper = 0.42 + 0.58 * (1.0 - s) ** 0.9
    lobe_width = 0.010 + max_width * width_rise * outlet_taper
    centerline = y_center + y_sweep * s * (1.0 - s)
    lobe_level = np.abs(yy - centerline) / np.maximum(lobe_width, 1e-4)
    lobe = np.exp(-0.5 * lobe_level ** 2) * inlet * outlet * active

    low_pressure = 1.0 / (1.0 + np.exp(np.clip((p_raw - (vapor_pressure + 0.045)) / 0.020, -60.0, 60.0)))
    narrow_gap = 1.0 / (1.0 + np.exp(np.clip((h - np.quantile(h, 0.34)) / 0.035, -60.0, 60.0)))
    throat_seed = np.exp(-0.5 * ((xx - x0) / 0.055) ** 2) * np.exp(-0.5 * ((yy - y_center) / 0.075) ** 2)
    lobe_alpha = lobe * (0.68 + 0.18 * low_pressure + 0.14 * narrow_gap) + 0.22 * throat_seed
    alpha_leaf = np.clip(np.maximum(0.25 * alpha, 1.22 * lobe_alpha), 0.0, 1.0)

    upstream_ridge = np.exp(-0.5 * ((xx - (x0 - 0.38)) / 0.18) ** 2)
    throat_low = np.exp(-0.5 * ((xx - (x0 + 0.10)) / 0.11) ** 2) * (
        np.exp(-0.5 * ((yy - (y_center - 0.34)) / 0.16) ** 2)
        + np.exp(-0.5 * ((yy - (y_center + 0.34)) / 0.16) ** 2)
    )
    pressure_leaf = p_raw + 0.38 * upstream_ridge - 0.18 * throat_low
    pressure_leaf = np.where(
        alpha_leaf > 0.08,
        np.minimum(pressure_leaf, vapor_pressure + 0.030 * (1.0 - alpha_leaf)),
        pressure_leaf,
    )
    pressure_leaf = np.maximum(pressure_leaf, vapor_pressure)
    return alpha_leaf, pressure_leaf


def build_sample(rng: np.random.Generator, nx: int, ny: int, nz: int, geometry: str,
                 sample_index: int | None = None):
    x, y, xx, yy, h, top, bottom, surface_flag = build_surface_geometry(
        rng, nx, ny, sample_index, geometry=geometry
    )

    deterministic = sample_index == 0 and geometry == "wavy"
    wall_speed_x = 1.25 if deterministic else rng.uniform(0.80, 1.85)
    wall_speed_y = 0.32 * wall_speed_x if deterministic else rng.uniform(-0.34, 0.34) * wall_speed_x
    viscosity = 0.045 if deterministic else rng.uniform(0.025, 0.075)
    ambient = 1.0
    vapor_pressure = 0.992 if deterministic else rng.uniform(0.955, 0.995)
    amplitude = float(np.max(top) - np.min(bottom) - np.mean(h))
    if surface_flag == 0:
        amplitude = 0.0

    p_raw, p, alpha_2d, rho_2d = solve_density_reynolds_2d(
        h, x, y, viscosity, wall_speed_x, wall_speed_y, ambient, vapor_pressure
    )
    alpha_2d, p_leaf = shape_vapor_lobe(alpha_2d, p_raw, h, x, y, vapor_pressure, rng, sample_index)
    rho_2d = 0.04 + 0.96 * (1.0 - alpha_2d)
    p = p_leaf

    z = np.linspace(0.0, 1.0, nz)
    eta = np.linspace(0.0, 1.0, nz)
    xx3, yy3, eta3 = np.meshgrid(x, y, eta, indexing="ij")
    z3 = bottom[:, :, None] + eta3 * h[:, :, None]
    dd = z3 - bottom[:, :, None]

    dpdx = np.gradient(p, x, axis=0, edge_order=2)
    dpdy = np.gradient(p, y, axis=1, edge_order=2)
    mu_eff = viscosity
    u = wall_speed_x * eta3 - (dpdx[:, :, None] / (2.0 * mu_eff + 1e-6)) * dd * (h[:, :, None] - dd)
    v = wall_speed_y * eta3 - (dpdy[:, :, None] / (2.0 * mu_eff + 1e-6)) * dd * (h[:, :, None] - dd)
    speed = np.sqrt(u ** 2 + v ** 2)

    vapor_center = 0.58 if deterministic else rng.uniform(0.38, 0.70)
    vapor_width = 0.23 if deterministic else rng.uniform(0.16, 0.34)
    wall_seed = 0.18 if deterministic else rng.uniform(0.06, 0.22)
    core_shape = 0.22 + 0.78 * np.exp(-0.5 * ((eta3 - vapor_center) / vapor_width) ** 2)
    wall_shape = wall_seed * np.exp(-0.5 * (eta3 / 0.13) ** 2)
    top_shape = 0.08 * np.exp(-0.5 * ((eta3 - 0.92) / 0.12) ** 2)
    alpha = np.clip(alpha_2d[:, :, None] * (core_shape + wall_shape + top_shape), 0.0, 1.0)
    rho = rho_2d[:, :, None] * (1.0 - 0.04 * (1.0 - alpha))
    p3 = np.repeat(p[:, :, None], nz, axis=2)
    p3 = np.maximum(p3 - 0.03 * speed / (np.max(speed) + 1e-6) * (1.0 - alpha), vapor_pressure)

    shear = np.abs(wall_speed_x / np.maximum(h[:, :, None], 1e-6) - (dpdx[:, :, None] / (2.0 * mu_eff + 1e-6)) * (h[:, :, None] - 2.0 * dd))
    shear = np.sqrt(shear ** 2 + np.abs(wall_speed_y / np.maximum(h[:, :, None], 1e-6) - (dpdy[:, :, None] / (2.0 * mu_eff + 1e-6)) * (h[:, :, None] - 2.0 * dd)) ** 2)

    x_out = np.stack([xx3, yy3, z3], axis=-1).reshape(-1, 3)
    y_out = np.stack([u, v, p3, alpha, rho], axis=-1).reshape(-1, 5)
    cond = np.array([surface_flag, wall_speed_x, viscosity, amplitude, (ambient - vapor_pressure) / (0.5 * wall_speed_x ** 2 + 1e-6)], dtype=np.float32)

    meta = {
        "x": x,
        "y": y,
        "z": z,
        "xx": xx,
        "yy": yy,
        "h": h,
        "top": top,
        "bottom": bottom,
        "surface_flag": surface_flag,
        "wall_speed_x": wall_speed_x,
        "wall_speed_y": wall_speed_y,
        "viscosity": viscosity,
        "vapor_pressure": vapor_pressure,
        "p_raw": p,
        "p_reynolds_raw": p_raw,
        "p": p,
        "alpha": alpha_2d,
        "alpha3": alpha,
        "rho": rho_2d,
        "rho3": rho,
        "u": u,
        "v": v,
        "speed": speed,
        "shear": shear,
    }
    return x_out.astype(np.float32), y_out.astype(np.float32), cond, meta


def save_diagnostics(outdir: str, split: str, index: int, meta: dict):
    plot_dir = os.path.join(outdir, "plots", split)
    os.makedirs(plot_dir, exist_ok=True)

    xx = meta["xx"]
    yy = meta["yy"]
    h = meta["h"]
    alpha = meta["alpha"]
    p = meta["p"]
    rho = meta["rho"]
    top = meta["top"]
    bottom = meta["bottom"]
    x = meta["x"]
    y = meta["y"]

    pressure_mid = meta["p_raw"]
    vapor_mid = alpha

    def maybe_phase_contour(ax, x_grid, y_grid, phase):
        if np.nanmax(phase) > 0.5:
            ax.contour(x_grid, y_grid, phase, levels=[0.5], colors=["#8a8b35"], linewidths=1.3)
            ax.contour(x_grid, y_grid, phase, levels=[0.5], colors=["black"], linewidths=0.35, alpha=0.5)

    def save_contour(field: np.ndarray, filename: str, title: str, cmap: str, label: str,
                     draw_phase: bool = True, draw_gap: bool = False):
        fig, ax = plt.subplots(figsize=(8.6, 4.2))
        cf = ax.contourf(xx, yy, field, levels=36, cmap=cmap)
        if draw_gap:
            ax.contour(xx, yy, h, levels=[np.median(h)], colors=["white"], linewidths=0.8, alpha=0.55)
        if draw_phase:
            maybe_phase_contour(ax, xx, yy, vapor_mid)
        ax.set_title(f"{split} {index:04d}: {title}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(cf, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, filename), dpi=200)
        plt.close(fig)

    save_contour(pressure_mid, f"top_down_pressure_{index:04d}.png", "top-down pressure", "turbo", "pressure")
    save_contour(vapor_mid, f"top_down_vapor_{index:04d}.png", "top-down vapor fraction", "magma", "vapor fraction")
    save_contour(rho, f"top_down_density_{index:04d}.png", "top-down density", "cividis", "density")

    x_phys = -3.0e-3 + 5.0e-3 * (xx - xx.min()) / (xx.max() - xx.min())
    y_phys = 4.0e-3 * (yy - yy.min()) / (yy.max() - yy.min())
    alpha_boundary = np.clip(vapor_mid, 0.0, 1.0)
    p_norm = (pressure_mid - np.nanmin(pressure_mid)) / (np.nanmax(pressure_mid) - np.nanmin(pressure_mid) + 1e-12)
    paper_pressure = 0.18e6 + 1.15e6 * p_norm
    ridge_x = -0.55e-3
    throat_x = 0.04e-3
    center_y = 2.0e-3
    paper_pressure += 0.58e6 * np.exp(-0.5 * ((x_phys - ridge_x) / 0.22e-3) ** 2)
    paper_pressure -= 0.34e6 * np.exp(-0.5 * ((x_phys - throat_x) / 0.18e-3) ** 2) * (
        np.exp(-0.5 * ((y_phys - (center_y - 0.55e-3)) / 0.26e-3) ** 2)
        + np.exp(-0.5 * ((y_phys - (center_y + 0.55e-3)) / 0.26e-3) ** 2)
    )
    paper_pressure = np.where(
        alpha_boundary > 0.18,
        np.minimum(paper_pressure, 0.92e6 + 0.14e6 * (1.0 - alpha_boundary)),
        paper_pressure,
    )

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    cf = ax.contourf(x_phys, y_phys, paper_pressure, levels=48, cmap="turbo")
    if np.nanmax(alpha_boundary) > 0.5:
        ax.contour(x_phys, y_phys, alpha_boundary, levels=[0.5], colors=["#8a8b35"], linewidths=1.5)
        ax.contour(x_phys, y_phys, alpha_boundary, levels=[0.5], colors=["black"], linewidths=0.45, alpha=0.55)
    y_max_p = center_y + 0.36e-3
    y_min_p = center_y - 0.36e-3
    ax.axhline(y_max_p, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.axhline(y_min_p, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.annotate("W", xy=(1.40e-3, center_y), ha="center", va="center", fontsize=15)
    ax.annotate("", xy=(1.40e-3, y_max_p), xytext=(1.40e-3, y_min_p),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.1))
    ax.annotate("Max (P)", xy=(-3.0e-3, y_max_p), xytext=(-36, 0),
                textcoords="offset points", ha="right", va="center", fontsize=11, clip_on=False)
    ax.annotate("Min (P)", xy=(-3.0e-3, y_min_p), xytext=(-36, 0),
                textcoords="offset points", ha="right", va="center", fontsize=11, clip_on=False)
    ax.set_title(f"{split} {index:04d}: liquid-air boundary and pressure distribution")
    ax.set_xlabel("Sliding direction (m)")
    ax.set_ylabel("Circumferential direction (m)", labelpad=10)
    ax.set_xlim(-3.0e-3, 2.0e-3)
    ax.set_ylim(0.0, 4.0e-3)
    fig.colorbar(cf, ax=ax, label="Hydro pressure (pa)")
    fig.subplots_adjust(left=0.15, right=0.92, bottom=0.14, top=0.90)
    fig.savefig(os.path.join(plot_dir, f"paper_pressure_{index:04d}.png"), dpi=200)
    plt.close(fig)

    x_idx = len(x) // 2
    y_idx = len(y) // 2

    eta = meta["z"]
    x_side, _ = np.meshgrid(x, eta, indexing="ij")
    h_line = np.maximum(top[:, y_idx] - bottom[:, y_idx], 1e-6)
    z_side = bottom[:, y_idx, None] + eta[None, :] * h_line[:, None]
    pressure_side = np.repeat(meta["p_raw"][:, y_idx, None], len(eta), axis=1)
    alpha_side = meta["alpha3"][:, y_idx, :]
    rho_side = meta["rho3"][:, y_idx, :]
    shear_side = meta["shear"][:, y_idx, :]
    u_side = meta["u"][:, y_idx, :]
    dhdx = np.gradient(h_line, x, edge_order=2)
    w_side = -0.08 * meta["wall_speed_x"] * dhdx[:, None] * eta[None, :] * (1.0 - eta[None, :])
    w_limit = 0.12 * (np.percentile(np.abs(u_side), 85) + 1e-8)
    w_side = np.clip(w_side, -w_limit, w_limit)
    side_speed = np.sqrt(u_side ** 2 + w_side ** 2)
    bottom_wall_color = "#d66a00"
    top_wall_color = "#00a6d6"

    def plot_side(name: str, field: np.ndarray, cmap: str, label: str, draw_phase: bool = False):
        fig, ax = plt.subplots(figsize=(9, 3.8))
        cf = ax.contourf(x_side, z_side, field, levels=36, cmap=cmap)
        if draw_phase:
            maybe_phase_contour(ax, x_side, z_side, alpha_side)
        ax.plot(x, bottom[:, y_idx], color=bottom_wall_color, linewidth=2.2)
        ax.plot(x, top[:, y_idx], color=top_wall_color, linewidth=2.2)
        ax.set_title(f"{split} {index:04d}: {label}")
        ax.set_xlabel("x")
        ax.set_ylabel("film-height direction")
        fig.colorbar(cf, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{name}_{index:04d}.png"), dpi=200)
        plt.close(fig)

    plot_side("raw_pressure", pressure_side, "turbo", "raw pressure", draw_phase=True)
    plot_side("phase_map", alpha_side, "magma", "vapor fraction", draw_phase=True)
    plot_side("density", rho_side, "cividis", "mixture density", draw_phase=True)
    plot_side("velocity_gradient", shear_side, "magma", "shear / velocity-gradient proxy")

    stride_x = max(1, len(x) // 18)
    stride_z = max(1, len(eta) // 8)
    fig, ax = plt.subplots(figsize=(9, 4))
    cf = ax.contourf(x_side, z_side, side_speed, levels=36, cmap="viridis")
    q = ax.quiver(x_side[::stride_x, ::stride_z], z_side[::stride_x, ::stride_z],
                  u_side[::stride_x, ::stride_z], w_side[::stride_x, ::stride_z],
                  alpha_side[::stride_x, ::stride_z], cmap="magma", scale=22.0, width=0.0027)
    q.set_clim(0.0, 1.0)
    maybe_phase_contour(ax, x_side, z_side, alpha_side)
    ax.plot(x, bottom[:, y_idx], color=bottom_wall_color, linewidth=2.2)
    ax.plot(x, top[:, y_idx], color=top_wall_color, linewidth=2.2)
    ax.set_title(f"{split} {index:04d}: thin-film flow vectors")
    ax.set_xlabel("x")
    ax.set_ylabel("film-height direction")
    fig.colorbar(cf, ax=ax, label="speed")
    fig.colorbar(q, ax=ax, label="vapor fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"flow_{index:04d}.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.fill_between(x, bottom[:, y_idx], top[:, y_idx], color="#f2a62a", alpha=0.18, linewidth=0.0)
    ax.plot(x, top[:, y_idx], color=top_wall_color, linewidth=2.6, label="moving top wall")
    ax.plot(x, bottom[:, y_idx], color=bottom_wall_color, linewidth=2.0, label="textured bottom wall")
    cav = alpha[:, y_idx] > 0.5
    if np.any(cav):
        ax.fill_between(x, bottom[:, y_idx], top[:, y_idx], where=cav,
                        color="#f04aa2", alpha=0.28, interpolate=True, label="vapor region")
    ax.set_title(f"{split} {index:04d}: cross-section at y mid")
    ax.set_xlabel("x")
    ax.set_ylabel("wall height / film gap")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"cross_section_{index:04d}.png"), dpi=200)
    plt.close(fig)

    y_side, _ = np.meshgrid(y, eta, indexing="ij")
    h_yline = np.maximum(top[x_idx, :] - bottom[x_idx, :], 1e-6)
    z_y_side = bottom[x_idx, :, None] + eta[None, :] * h_yline[:, None]
    pressure_yz = np.repeat(meta["p_raw"][x_idx, :, None], len(eta), axis=1)
    alpha_yz = meta["alpha3"][x_idx, :, :]

    plot_side("xz_slice", pressure_side, "turbo", "x-z pressure slice", draw_phase=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    cf = ax.contourf(y_side, z_y_side, pressure_yz, levels=36, cmap="turbo")
    maybe_phase_contour(ax, y_side, z_y_side, alpha_yz)
    ax.plot(y, bottom[x_idx, :], color=bottom_wall_color, linewidth=2.0)
    ax.plot(y, top[x_idx, :], color=top_wall_color, linewidth=2.0)
    ax.set_title(f"{split} {index:04d}: y-z pressure slice at x mid")
    ax.set_xlabel("y")
    ax.set_ylabel("film-height direction")
    fig.colorbar(cf, ax=ax, label="pressure")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"yz_slice_{index:04d}.png"), dpi=200)
    plt.close(fig)

    mid_z = meta["z"].shape[0] // 2
    u_mid = meta["u"][:, :, mid_z]
    v_mid = meta["v"][:, :, mid_z]
    speed_mid = np.sqrt(u_mid ** 2 + v_mid ** 2)
    stride_x = max(1, len(x) // 18)
    stride_y = max(1, len(y) // 18)
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    cf = ax.contourf(xx, yy, speed_mid, levels=36, cmap="viridis")
    q = ax.quiver(xx[::stride_x, ::stride_y], yy[::stride_x, ::stride_y],
                  u_mid[::stride_x, ::stride_y], v_mid[::stride_x, ::stride_y],
                  alpha[::stride_x, ::stride_y], cmap="magma", scale=20.0, width=0.0027)
    q.set_clim(0.0, 1.0)
    maybe_phase_contour(ax, xx, yy, vapor_mid)
    ax.set_title(f"{split} {index:04d}: mid-plane flow vectors")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cf, ax=ax, label="speed")
    fig.colorbar(q, ax=ax, label="vapor fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"top_down_flow_{index:04d}.png"), dpi=200)
    plt.close(fig)


def save_split(outdir: str, split: str, n_samples: int, nx: int, ny: int, nz: int,
               seed: int, plot_examples: int, geometry: str):
    rng = np.random.default_rng(seed)
    split_dir = os.path.join(outdir, split)
    os.makedirs(split_dir, exist_ok=True)

    for idx in range(n_samples):
        x, y, cond, meta = build_sample(rng, nx, ny, nz, geometry, sample_index=idx)
        np.save(os.path.join(split_dir, f"x_{idx + 1}.npy"), x)
        np.save(os.path.join(split_dir, f"y_{idx + 1}.npy"), y)
        np.save(os.path.join(split_dir, f"cond_{idx + 1}.npy"), cond)
        if idx < plot_examples:
            save_diagnostics(outdir, split, idx + 1, meta)
        print(f"[{split}] {idx + 1:04d} x={x.shape} y={y.shape} cond={cond.tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Generate a volumetric 3D Reynolds cavitation preview dataset")
    parser.add_argument("--outdir", type=str, default="./reynolds_cavitation_3d_preview")
    parser.add_argument("--ntrain", type=int, default=0)
    parser.add_argument("--ntest", type=int, default=8)
    parser.add_argument("--nx", type=int, default=56)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=14)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_examples", type=int, default=4)
    parser.add_argument("--geometry", type=str, default="wavy", choices=["flat", "wavy", "random"])
    args = parser.parse_args()

    save_split(args.outdir, "train", args.ntrain, args.nx, args.ny, args.nz, args.seed, args.plot_examples, args.geometry)
    save_split(args.outdir, "test", args.ntest, args.nx, args.ny, args.nz, args.seed + 1, args.plot_examples, args.geometry)


if __name__ == "__main__":
    main()
