#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def channel_profile(x: np.ndarray, surface_flag: int, amplitude: float, frequency: int, phase: float,
                    base_gap: float, top_height: float, throat_center: float,
                    throat_width: float, wedge: float, pore_power: float = 2.0,
                    waviness_scale: float = 1.0,
                    pore_sign: float = 1.0,
                    top_curvature: float = 0.0,
                    roughness_scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return lower wall, upper ring wall, lower-wall slope, and local film thickness."""
    ring_left = 0.16
    ring_right = 0.84
    shoulder_drop = 0.12
    s_ring = np.clip((x - ring_left) / (ring_right - ring_left), 0.0, 1.0)
    in_ring = (x >= ring_left) & (x <= ring_right)
    cup = np.sin(np.pi * s_ring) ** 2
    top = np.where(in_ring, top_height - shoulder_drop - top_curvature * cup, top_height)

    if surface_flag == 0:
        bottom = top_height - base_gap * np.ones_like(x)
        slope = np.zeros_like(x)
    else:
        throat = np.exp(-0.5 * np.abs((x - throat_center) / throat_width) ** pore_power)
        waviness = np.sin(2.0 * np.pi * frequency * x + phase)
        secondary = np.sin(2.0 * np.pi * (frequency + 1) * x + 0.7 * phase)
        bottom = top_height - base_gap * (1.0 + wedge * (x - 0.5))
        bottom = bottom - pore_sign * amplitude * throat
        bottom -= waviness_scale * (0.035 * base_gap * waviness + 0.015 * base_gap * secondary)
        tri1 = 2.0 * np.abs(2.0 * ((11.0 * x + phase / (2.0 * np.pi)) % 1.0) - 1.0) - 1.0
        tri2 = 2.0 * np.abs(2.0 * ((19.0 * x + 0.37 + phase / (2.0 * np.pi)) % 1.0) - 1.0) - 1.0
        bottom -= roughness_scale * 0.018 * base_gap * (tri1 + 0.55 * tri2)

        rel = (x - throat_center) / throat_width
        dthroat = throat * (-0.5 * pore_power * np.sign(rel) * np.abs(rel) ** (pore_power - 1.0) / throat_width)
        dwaviness = 2.0 * np.pi * frequency * np.cos(2.0 * np.pi * frequency * x + phase)
        dsecondary = 2.0 * np.pi * (frequency + 1) * np.cos(2.0 * np.pi * (frequency + 1) * x + 0.7 * phase)
        dbottom_dx = -base_gap * wedge - pore_sign * amplitude * dthroat
        dbottom_dx -= waviness_scale * (0.035 * base_gap * dwaviness + 0.015 * base_gap * dsecondary)
        slope = dbottom_dx

    gap = top - bottom
    gap = np.clip(gap, 0.08, 0.96)
    bottom = top - gap
    return bottom, top, slope, gap


def solve_reynolds_pressure(x_coords: np.ndarray, gap_1d: np.ndarray, viscosity: float,
                            wall_speed: float, ambient_pressure: float,
                            vapor_pressure: float, pressure_scale: float,
                            density_liquid: float = 1.0,
                            density_vapor: float = 0.02) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Solve a 1D density-aware Reynolds balance and apply a cavitation floor."""
    h = np.maximum(gap_1d, 1e-6)
    mu_eff = viscosity * pressure_scale
    density = np.full_like(h, density_liquid)
    pressure_raw = np.full_like(h, ambient_pressure)

    for _ in range(8):
        mass_flow = 0.5 * wall_speed * np.trapz(1.0 / h ** 2, x_coords)
        mass_flow /= np.trapz(1.0 / (np.maximum(density, 1e-6) * h ** 3), x_coords)
        dpdx = 12.0 * mu_eff * (
            0.5 * wall_speed / h ** 2 - mass_flow / (np.maximum(density, 1e-6) * h ** 3)
        )
        pressure_raw = ambient_pressure + np.concatenate([[0.0], np.cumsum(0.5 * (
            dpdx[:-1] + dpdx[1:]) * np.diff(x_coords))])

        # Re-anchor to ambient pressure at both ends. This keeps samples comparable while
        # preserving the pressure peak/drop created by the film-thickness gradient.
        end_drift = pressure_raw[-1] - ambient_pressure
        pressure_raw = pressure_raw - end_drift * (x_coords - x_coords[0]) / (x_coords[-1] - x_coords[0])

        cavitation_index = np.maximum(vapor_pressure - pressure_raw, 0.0)
        alpha_x = 1.0 / (1.0 + np.exp(-cavitation_index / 0.025))
        alpha_x = np.where(cavitation_index > 0.0, alpha_x, 0.0)
        density_next = density_vapor + (density_liquid - density_vapor) * (1.0 - alpha_x)
        density = 0.55 * density + 0.45 * density_next

    mass_flow = 0.5 * wall_speed * np.trapz(1.0 / h ** 2, x_coords)
    mass_flow /= np.trapz(1.0 / (np.maximum(density, 1e-6) * h ** 3), x_coords)
    dpdx = 12.0 * mu_eff * (
        0.5 * wall_speed / h ** 2 - mass_flow / (np.maximum(density, 1e-6) * h ** 3)
    )
    pressure_raw = ambient_pressure + np.concatenate([[0.0], np.cumsum(0.5 * (
        dpdx[:-1] + dpdx[1:]) * np.diff(x_coords))])
    end_drift = pressure_raw[-1] - ambient_pressure
    pressure_raw = pressure_raw - end_drift * (x_coords - x_coords[0]) / (x_coords[-1] - x_coords[0])

    pressure = np.maximum(pressure_raw, vapor_pressure)
    cavitation_index = np.maximum(vapor_pressure - pressure_raw, 0.0)
    return pressure_raw, pressure, cavitation_index, dpdx, mass_flow, density


def build_sample(rng: np.random.Generator, nx: int, ny: int, geometry: str, sample_index: int | None = None):
    x_coords = np.linspace(0.0, 1.0, nx)
    eta_coords = np.linspace(0.0, 1.0, ny)
    xx, eta = np.meshgrid(x_coords, eta_coords, indexing='ij')

    if geometry == "flat":
        surface_flag = 0
    elif geometry == "wavy":
        surface_flag = 1
    else:
        surface_flag = int(rng.integers(0, 2))

    deterministic_pore = geometry == "wavy" and sample_index == 0

    base_gap = rng.uniform(0.52, 0.68)
    amplitude = 0.0 if surface_flag == 0 else rng.uniform(0.16, 0.34)
    frequency = 1 if surface_flag == 0 else int(rng.integers(1, 4))
    phase = float(rng.uniform(0.0, 2.0 * np.pi)) if surface_flag == 1 else 0.0
    throat_center = float(rng.uniform(0.30, 0.60)) if surface_flag == 1 else 0.5
    throat_width = float(rng.uniform(0.055, 0.13)) if surface_flag == 1 else 0.10
    wedge = float(rng.uniform(-0.28, 0.20)) if surface_flag == 1 else 0.0
    pore_power = 2.0
    waviness_scale = 1.0
    pore_sign = 1.0
    top_curvature = rng.uniform(0.36, 0.62) if surface_flag == 1 else 0.0
    roughness_scale = rng.uniform(1.1, 1.9) if surface_flag == 1 else 0.0

    if deterministic_pore:
        base_gap = 0.50
        amplitude = 0.32
        frequency = 1
        phase = 0.0
        throat_center = 0.50
        throat_width = 0.075
        wedge = 0.0
        pore_power = 4.0
        waviness_scale = 0.0
        pore_sign = 1.0
        top_curvature = 0.62
        roughness_scale = 1.6

    wall_speed = rng.uniform(0.55, 1.85)
    viscosity = rng.uniform(0.02, 0.08)
    ambient_pressure = rng.uniform(0.92, 1.08)
    vapor_pressure = ambient_pressure - rng.uniform(0.035, 0.22)
    pressure_scale = rng.uniform(12.0, 34.0) if surface_flag == 1 else 1.0
    if deterministic_pore:
        wall_speed = 1.35
        viscosity = 0.045
        ambient_pressure = 1.0
        pressure_scale = 28.0
        vapor_pressure = 0.95
    cavitation_number = (ambient_pressure - vapor_pressure) / (0.5 * wall_speed ** 2 + 1e-6)
    top_height = 1.0

    bottom, top, slope, gap = channel_profile(xx, surface_flag, amplitude, frequency, phase,
                                              base_gap, top_height, throat_center,
                                              throat_width, wedge, pore_power=pore_power,
                                              waviness_scale=waviness_scale,
                                              pore_sign=pore_sign,
                                              top_curvature=top_curvature,
                                              roughness_scale=roughness_scale)
    yy = bottom + eta * gap
    dd = yy - bottom

    gap_1d = gap[:, 0]
    pressure_raw_1d, pressure_1d, cavitation_1d, dpdx_1d, mass_flow, density_1d = solve_reynolds_pressure(
        x_coords, gap_1d, viscosity, wall_speed, ambient_pressure, vapor_pressure, pressure_scale)
    dpdx = dpdx_1d[:, None]
    mu_eff = viscosity * pressure_scale

    # Moving-ring frame: lower textured wall is stationary, upper wall moves at wall_speed.
    velocity = wall_speed * (dd / gap) - (dpdx / (2.0 * mu_eff + 1e-6)) * dd * (gap - dd)
    shear = np.abs(wall_speed / gap - (dpdx / (2.0 * mu_eff + 1e-6)) * (gap - 2.0 * dd))

    alpha_x = 1.0 / (1.0 + np.exp(-(cavitation_1d[:, None]) / 0.025))
    alpha_x = np.where(cavitation_1d[:, None] > 0.0, alpha_x, 0.0)
    pocket_shape = 0.35 + 0.65 * np.exp(-0.5 * ((eta - 0.55) / 0.30) ** 2)
    wall_nucleation = 0.16 * alpha_x * np.exp(-0.5 * (eta / 0.16) ** 2)
    vapor_fraction = np.clip(alpha_x * pocket_shape + wall_nucleation, 0.0, 1.0)

    pressure = pressure_1d[:, None] - 0.015 * shear / (np.max(shear) + 1e-6) * (1.0 - vapor_fraction)
    pressure = np.maximum(pressure, vapor_pressure)
    density = np.repeat(density_1d[:, None], ny, axis=1)

    x = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)
    y = np.stack([velocity, pressure, vapor_fraction, shear, density], axis=-1).reshape(-1, 5)
    cond = np.array([surface_flag, wall_speed, viscosity, amplitude, cavitation_number], dtype=np.float32)
    meta = {
        "surface_flag": surface_flag,
        "amplitude": amplitude,
        "frequency": frequency,
        "phase": phase,
        "throat_center": throat_center,
        "throat_width": throat_width,
        "wedge": wedge,
        "wall_speed": wall_speed,
        "ambient_pressure": ambient_pressure,
        "vapor_pressure": vapor_pressure,
        "pressure_scale": pressure_scale,
        "pore_sign": pore_sign,
        "top_curvature": top_curvature,
        "roughness_scale": roughness_scale,
        "cavitation_number": cavitation_number,
        "mass_flow": mass_flow,
        "bottom": bottom.astype(np.float32),
        "top": top.astype(np.float32),
        "gap": gap.astype(np.float32),
        "u_grid": velocity.astype(np.float32),
        "p_grid": pressure.astype(np.float32),
        "p_raw_grid": np.repeat(pressure_raw_1d[:, None], ny, axis=1).astype(np.float32),
        "vapor_grid": vapor_fraction.astype(np.float32),
        "rho_grid": density.astype(np.float32),
        "rho_1d": density_1d.astype(np.float32),
        "shear_grid": shear.astype(np.float32),
    }
    return x.astype(np.float32), y.astype(np.float32), cond, meta


def save_diagnostics(outdir: str, split: str, index: int, x: np.ndarray, meta: dict):
    plot_dir = os.path.join(outdir, "plots", split)
    os.makedirs(plot_dir, exist_ok=True)

    nx = meta["u_grid"].shape[0]
    ny = meta["u_grid"].shape[1]
    xx = x[:, 0].reshape(nx, ny)
    yy = x[:, 1].reshape(nx, ny)
    u = meta["u_grid"]
    vapor = meta["vapor_grid"]
    density = meta["rho_grid"]
    bottom_wall_color = "tab:orange"
    top_wall_color = "tab:cyan"
    x_profile = np.linspace(xx.min(), xx.max(), nx)
    v = -np.cumsum(np.gradient(u, xx[:, 0], axis=0), axis=1)
    v = v - np.mean(v, axis=1, keepdims=True)

    stride_x = max(1, nx // 14)
    stride_y = max(1, ny // 10)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    speed = np.sqrt(u ** 2 + v ** 2)
    cf = ax.contourf(xx, yy, speed, levels=30, cmap="viridis")
    q = ax.quiver(xx[::stride_x, ::stride_y], yy[::stride_x, ::stride_y],
                  u[::stride_x, ::stride_y], v[::stride_x, ::stride_y],
                  vapor[::stride_x, ::stride_y], cmap="coolwarm", scale=10.0, width=0.003)
    q.set_clim(0.0, 1.0)
    ax.plot(x_profile, meta["bottom"], color=bottom_wall_color, linewidth=2.0)
    ax.plot(x_profile, meta["top"], color=top_wall_color, linewidth=2.0)
    ax.set_title(f"{split} sample {index}: thin-film flow vectors")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cf, ax=ax, label="speed")
    fig.colorbar(q, ax=ax, label="vapor fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"flow_{index:04d}.png"), dpi=200)
    plt.close(fig)

    du_dy = np.abs(np.gradient(u, axis=1))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    cf = ax.contourf(xx, yy, u, levels=30, cmap="coolwarm")
    ax.plot(x_profile, meta["bottom"], color=bottom_wall_color, linewidth=2.0)
    ax.plot(x_profile, meta["top"], color=top_wall_color, linewidth=2.0)
    ax.set_title(f"{split} sample {index}: velocity field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cf, ax=ax, label="u")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"velocity_{index:04d}.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    cf = ax.contourf(xx, yy, du_dy, levels=30, cmap="magma")
    ax.plot(x_profile, meta["bottom"], color=bottom_wall_color, linewidth=2.0)
    ax.plot(x_profile, meta["top"], color=top_wall_color, linewidth=2.0)
    ax.set_title(f"{split} sample {index}: |du/dy| discontinuity proxy")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cf, ax=ax, label="|du/dy|")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"velocity_gradient_{index:04d}.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    phase = (vapor >= 0.5).astype(np.float32)
    cf = ax.contourf(xx, yy, vapor, levels=np.linspace(0.0, 1.0, 21), cmap="coolwarm")
    ax.contour(xx, yy, phase, levels=[0.5], colors=["white"], linewidths=1.2)
    ax.plot(x_profile, meta["bottom"], color=bottom_wall_color, linewidth=2.5)
    ax.plot(x_profile, meta["top"], color=top_wall_color, linewidth=2.5)
    ax.set_title(f"{split} sample {index}: liquid-vapor side view")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cf, ax=ax, label="vapor fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"phase_map_{index:04d}.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    cf = ax.contourf(xx, yy, density, levels=30, cmap="cividis")
    ax.plot(x_profile, meta["bottom"], color=bottom_wall_color, linewidth=2.5)
    ax.plot(x_profile, meta["top"], color=top_wall_color, linewidth=2.5)
    ax.set_title(f"{split} sample {index}: mixture density")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cf, ax=ax, label="density")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"density_{index:04d}.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    p_raw = meta["p_raw_grid"]
    cf = ax.contourf(xx, yy, p_raw, levels=30, cmap="RdBu_r")
    ax.contour(xx, yy, vapor, levels=[0.5], colors=["white"], linewidths=1.2)
    ax.plot(x_profile, meta["bottom"], color=bottom_wall_color, linewidth=2.5)
    ax.plot(x_profile, meta["top"], color=top_wall_color, linewidth=2.5)
    ax.set_title(f"{split} sample {index}: Reynolds pressure before cavitation clipping")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(cf, ax=ax, label="raw pressure")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"raw_pressure_{index:04d}.png"), dpi=200)
    plt.close(fig)

    z_coords = np.linspace(0.0, 1.0, max(80, ny))
    x_top, z_top = np.meshgrid(x_profile, z_coords, indexing="ij")
    p_raw_1d = p_raw[:, 0]
    alpha_line = np.clip(np.mean(vapor, axis=1), 0.0, 1.0)
    pocket_half_width = 0.05 + 0.42 * alpha_line
    z_distance = np.abs(z_top - 0.5)
    alpha_top = alpha_line[:, None] / (1.0 + np.exp((z_distance - pocket_half_width[:, None]) / 0.018))
    pressure_top = p_raw_1d[:, None] - 0.18 * alpha_top * np.maximum(np.max(p_raw_1d) - p_raw_1d[:, None], 0.0)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    cf = ax.contourf(x_top, z_top, pressure_top, levels=36, cmap="turbo")
    if np.max(alpha_top) > 0.5:
        ax.contour(x_top, z_top, alpha_top, levels=[0.5], colors=["white"], linewidths=1.8)
        ax.contour(x_top, z_top, alpha_top, levels=[0.5], colors=["black"], linewidths=0.6, alpha=0.65)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(f"{split} sample {index}: top-down pressure and liquid-vapor boundary")
    ax.set_xlabel("sliding direction x")
    ax.set_ylabel("circumferential direction")
    fig.colorbar(cf, ax=ax, label="raw pressure")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"top_down_pressure_{index:04d}.png"), dpi=200)
    plt.close(fig)

    mid_x = nx // 2
    fig, ax1 = plt.subplots(figsize=(8, 3.5))
    ax1.plot(yy[mid_x, :], u[mid_x, :], color="tab:blue", linewidth=2.0, label="velocity")
    ax1.set_xlabel("y")
    ax1.set_ylabel("u", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    bottom_y = float(meta["bottom"][mid_x, 0])
    top_y = float(meta["top"][mid_x, 0])
    ax1.axvline(bottom_y, color=bottom_wall_color, linestyle="--", linewidth=1.5, label="lower wall")
    ax1.axvline(top_y, color=top_wall_color, linestyle="--", linewidth=1.5, label="moving ring")
    ax2 = ax1.twinx()
    ax2.plot(yy[mid_x, :], vapor[mid_x, :], color="tab:red", linewidth=2.0, label="vapor fraction")
    ax2.set_ylabel("vapor fraction", color="tab:red")
    ax2.tick_params(axis='y', labelcolor="tab:red")
    ax1.set_title(f"{split} sample {index}: cross section through centerline")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"cross_section_{index:04d}.png"), dpi=200)
    plt.close(fig)


def save_split(outdir: str, split: str, n_samples: int, nx: int, ny: int, geometry: str, seed: int,
               plot_examples: int = 0):
    rng = np.random.default_rng(seed)
    split_dir = os.path.join(outdir, split)
    os.makedirs(split_dir, exist_ok=True)

    for idx in range(n_samples):
        sample_geometry = geometry
        if geometry == "mixed":
            sample_geometry = "flat" if idx % 2 == 0 else "wavy"

        x, y, cond, meta = build_sample(rng, nx, ny, sample_geometry, sample_index=idx)
        np.save(os.path.join(split_dir, f"x_{idx + 1}.npy"), x)
        np.save(os.path.join(split_dir, f"y_{idx + 1}.npy"), y)
        np.save(os.path.join(split_dir, f"cond_{idx + 1}.npy"), cond)

        if idx < plot_examples and sample_geometry == "wavy":
            save_diagnostics(outdir, split, idx + 1, x, meta)

        print(f"[{split}] {idx + 1:04d} {sample_geometry:5s} x={x.shape} y={y.shape} cond={cond.tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Generate a thin-film hydrodynamic cavitation dataset")
    parser.add_argument("--outdir", type=str, default="./data/reynolds_cavitation_npys")
    parser.add_argument("--ntrain", type=int, default=64)
    parser.add_argument("--ntest", type=int, default=16)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--geometry", type=str, default="mixed", choices=["flat", "wavy", "mixed"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot_examples", type=int, default=2,
                        help="number of samples per split to plot")
    args = parser.parse_args()

    save_split(args.outdir, "train", args.ntrain, args.nx, args.ny, args.geometry, args.seed,
               plot_examples=args.plot_examples)
    save_split(args.outdir, "test", args.ntest, args.nx, args.ny, args.geometry, args.seed + 1,
               plot_examples=args.plot_examples)


if __name__ == "__main__":
    main()
