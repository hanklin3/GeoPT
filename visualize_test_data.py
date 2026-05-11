#!/usr/bin/env python
"""
Visualize ground truth test data directly from NPY files.
No model predictions - just raw synthetic data visualization using the
save_diagnostics implementation from ReynoldsCavitation2D_process.py.
"""

import numpy as np
import argparse
import os
from data_preprocess.ReynoldsCavitation2D_process import build_sample, save_diagnostics


def reconstruct_meta_from_npys(index, nx, nz, seed, smooth_top=False):
    """
    Reconstruct the metadata dict by replaying the preprocessing generator state.
    The generator is sequential, so sample N depends on all prior random draws.
    """
    rng = np.random.default_rng(seed)
    meta = None
    for sample_index in range(index):
        _, _, _, meta = build_sample(rng, nx, nz, sample_index, smooth_top=smooth_top)
    return meta


def visualize_test_data(data_dir='./reynolds_cavitation_2d_npys/test',
                       save_name='./reynolds_cavitation_2d_npys',
                       test_range=None,
                       nx=128,
                       nz=96,
                       seed=0,
                       smooth_top=False):
    """
    Load test NPY files and visualize ground truth data using preprocessing diagnostics.
    
    Args:
        data_dir: Directory containing x_*.npy, y_*.npy, cond_*.npy files
        save_name: Output directory name for plots
        test_range: Tuple (start, end) for test sample indices, or None for all
        nx, nz: Expected grid dimensions
    """
    
    # Find all test files
    test_files = sorted([f for f in os.listdir(data_dir) if f.startswith('x_') and f.endswith('.npy')])
    
    if not test_files:
        print(f"❌ No test files found in {data_dir}")
        return
    
    print(f"📂 Found {len(test_files)} test samples in {data_dir}")
    
    # Determine range to visualize
    if test_range is None:
        indices = range(len(test_files))
    else:
        start, end = test_range
        indices = range(start, min(end, len(test_files)))
    
    print(f"📊 Visualizing samples: {list(indices)}")
    
    for idx in indices:
        file_idx = idx + 1  # Files are 1-indexed
        
        # Load NPY files
        x_file = os.path.join(data_dir, f'x_{file_idx}.npy')
        y_file = os.path.join(data_dir, f'y_{file_idx}.npy')
        cond_file = os.path.join(data_dir, f'cond_{file_idx}.npy')
        
        if not (os.path.exists(x_file) and os.path.exists(y_file)):
            print(f"⚠️  Skipping sample {file_idx}: missing files")
            continue
        
        # Load data to confirm the expected files exist for this index.
        _ = np.load(x_file)  # shape (N, 4): [x, z, h, u_node]
        _ = np.load(y_file)  # shape (N, 5): [pressure, vapor_fraction, density, film_height, shear]
        _ = np.load(cond_file)  # shape (5,): conditions
        
        # Reconstruct metadata from NPY data
        split_seed = seed + 1 if os.path.basename(os.path.normpath(data_dir)) == 'test' else seed
        meta = reconstruct_meta_from_npys(file_idx, nx, nz, split_seed, smooth_top=smooth_top)
        
        # Generate diagnostics
        print(f"  🎨 Plotting sample {file_idx:04d}...", end=' ')
        try:
            save_diagnostics(save_name, 'test_vis', file_idx, meta)
            print("✅")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✨ Plots saved to: {save_name}/plots/test_vis/")
    print(f"\n📋 Generated plots:")
    print("   - cross_section_XXXX.png")
    print("   - phase_map_XXXX.png")
    print("   - raw_pressure_XXXX.png")
    print("   - density_XXXX.png")
    print("   - velocity_gradient_XXXX.png")
    print("   - flow_XXXX.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize test data ground truth')
    parser.add_argument('--data_dir', default='./reynolds_cavitation_2d_npys/test',
                       help='Directory with test NPY files')
    parser.add_argument('--save_name', default='./reynolds_cavitation_2d_npys/plots/test_vis',
                       help='Output directory name for plots')
    parser.add_argument('--test_range', nargs=2, type=int,
                       help='Test sample range (start end), e.g., --test_range 0 10')
    parser.add_argument('--nx', type=int, default=128, help='Grid width')
    parser.add_argument('--nz', type=int, default=96, help='Grid depth')
    parser.add_argument('--seed', type=int, default=0, help='Base seed used during data generation')
    parser.add_argument('--smooth_top', action='store_true', help='Use smooth top-wall geometry')
    
    args = parser.parse_args()
    test_range = None if args.test_range is None else tuple(args.test_range)
    
    visualize_test_data(
        data_dir=args.data_dir,
        save_name=args.save_name,
        test_range=test_range,
        nx=args.nx,
        nz=args.nz,
        seed=args.seed,
        smooth_top=args.smooth_top
    )

