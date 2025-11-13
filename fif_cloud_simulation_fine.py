#!/usr/bin/env python3
"""
FIF Cloud Condensate Simulation - FINE RESOLUTION

Creates high-resolution 3D cloud condensate (QC) field (512x512x256).
This script runs the fine resolution simulation after reviewing coarse results.

Domain: 6km x 6km x 2km
Resolution: 512x512x256 (~11.7m x 11.7m x 7.8m)
Parameters: H=0.4, C1=0.02, alpha=1.8
Anisotropy: spheroscale=10m, Hz=5/9, scale_metric_dim=23/9
Boundary conditions: Periodic in x,y; non-periodic in z
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scaleinvariance import FIF_ND, canonical_scale_metric

# Import functions from main simulation script
import sys
sys.path.insert(0, '.')
from fif_cloud_simulation import (
    load_reference_data, interpolate_to_grid, saturation_adjustment,
    normalize_and_scale_fif, save_to_netcdf, plot_vertically_integrated,
    plot_isosurfaces, DOMAIN_X, DOMAIN_Y, DOMAIN_Z, H, C1, ALPHA,
    SPHEROSCALE, HZ, SCALE_METRIC_DIM
)

def generate_fif_simulation(nx, ny, nz, domain_x, domain_y, domain_z,
                            H, C1, alpha, spheroscale, Hz, scale_metric_dim,
                            seed=None):
    """
    Generate 3D FIF simulation with anisotropic scaling.
    """
    print(f"\nGenerating FIF simulation ({nx}x{ny}x{nz})...")
    print(f"  Domain: {domain_x}m x {domain_y}m x {domain_z}m")
    print(f"  Resolution: {domain_x/nx:.1f}m x {domain_y/ny:.1f}m x {domain_z/nz:.1f}m")
    print(f"  Parameters: H={H}, C1={C1}, alpha={alpha}")
    print(f"  Anisotropy: spheroscale={spheroscale}m, Hz={Hz:.3f}, dim={scale_metric_dim:.3f}")

    if seed is not None:
        np.random.seed(seed)

    # Grid size for simulation
    size = (nx, ny, nz)

    # periodic=(True, True, False) means only z is doubled internally
    sim_size = (nx, ny, nz * 2)
    print(f"  Internal simulation size (for non-periodic z): {sim_size}")

    # Create coordinate grids (for output)
    grid_x = np.linspace(0, domain_x, nx, endpoint=False)
    grid_y = np.linspace(0, domain_y, ny, endpoint=False)
    grid_z = np.linspace(domain_z/(2*nz), domain_z, nz, endpoint=True)

    # Create anisotropic scale metric for the simulation domain
    dx = domain_x / nx
    spheroscale_grid = spheroscale / dx

    print(f"  Creating anisotropic scale metric...")
    scale_metric = canonical_scale_metric(sim_size, ls=spheroscale_grid, Hz=Hz)

    # Generate FIF with periodic in x,y and non-periodic in z
    print(f"  Running FIF_ND (this will take several minutes)...")
    fif_field = FIF_ND(
        size=size,
        alpha=alpha,
        C1=C1,
        H=H,
        periodic=(True, True, False),
        scale_metric=scale_metric,
        scale_metric_dim=scale_metric_dim,
        kernel_construction_method='LS2010'
    )

    print(f"  FIF field statistics:")
    print(f"    Mean: {np.mean(fif_field):.6f}")
    print(f"    Std: {np.std(fif_field):.6f}")
    print(f"    Min: {np.min(fif_field):.6f}")
    print(f"    Max: {np.max(fif_field):.6f}")

    return fif_field, grid_x, grid_y, grid_z


def main():
    """Main simulation workflow for fine resolution."""

    print("=" * 70)
    print("FIF CLOUD CONDENSATE SIMULATION - FINE RESOLUTION")
    print("=" * 70)

    # Load reference data
    ref_data = load_reference_data(max_height=DOMAIN_Z)

    # FINE RESOLUTION SIMULATION
    print("\n" + "=" * 70)
    print("FINE RESOLUTION SIMULATION (512x512x256)")
    print("=" * 70)
    print("\nWARNING: This will take several minutes to complete.")
    print("         Memory usage will be ~2-4 GB.")
    print()

    nx_fine, ny_fine, nz_fine = 512, 512, 256

    # Generate FIF simulation
    print("Starting fine resolution simulation...")
    fif_QT, grid_x, grid_y, grid_z = generate_fif_simulation(
        nx_fine, ny_fine, nz_fine,
        DOMAIN_X, DOMAIN_Y, DOMAIN_Z,
        H, C1, ALPHA,
        SPHEROSCALE, HZ, SCALE_METRIC_DIM,
        seed=42
    )

    # Interpolate reference profiles to simulation grid
    ref_interp = interpolate_to_grid(ref_data, grid_z)

    # Normalize and scale to match reference QT statistics
    QT_scaled = normalize_and_scale_fif(
        fif_QT,
        ref_interp['QT_mean'],
        ref_interp['QT_normalized_log_std']
    )

    print(f"\nScaled QT statistics:")
    print(f"  Mean: {np.mean(QT_scaled):.4f} g/kg")
    print(f"  Std: {np.std(QT_scaled):.4f} g/kg")
    print(f"  Min: {np.min(QT_scaled):.4f} g/kg")
    print(f"  Max: {np.max(QT_scaled):.4f} g/kg")

    # Apply saturation adjustment to get QC
    T_profile_celsius = ref_interp['TABS_mean'] - 273.15  # Convert K to C
    QC_fine = saturation_adjustment(QT_scaled, T_profile_celsius, grid_z)

    print(f"\nQC statistics:")
    print(f"  Mean: {np.mean(QC_fine):.4f} g/kg")
    print(f"  Std: {np.std(QC_fine):.4f} g/kg")
    print(f"  Min: {np.min(QC_fine):.4f} g/kg")
    print(f"  Max: {np.max(QC_fine):.4f} g/kg")
    print(f"  Cloud fraction: {100 * np.sum(QC_fine > 0) / QC_fine.size:.1f}%")

    # Save outputs
    save_to_netcdf(QC_fine, grid_x, grid_y, grid_z, 'QC_fine_512x512x256.nc')
    plot_vertically_integrated(QC_fine, grid_x, grid_y, grid_z, 'QC_fine_integrated.png')
    plot_isosurfaces(QC_fine, grid_x, grid_y, grid_z, [0.01, 0.1], 'QC_fine_isosurfaces.png')

    print("\n" + "=" * 70)
    print("FINE SIMULATION COMPLETE!")
    print("=" * 70)
    print("\nOutput files:")
    print("  - QC_fine_512x512x256.nc")
    print("  - QC_fine_integrated.png")
    print("  - QC_fine_isosurfaces.png")


if __name__ == '__main__':
    main()
