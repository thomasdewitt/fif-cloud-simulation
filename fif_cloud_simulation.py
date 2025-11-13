#!/usr/bin/env python3
"""
FIF Cloud Condensate Simulation

Creates 3D cloud condensate (QC) fields using the Fractionally Integrated Flux (FIF)
multifractal simulation method with the scaleinvariance package.

Domain: 6km x 6km x 2km
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

# Domain parameters
DOMAIN_X = 6000  # meters
DOMAIN_Y = 6000  # meters
DOMAIN_Z = 2000  # meters

# FIF parameters
H = 0.4
C1 = 0.02
ALPHA = 1.8
SPHEROSCALE = 10.0  # meters
HZ = 5/9
SCALE_METRIC_DIM = 23/9  # 1 + 1 + Hz

# Reference data path
REF_DATA_PATH = 'reference-profiles/SAM_COMBLE_profiles.nc'


def load_reference_data(max_height=2000):
    """
    Load COMBLE reference profiles and trim to specified max height.

    Parameters
    ----------
    max_height : float
        Maximum height in meters (default: 2000m)

    Returns
    -------
    dict
        Dictionary containing height, QT_mean, QT_normalized_log_std, and TABS_mean arrays
    """
    print(f"Loading reference data from {REF_DATA_PATH}...")
    ds = xr.open_dataset(REF_DATA_PATH)

    # Extract height coordinate and trim to max_height
    height = ds['height'].values
    mask = height <= max_height

    ref_data = {
        'height': height[mask],
        'QT_mean': ds['QT_mean_profile'].values[mask],
        'QT_normalized_log_std': ds['QT_normalized_log_std'].values[mask],
        'QN_mean': ds['QN_mean_profile'].values[mask],
        'TABS_mean': ds['TABS_mean_profile'].values[mask],
    }

    print(f"  Height range: {ref_data['height'][0]:.1f} - {ref_data['height'][-1]:.1f} m")
    print(f"  Number of levels: {len(ref_data['height'])}")

    return ref_data


def interpolate_to_grid(ref_data, grid_z):
    """
    Interpolate reference profiles to match FIF simulation grid.

    Parameters
    ----------
    ref_data : dict
        Reference data from load_reference_data()
    grid_z : ndarray
        Target z-coordinates (in meters) for interpolation

    Returns
    -------
    dict
        Interpolated profiles on the target grid
    """
    print("Interpolating reference profiles to simulation grid...")

    interp_data = {}
    for key in ['QT_mean', 'QT_normalized_log_std', 'QN_mean', 'TABS_mean']:
        interp_data[key] = np.interp(grid_z, ref_data['height'], ref_data[key])

    return interp_data


def saturation_mixing_ratio(T_celsius, P_hPa=1000.0):
    """
    Calculate saturation mixing ratio using Bolton (1980) formula.

    Parameters
    ----------
    T_celsius : float or ndarray
        Temperature in degrees Celsius
    P_hPa : float
        Pressure in hPa (default: 1000 hPa)

    Returns
    -------
    float or ndarray
        Saturation mixing ratio in g/kg
    """
    # Bolton 1980 formula for saturation vapor pressure (hPa)
    es = 6.112 * np.exp(17.67 * T_celsius / (T_celsius + 243.5))

    # Saturation mixing ratio (dimensionless)
    ws = 0.622 * es / (P_hPa - es)

    # Convert to g/kg
    return ws * 1000.0


def saturation_adjustment(QT, T_profile_celsius, grid_z, P_hPa=1000.0):
    """
    Apply saturation adjustment to calculate QC from QT.

    QC = max(0, QT - QS) where QS is saturation mixing ratio.

    Parameters
    ----------
    QT : ndarray
        Total water mixing ratio (QV + QC) in g/kg, shape (nx, ny, nz)
    T_profile_celsius : ndarray
        Temperature profile in degrees Celsius, shape (nz,)
    grid_z : ndarray
        Height coordinates in meters, shape (nz,)
    P_hPa : float
        Pressure in hPa (default: 1000 hPa)

    Returns
    -------
    ndarray
        Cloud water mixing ratio (QC) in g/kg, same shape as QT
    """
    print("Applying saturation adjustment...")

    nx, ny, nz = QT.shape
    QC = np.zeros_like(QT)

    # Calculate saturation mixing ratio for each level
    QS_profile = saturation_mixing_ratio(T_profile_celsius, P_hPa)

    # Apply saturation adjustment at each level
    for k in range(nz):
        QV = np.minimum(QT[:, :, k], QS_profile[k])
        QC[:, :, k] = np.maximum(0.0, QT[:, :, k] - QS_profile[k])

    return QC


def normalize_and_scale_fif(fif_field, ref_mean_profile, ref_log_std_profile):
    """
    Normalize and scale FIF simulation to match reference statistics.

    Process:
    1. Normalize entire field to mean 1
    2. Take natural log
    3. Scale log-std to match reference normalized-log-std
    4. Exponentiate back
    5. Multiply by reference mean profile

    Parameters
    ----------
    fif_field : ndarray
        Raw FIF simulation, shape (nx, ny, nz)
    ref_mean_profile : ndarray
        Reference mean profile, shape (nz,)
    ref_log_std_profile : ndarray
        Reference normalized log std profile, shape (nz,)

    Returns
    -------
    ndarray
        Scaled field matching reference statistics
    """
    print("Normalizing and scaling FIF simulation...")

    # Step 1: Normalize to mean 1
    field_normalized = fif_field / np.mean(fif_field)

    # Step 2: Take natural log
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    field_log = np.log(field_normalized + epsilon)

    # Step 3: Scale log-std to match reference
    # Do this level by level
    nx, ny, nz = fif_field.shape
    field_scaled_log = np.zeros_like(field_log)

    for k in range(nz):
        level_log = field_log[:, :, k]
        level_mean = np.mean(level_log)
        level_std = np.std(level_log)

        # Scale to have std = ref_log_std_profile[k]
        if level_std > 0:
            field_scaled_log[:, :, k] = (level_log - level_mean) / level_std * ref_log_std_profile[k] + level_mean
        else:
            field_scaled_log[:, :, k] = level_log

    # Step 4: Exponentiate
    field_scaled = np.exp(field_scaled_log)

    # Renormalize to mean 1 before applying reference mean
    field_scaled = field_scaled / np.mean(field_scaled)

    # Step 5: Multiply by reference mean profile
    for k in range(nz):
        field_scaled[:, :, k] *= ref_mean_profile[k]

    return field_scaled


def generate_fif_simulation(nx, ny, nz, domain_x, domain_y, domain_z,
                            H, C1, alpha, spheroscale, Hz, scale_metric_dim,
                            seed=None):
    """
    Generate 3D FIF simulation with anisotropic scaling.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions
    domain_x, domain_y, domain_z : float
        Domain size in meters
    H, C1, alpha : float
        FIF parameters
    spheroscale : float
        Horizontal characteristic scale in meters
    Hz : float
        Vertical anisotropy exponent
    scale_metric_dim : float
        Scale metric dimension for GSI
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    tuple
        (fif_field, grid_x, grid_y, grid_z) where grids are coordinate arrays
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
    # The scale metric must match the simulation domain size
    # With periodic=(True, True, False), sim_domain = (nx, ny, nz*2)
    sim_size = (nx, ny, nz * 2)

    print(f"  Internal simulation size (for non-periodic z): {sim_size}")

    # Create coordinate grids (for output)
    grid_x = np.linspace(0, domain_x, nx, endpoint=False)
    grid_y = np.linspace(0, domain_y, ny, endpoint=False)
    grid_z = np.linspace(domain_z/(2*nz), domain_z, nz, endpoint=True)

    # Create anisotropic scale metric for the doubled domain
    # Convert spheroscale from meters to grid units
    dx = domain_x / nx
    dy = domain_y / ny
    dz = domain_z / nz

    # Scale metric uses dx=2 spacing (LS2010), so we need to scale spheroscale accordingly
    # The dx=2 spacing means each grid point in the kernel is 2 units apart
    # Our actual grid spacing is dx meters, so we need to scale
    spheroscale_grid = spheroscale / dx  # in grid units (assuming isotropic x-y)

    print(f"  Creating anisotropic scale metric...")
    scale_metric = canonical_scale_metric(sim_size, ls=spheroscale_grid, Hz=Hz)

    # Generate FIF with periodic in x,y and non-periodic in z
    print(f"  Running FIF_ND (this may take a while)...")
    fif_field = FIF_ND(
        size=size,
        alpha=alpha,
        C1=C1,
        H=H,
        periodic=(True, True, False),  # periodic in x,y; non-periodic in z
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


def save_to_netcdf(QC, grid_x, grid_y, grid_z, filename):
    """
    Save QC field to NetCDF file.

    Parameters
    ----------
    QC : ndarray
        Cloud water mixing ratio in g/kg, shape (nx, ny, nz)
    grid_x, grid_y, grid_z : ndarray
        Coordinate arrays in meters
    filename : str
        Output filename
    """
    print(f"\nSaving to NetCDF: {filename}")

    ds = xr.Dataset(
        {
            'QC': (['x', 'y', 'z'], QC, {
                'long_name': 'Cloud water mixing ratio',
                'units': 'g/kg',
                'description': 'Generated using FIF multifractal simulation'
            })
        },
        coords={
            'x': (['x'], grid_x, {'long_name': 'X coordinate', 'units': 'm'}),
            'y': (['y'], grid_y, {'long_name': 'Y coordinate', 'units': 'm'}),
            'z': (['z'], grid_z, {'long_name': 'Height', 'units': 'm'}),
        },
        attrs={
            'title': 'FIF Cloud Condensate Simulation',
            'H': H,
            'C1': C1,
            'alpha': ALPHA,
            'spheroscale': SPHEROSCALE,
            'Hz': HZ,
            'scale_metric_dim': SCALE_METRIC_DIM,
        }
    )

    ds.to_netcdf(filename)
    print(f"  Saved successfully!")


def plot_vertically_integrated(QC, grid_x, grid_y, grid_z, filename):
    """
    Create vertically integrated plot of QC (liquid water path).

    Parameters
    ----------
    QC : ndarray
        Cloud water mixing ratio in g/kg, shape (nx, ny, nz)
    grid_x, grid_y, grid_z : ndarray
        Coordinate arrays in meters
    filename : str
        Output filename for plot
    """
    print(f"\nCreating vertically integrated plot: {filename}")

    # Calculate dz for integration
    dz = np.diff(grid_z)
    dz = np.append(dz, dz[-1])  # extend to match nz

    # Integrate: LWP = integral(rho * QC * dz)
    # Assuming rho ~ 1 kg/m^3, QC in g/kg = 0.001 * QC in kg/kg
    # LWP will be in g/m^2
    LWP = np.sum(QC * dz[np.newaxis, np.newaxis, :] * 0.001, axis=2)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.contourf(grid_x/1000, grid_y/1000, LWP.T, levels=20, cmap='viridis')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('Vertically Integrated Cloud Water (Liquid Water Path)')
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('LWP (g/m²)')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"  LWP statistics:")
    print(f"    Mean: {np.mean(LWP):.2f} g/m²")
    print(f"    Max: {np.max(LWP):.2f} g/m²")


def plot_isosurfaces(QC, grid_x, grid_y, grid_z, isovalues, filename):
    """
    Create 3D isosurface plot of QC.

    Parameters
    ----------
    QC : ndarray
        Cloud water mixing ratio in g/kg, shape (nx, ny, nz)
    grid_x, grid_y, grid_z : ndarray
        Coordinate arrays in meters
    isovalues : list of float
        Isosurface values to plot in g/kg
    filename : str
        Output filename for plot
    """
    print(f"\nCreating 3D isosurface plot: {filename}")
    print(f"  Isovalues: {isovalues} g/kg")

    try:
        from skimage import measure
    except ImportError:
        print("  Warning: scikit-image not installed, skipping isosurface plot")
        return

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    colors = ['cyan', 'blue']
    alphas = [0.3, 0.6]

    for i, isovalue in enumerate(isovalues):
        print(f"  Computing isosurface at {isovalue} g/kg...")

        # Check if isovalue exists in data
        if np.max(QC) < isovalue:
            print(f"    Warning: max QC ({np.max(QC):.4f}) < {isovalue}, skipping")
            continue

        try:
            verts, faces, normals, values = measure.marching_cubes(
                QC.T,  # transpose to (z, y, x) for correct orientation
                level=isovalue,
                spacing=(grid_z[1]-grid_z[0], grid_y[1]-grid_y[0], grid_x[1]-grid_x[0])
            )

            # Adjust coordinates
            verts[:, 0] += grid_z[0]
            verts[:, 1] += grid_y[0]
            verts[:, 2] += grid_x[0]

            # Convert to km for plotting
            verts = verts / 1000.0

            # Plot surface
            ax.plot_trisurf(verts[:, 2], verts[:, 1], faces, verts[:, 0],
                           color=colors[i % len(colors)], alpha=alphas[i % len(alphas)],
                           linewidth=0, antialiased=True,
                           label=f'QC = {isovalue} g/kg')
        except Exception as e:
            print(f"    Error computing isosurface: {e}")

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Cloud Water Isosurfaces')
    ax.set_xlim(0, DOMAIN_X/1000)
    ax.set_ylim(0, DOMAIN_Y/1000)
    ax.set_zlim(0, DOMAIN_Z/1000)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"  Saved isosurface plot!")


def main():
    """Main simulation workflow."""

    print("=" * 70)
    print("FIF CLOUD CONDENSATE SIMULATION")
    print("=" * 70)

    # Load reference data
    ref_data = load_reference_data(max_height=DOMAIN_Z)

    # COARSE RESOLUTION SIMULATION
    print("\n" + "=" * 70)
    print("COARSE RESOLUTION SIMULATION (64x64x32)")
    print("=" * 70)

    nx_coarse, ny_coarse, nz_coarse = 64, 64, 32

    # Generate FIF simulation
    fif_QT, grid_x, grid_y, grid_z = generate_fif_simulation(
        nx_coarse, ny_coarse, nz_coarse,
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
    QC_coarse = saturation_adjustment(QT_scaled, T_profile_celsius, grid_z)

    print(f"\nQC statistics:")
    print(f"  Mean: {np.mean(QC_coarse):.4f} g/kg")
    print(f"  Std: {np.std(QC_coarse):.4f} g/kg")
    print(f"  Min: {np.min(QC_coarse):.4f} g/kg")
    print(f"  Max: {np.max(QC_coarse):.4f} g/kg")
    print(f"  Cloud fraction: {100 * np.sum(QC_coarse > 0) / QC_coarse.size:.1f}%")

    # Save outputs
    save_to_netcdf(QC_coarse, grid_x, grid_y, grid_z, 'QC_coarse_64x64x32.nc')
    plot_vertically_integrated(QC_coarse, grid_x, grid_y, grid_z, 'QC_coarse_integrated.png')
    plot_isosurfaces(QC_coarse, grid_x, grid_y, grid_z, [0.01, 0.1], 'QC_coarse_isosurfaces.png')

    print("\n" + "=" * 70)
    print("COARSE SIMULATION COMPLETE!")
    print("=" * 70)
    print("\nReview the coarse resolution outputs before proceeding to fine resolution.")
    print("Output files:")
    print("  - QC_coarse_64x64x32.nc")
    print("  - QC_coarse_integrated.png")
    print("  - QC_coarse_isosurfaces.png")


if __name__ == '__main__':
    main()
