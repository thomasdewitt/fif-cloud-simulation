#!/usr/bin/env python3
"""
FIF Cloud Condensate Simulation

Creates 3D cloud condensate (QC) fields using the Fractionally Integrated Flux (FIF)
multifractal simulation method with the scaleinvariance package.

Domain: 128km x 128km x 15km
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
DOMAIN_X = 4000  # meters
DOMAIN_Y = 4000  # meters
DOMAIN_Z = 1500  # meters

# FIF parameters
H = 0.4
C1 = 0.02
ALPHA = 2
SPHEROSCALE = 30.0  # meters
HZ = 5/9
SCALE_METRIC_DIM = 23/9  # 1 + 1 + Hz
OUTER_SCALE = 50000

# Reference data path
REF_DATA_PATH = 'reference-profiles/SAM_RCEMIP_profiles.nc'

# Output filename
OUTPUT_FILENAME = 'output/QC_FIF_Low_StrCu_warm.nc'


def load_reference_data(max_height=15000):
    """
    Load reference profiles and trim to specified max height.

    Parameters
    ----------
    max_height : float
        Maximum height in meters (default: 15000m)

    Returns
    -------
    dict
        Dictionary containing height, QT_mean, QT_normalized_log_std, and TABS_mean arrays
    """
    ds = xr.open_dataset(REF_DATA_PATH)

    # Extract height coordinate and trim to max_height
    height = ds['height'].values
    mask = height <= max_height

    ref_data = {
        'height': height[mask],
        'QT_mean': ds['QT_mean_profile'].values[mask],
        'QT_normalized_log_std': ds['QT_normalized_log_std'].values[mask],
        'TABS_mean': ds['TABS_mean_profile'].values[mask],
    }

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

    interp_data = {}
    for key in ['QT_mean', 'QT_normalized_log_std', 'TABS_mean']:
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


def apply_linear_temperature_perturbation(T_profile, grid_z, DOMAIN_Z, T_bottom=5.0):
    """
    Apply a linear temperature perturbation: T_bottom K at bottom, 0 K at top.

    Parameters
    ----------
    T_profile : ndarray
        Temperature profile in K, shape (nz,)
    grid_z : ndarray
        Height coordinates in meters, shape (nz,)
    DOMAIN_Z : float
        Domain height in meters
    T_bottom : float
        Temperature perturbation at bottom (default: 5 K)

    Returns
    -------
    ndarray
        Temperature profile with linear perturbation applied
    """
    # Linear interpolation: T_bottom at z=0, 0 at z=DOMAIN_Z
    perturbation = T_bottom * (1 - grid_z / DOMAIN_Z)
    return T_profile + perturbation


def apply_top_temperature_perturbation(T_profile, grid_z, DOMAIN_Z, T_top=5.0):
    """
    Apply a linear temperature perturbation starting at 3/4 domain height.

    Perturbation is 0 K at 3/4*DOMAIN_Z and T_top K at domain top.

    Parameters
    ----------
    T_profile : ndarray
        Temperature profile in K, shape (nz,)
    grid_z : ndarray
        Height coordinates in meters, shape (nz,)
    DOMAIN_Z : float
        Domain height in meters
    T_top : float
        Temperature perturbation at top (default: 5 K)

    Returns
    -------
    ndarray
        Temperature profile with linear perturbation applied
    """
    # Starting height at 3/4 of domain
    z_start = 0.75 * DOMAIN_Z

    # Linear interpolation: 0 at z=z_start, T_top at z=DOMAIN_Z
    # For z < z_start: perturbation = 0
    # For z >= z_start: perturbation = T_top * (z - z_start) / (DOMAIN_Z - z_start)
    perturbation = np.where(
        grid_z >= z_start,
        T_top * (grid_z - z_start) / (DOMAIN_Z - z_start),
        0.0
    )
    return T_profile + perturbation


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

    nx, ny, nz = QT.shape
    QC = np.zeros_like(QT)

    # Calculate saturation mixing ratio for each level
    QS_profile = saturation_mixing_ratio(T_profile_celsius, P_hPa)

    # Apply saturation adjustment at each level
    for k in range(nz):
        QV = np.minimum(QT[:, :, k], QS_profile[k])
        QC[:, :, k] = np.maximum(0.0, QT[:, :, k] - QS_profile[k])

    return QC


def separate_liquid_ice(QC, T_profile_K):
    """
    Separate condensate into liquid water and ice based on temperature.

    Ice fraction increases linearly from 0 at T=273.15 K to 1 at T=233.15 K.
    QL = QC * (1 - ice_fraction)
    QI = QC * ice_fraction

    Parameters
    ----------
    QC : ndarray
        Cloud water mixing ratio in g/kg, shape (nx, ny, nz)
    T_profile_K : ndarray
        Temperature profile in K, shape (nz,)

    Returns
    -------
    tuple
        (QL, QI) where both are ndarrays of shape (nx, ny, nz)
    """

    nx, ny, nz = QC.shape
    QL = np.zeros_like(QC)
    QI = np.zeros_like(QC)

    # Temperature thresholds
    T_liquid = 273.15  # 0°C - all liquid
    T_ice = 233.15     # -40°C - all ice
    T_range = T_liquid - T_ice

    # Calculate ice fraction for each level
    for k in range(nz):
        T = T_profile_K[k]

        # Linear interpolation between T_liquid and T_ice
        if T >= T_liquid:
            ice_frac = 0.0
        elif T <= T_ice:
            ice_frac = 1.0
        else:
            ice_frac = (T_liquid - T) / T_range

        QL[:, :, k] = QC[:, :, k] * (1.0 - ice_frac)
        QI[:, :, k] = QC[:, :, k] * ice_frac

    return QL, QI


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

    # Step 1: Normalize to mean 1
    field = fif_field / np.mean(fif_field, axis=(0,1))

    log_field = np.log(field)

    scaled_log_field = ref_log_std_profile * log_field/np.std(log_field, axis=(0,1))

    scaled_field = np.exp(scaled_log_field)

    scaled_field[np.isnan(scaled_field)] = 1

    return scaled_field * ref_mean_profile[None, None, :]


def generate_fif_simulation(nx, ny, nz, domain_x, domain_y, domain_z,
                            H, C1, alpha, spheroscale, Hz, scale_metric_dim,
                            seed=None, upscale_factor=1):
    """
    Generate 3D FIF simulation with anisotropic scaling.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions (final output size)
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
    upscale_factor : int, optional
        Generate at upscale_factor higher resolution then downsample (default: 1)

    Returns
    -------
    tuple
        (fif_field, grid_x, grid_y, grid_z) where grids are coordinate arrays
    """

    # Adjust for upscaling
    nx_gen = nx * upscale_factor
    ny_gen = ny * upscale_factor
    nz_gen = nz * upscale_factor

    # Grid size for simulation
    size = (nx_gen, ny_gen, nz_gen)

    # periodic=(True, True, False) means only z is doubled internally
    # The scale metric must match the simulation domain size
    # With periodic=(True, True, False), sim_domain = (nx_gen, ny_gen, nz_gen*2)
    sim_size = (nx_gen, ny_gen, nz_gen )

    # Create coordinate grids for full resolution
    grid_x_full = np.linspace(0, domain_x, nx_gen, endpoint=False)
    grid_y_full = np.linspace(0, domain_y, ny_gen, endpoint=False)
    grid_z_full = np.linspace(domain_z/(2*nz_gen), domain_z, nz_gen, endpoint=True)

    # Create anisotropic scale metric for the doubled domain
    # Convert spheroscale from meters to grid units
    dx = domain_x / nx_gen
    dy = domain_y / ny_gen
    dz = domain_z / nz_gen

    # Scale metric uses dx=2 spacing (LS2010), so we need to scale spheroscale accordingly
    # The dx=2 spacing means each grid point in the kernel is 2 units apart
    # Our actual grid spacing is dx meters, so we need to scale
    spheroscale_grid = spheroscale / dx  # in grid units (assuming isotropic x-y)
    print(f"  Running FIF_ND {size} ... ", end='', flush=True)

    scale_metric = canonical_scale_metric(sim_size, ls=spheroscale_grid, Hz=Hz)

    # Generate FIF with periodic in x,y and non-periodic in z
    fif_field_full = FIF_ND(
        size=size,
        alpha=alpha,
        C1=C1,
        H=H,
        periodic=(True, True, True),  # periodic in x,y; non-periodic in z
        scale_metric=scale_metric,
        scale_metric_dim=scale_metric_dim,
        kernel_construction_method='LS2010',
        outer_scale=OUTER_SCALE/dx
    )
    print(f"done")

    # Downsample if upscale_factor > 1
    if upscale_factor > 1:
        fif_field = fif_field_full[::upscale_factor, ::upscale_factor, ::upscale_factor]
        grid_x = grid_x_full[::upscale_factor]
        grid_y = grid_y_full[::upscale_factor]
        grid_z = grid_z_full[::upscale_factor]
    else:
        fif_field = fif_field_full
        grid_x = grid_x_full
        grid_y = grid_y_full
        grid_z = grid_z_full

    return fif_field, grid_x, grid_y, grid_z


def save_to_netcdf(QL, QI, grid_x, grid_y, grid_z, filename):
    """
    Save liquid water and ice fields to NetCDF file.

    Parameters
    ----------
    QL : ndarray
        Liquid water mixing ratio in g/kg, shape (nx, ny, nz)
    QI : ndarray
        Ice water mixing ratio in g/kg, shape (nx, ny, nz)
    grid_x, grid_y, grid_z : ndarray
        Coordinate arrays in meters
    filename : str
        Output filename
    """

    ds = xr.Dataset(
        {
            'QL': (['x', 'y', 'z'], QL, {
                'long_name': 'Liquid water mixing ratio',
                'units': 'g/kg',
                'description': 'Generated using FIF multifractal simulation'
            }),
            'QI': (['x', 'y', 'z'], QI, {
                'long_name': 'Ice water mixing ratio',
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



def plot_profiles(QC, QT, T, grid_z, filename, QL=None, QI=None):
    """
    Create profile plots of QC, QT, T, QL, and QI with mean and std shading.

    Parameters
    ----------
    QC : ndarray
        Cloud water mixing ratio in g/kg, shape (nx, ny, nz)
    QT : ndarray
        Total water mixing ratio in g/kg, shape (nx, ny, nz)
    T : ndarray
        Temperature in K, shape (nz,) - mean profile
    grid_z : ndarray
        Height coordinate in meters
    filename : str
        Output filename for plot
    QL : ndarray, optional
        Liquid water mixing ratio in g/kg, shape (nx, ny, nz)
    QI : ndarray, optional
        Ice water mixing ratio in g/kg, shape (nx, ny, nz)
    """

    # Calculate mean and std over horizontal dimensions (x, y)
    QC_mean = np.mean(QC, axis=(0, 1))
    QC_std = np.std(QC, axis=(0, 1))

    QT_mean = np.mean(QT, axis=(0, 1))
    QT_std = np.std(QT, axis=(0, 1))

    # Convert height to km
    grid_z_km = grid_z / 1000.0

    # Determine number of subplots
    n_plots = 3
    if QL is not None and QI is not None:
        n_plots = 5
        QL_mean = np.mean(QL, axis=(0, 1))
        QL_std = np.std(QL, axis=(0, 1))
        QI_mean = np.mean(QI, axis=(0, 1))
        QI_std = np.std(QI, axis=(0, 1))

    fig, axes = plt.subplots(1, n_plots, figsize=(4*n_plots, 6))

    # QC profile
    ax = axes[0]
    ax.plot(QC_mean, grid_z_km, 'b-', linewidth=2, label='Mean')
    ax.fill_betweenx(grid_z_km, QC_mean - QC_std, QC_mean + QC_std,
                     alpha=0.3, color='blue', label='±1 Std')
    ax.set_xlabel('QC (g/kg)')
    ax.set_ylabel('Height (km)')
    ax.set_title('Cloud Water Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # QT profile
    ax = axes[1]
    ax.plot(QT_mean, grid_z_km, 'r-', linewidth=2, label='Mean')
    ax.fill_betweenx(grid_z_km, QT_mean - QT_std, QT_mean + QT_std,
                     alpha=0.3, color='red', label='±1 Std')
    ax.set_xlabel('QT (g/kg)')
    ax.set_ylabel('Height (km)')
    ax.set_title('Total Water Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Temperature profile
    ax = axes[2]
    ax.plot(T - 273.15, grid_z_km, 'g-', linewidth=2, label='Mean')
    ax.set_xlabel('T (°C)')
    ax.set_ylabel('Height (km)')
    ax.set_title('Temperature Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Liquid water profile (if provided)
    if QL is not None and n_plots > 3:
        ax = axes[3]
        ax.plot(QL_mean, grid_z_km, 'c-', linewidth=2, label='Mean')
        ax.fill_betweenx(grid_z_km, QL_mean - QL_std, QL_mean + QL_std,
                         alpha=0.3, color='cyan', label='±1 Std')
        ax.set_xlabel('QL (g/kg)')
        ax.set_ylabel('Height (km)')
        ax.set_title('Liquid Water Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Ice water profile (if provided)
    if QI is not None and n_plots > 3:
        ax = axes[4]
        ax.plot(QI_mean, grid_z_km, 'm-', linewidth=2, label='Mean')
        ax.fill_betweenx(grid_z_km, QI_mean - QI_std, QI_mean + QI_std,
                         alpha=0.3, color='magenta', label='±1 Std')
        ax.set_xlabel('QI (g/kg)')
        ax.set_ylabel('Height (km)')
        ax.set_title('Ice Water Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle('Vertical Profiles (horizontally averaged)', fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_isosurfaces(QC, grid_x, grid_y, grid_z, isovalues, filename):
    """
    Create side-by-side 3D isosurface plots of QC.

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

    try:
        from skimage import measure
    except ImportError:
        print("  Warning: scikit-image not installed, skipping isosurface plot")
        return

    fig = plt.figure(figsize=(16, 7))
    colors = ['cyan', 'blue']

    for idx, isovalue in enumerate(isovalues):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')

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
                           color=colors[idx % len(colors)], alpha=0.7,
                           linewidth=0, antialiased=True)
        except Exception as e:
            print(f"    Error computing isosurface for {isovalue} g/kg: {e}")

        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        ax.set_zlabel('Z (km)')
        ax.set_title(f'QC = {isovalue} g/kg')
        ax.set_xlim(0, DOMAIN_X/1000)
        ax.set_ylim(0, DOMAIN_Y/1000)
        ax.set_zlim(0, DOMAIN_Z/1000)
        ax.set_aspect('equal')

    fig.suptitle('Cloud Water Isosurfaces', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()



def main():
    """Main simulation workflow."""


    # Load reference data
    ref_data = load_reference_data(max_height=DOMAIN_Z)


    nx, ny, nz = 512,512,256
    upscale_factor = 2

    reference_std_scale_factor = 0.6
    bottom_temp_boost = 0
    top_temp_boost = 7
    uniform_temp_boost = -0.

    # Generate FIF simulation (with optional upscaling and downsampling)
    fif_QT, grid_x, grid_y, grid_z = generate_fif_simulation(
        nx, ny, nz,
        DOMAIN_X, DOMAIN_Y, DOMAIN_Z,
        H, C1, ALPHA,
        SPHEROSCALE, HZ, SCALE_METRIC_DIM,
        upscale_factor=upscale_factor
    )

    # Interpolate reference profiles to simulation grid
    ref_interp = interpolate_to_grid(ref_data, grid_z)
    ref_interp['QT_normalized_log_std'] *= reference_std_scale_factor
    ref_interp['TABS_mean'] += uniform_temp_boost

    # Normalize and scale to match reference QT statistics
    QT_scaled = normalize_and_scale_fif(
        fif_QT,
        ref_interp['QT_mean'],
        ref_interp['QT_normalized_log_std']
    )


    # Apply linear temperature perturbation (bottom: T_bottom at z=0, 0 at top)
    T_profile_K = apply_linear_temperature_perturbation(
        ref_interp['TABS_mean'], grid_z, DOMAIN_Z, T_bottom=bottom_temp_boost
    )

    # Apply top temperature perturbation (top: 0 at 3/4 height, T_top at top)
    T_profile_K = apply_top_temperature_perturbation(
        T_profile_K, grid_z, DOMAIN_Z, T_top=top_temp_boost
    )

    # Apply saturation adjustment to get QC
    T_profile_celsius = T_profile_K - 273.15  # Convert K to C
    QC = saturation_adjustment(QT_scaled, T_profile_celsius, grid_z)

    # Separate into liquid and ice
    QL, QI = separate_liquid_ice(QC, T_profile_K)

    print(f"  Cloud fraction (0.01 g/kg): {100 * np.sum(np.any(QC > 0.01,axis=2)) / QC[:,:,0].size:.1f}%")
    print(f"  Cloud fraction (0.1 g/kg): {100 * np.sum(np.any(QC > 0.1,axis=2)) / QC[:,:,0].size:.1f}%")
    print(f"  Cloud fraction (1 g/kg): {100 * np.sum(np.any(QC > 1,axis=2)) / QC[:,:,0].size:.1f}%")

    # Save outputs
    save_to_netcdf(QL, QI, grid_x, grid_y, grid_z, OUTPUT_FILENAME)
    plot_vertically_integrated(QC, grid_x, grid_y, grid_z, 'plots/QC_integrated.png')
    plot_profiles(QC, QT_scaled, T_profile_K, grid_z, 'plots/profiles.png', QL=QL, QI=QI)
    # plot_isosurfaces(QC, grid_x, grid_y, grid_z, [0.01, 0.1], 'plots/QC_isosurfaces.png')



if __name__ == '__main__':
    main()
