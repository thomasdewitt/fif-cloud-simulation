#!/usr/bin/env python
"""
Coarsen the tiled NetCDF data by averaging 3x3x1 blocks (horizontal only).
Results in same physical dimensions but 1/3 the resolution.
"""
import numpy as np
import netCDF4 as nc

# Input and output files
input_file = 'QC_FIF_Low_StrCu_warm_highly_intermittent_tiled.nc'
output_file = 'QC_FIF_Low_StrCu_warm_highly_intermittent_tiled_coarse.nc'

print(f"Reading {input_file}...")
ds_in = nc.Dataset(input_file, 'r')

# Get dimensions
nx, ny, nz = len(ds_in.dimensions['x']), len(ds_in.dimensions['y']), len(ds_in.dimensions['z'])
print(f"Original dimensions: x={nx}, y={ny}, z={nz}")

# Read data
QL = ds_in.variables['QL'][:]
QI = ds_in.variables['QI'][:]
x_coord = ds_in.variables['x'][:]
y_coord = ds_in.variables['y'][:]
z_coord = ds_in.variables['z'][:]

# Get attributes (separate _FillValue from other attributes)
QL_attrs = {}
QL_fill_value = None
for attr in ds_in.variables['QL'].ncattrs():
    if attr == '_FillValue':
        QL_fill_value = ds_in.variables['QL'].getncattr(attr)
    else:
        QL_attrs[attr] = ds_in.variables['QL'].getncattr(attr)

QI_attrs = {}
QI_fill_value = None
for attr in ds_in.variables['QI'].ncattrs():
    if attr == '_FillValue':
        QI_fill_value = ds_in.variables['QI'].getncattr(attr)
    else:
        QI_attrs[attr] = ds_in.variables['QI'].getncattr(attr)

# Get global attributes
global_attrs = {attr: ds_in.getncattr(attr) for attr in ds_in.ncattrs()}

ds_in.close()

# Coarsen by averaging 3x3x1 blocks
nx_coarse = nx // 3
ny_coarse = ny // 3

print(f"Coarsening to: x={nx_coarse}, y={ny_coarse}, z={nz}")
print("Averaging 3x3x1 blocks...")

QL_coarse = np.zeros((nx_coarse, ny_coarse, nz), dtype=QL.dtype)
QI_coarse = np.zeros((nx_coarse, ny_coarse, nz), dtype=QI.dtype)

for i in range(nx_coarse):
    if i % 50 == 0:
        print(f"  Processing row {i}/{nx_coarse}...")
    for j in range(ny_coarse):
        # Average over 3x3 horizontal blocks for each vertical level
        QL_coarse[i, j, :] = QL[i*3:(i+1)*3, j*3:(j+1)*3, :].mean(axis=(0, 1))
        QI_coarse[i, j, :] = QI[i*3:(i+1)*3, j*3:(j+1)*3, :].mean(axis=(0, 1))

# Create coarsened coordinate arrays (sample every 3rd point, adjusted for cell centers)
# The physical domain should be the same, so we sample coordinates appropriately
x_coord_coarse = x_coord[1::3]  # Take every 3rd point starting from index 1 (center of first block)
y_coord_coarse = y_coord[1::3]

print(f"\nWriting {output_file}...")
ds_out = nc.Dataset(output_file, 'w', format='NETCDF4')

# Create dimensions
ds_out.createDimension('x', nx_coarse)
ds_out.createDimension('y', ny_coarse)
ds_out.createDimension('z', nz)

# Create coordinate variables
x_var = ds_out.createVariable('x', 'f4', ('x',))
y_var = ds_out.createVariable('y', 'f4', ('y',))
z_var = ds_out.createVariable('z', 'f4', ('z',))

x_var[:] = x_coord_coarse
y_var[:] = y_coord_coarse
z_var[:] = z_coord

# Create data variables
QL_var = ds_out.createVariable('QL', QL_coarse.dtype, ('x', 'y', 'z'),
                                zlib=True, complevel=4, fill_value=QL_fill_value)
QI_var = ds_out.createVariable('QI', QI_coarse.dtype, ('x', 'y', 'z'),
                                zlib=True, complevel=4, fill_value=QI_fill_value)

# Copy attributes
for attr, value in QL_attrs.items():
    QL_var.setncattr(attr, value)
for attr, value in QI_attrs.items():
    QI_var.setncattr(attr, value)

# Write data
print("Writing QL...")
QL_var[:] = QL_coarse
print("Writing QI...")
QI_var[:] = QI_coarse

# Copy and update global attributes
for attr, value in global_attrs.items():
    ds_out.setncattr(attr, value)

ds_out.setncattr('coarsening_method', 'Averaged 3x3x1 horizontal blocks from tiled data')
ds_out.setncattr('coarsening_factor', '3x in horizontal dimensions')
ds_out.setncattr('coarsened_from', input_file)

ds_out.close()
print(f"\nDone! Created {output_file}")
print(f"Coarsened domain size: {nx_coarse} x {ny_coarse} x {nz}")
print(f"Same physical dimensions as original, 1/3 the resolution")
