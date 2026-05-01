#!/usr/bin/env python
"""
Tile NetCDF data to create a 3x larger domain by repeating.
Original data is duplicated in a 3x3 grid (9 identical copies).
"""
import numpy as np
import netCDF4 as nc

# Input and output files
input_file = 'QC_FIF_Low_StrCu_warm_highly_intermittent.nc'
output_file = 'QC_FIF_Low_StrCu_warm_highly_intermittent_tiled.nc'

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

# Get attributes if they exist (separate _FillValue from other attributes)
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

ds_in.close()

# Create new arrays (3x larger in x and y)
nx_new, ny_new = 3 * nx, 3 * ny

print(f"New dimensions: x={nx_new}, y={ny_new}, z={nz}")
print("Creating repeated tiles (3x3 duplicates)...")

# Simply tile/repeat the data using numpy.tile
# tile() repeats the array: (3, 3, 1) means 3 times in x, 3 times in y, 1 time in z
QL_new = np.tile(QL, (3, 3, 1))
QI_new = np.tile(QI, (3, 3, 1))

# Create new coordinate arrays
dx = x_coord[1] - x_coord[0] if len(x_coord) > 1 else 1.0
dy = y_coord[1] - y_coord[0] if len(y_coord) > 1 else 1.0

x_coord_new = np.arange(nx_new) * dx + x_coord[0] - nx * dx
y_coord_new = np.arange(ny_new) * dy + y_coord[0] - ny * dy

# Write output file
print(f"\nWriting {output_file}...")
ds_out = nc.Dataset(output_file, 'w', format='NETCDF4')

# Create dimensions
ds_out.createDimension('x', nx_new)
ds_out.createDimension('y', ny_new)
ds_out.createDimension('z', nz)

# Create coordinate variables
x_var = ds_out.createVariable('x', 'f4', ('x',))
y_var = ds_out.createVariable('y', 'f4', ('y',))
z_var = ds_out.createVariable('z', 'f4', ('z',))

x_var[:] = x_coord_new
y_var[:] = y_coord_new
z_var[:] = z_coord

# Create data variables (with fill_value if it exists)
QL_var = ds_out.createVariable('QL', QL.dtype, ('x', 'y', 'z'),
                                zlib=True, complevel=4, fill_value=QL_fill_value)
QI_var = ds_out.createVariable('QI', QI.dtype, ('x', 'y', 'z'),
                                zlib=True, complevel=4, fill_value=QI_fill_value)

# Copy attributes (excluding _FillValue which was already set)
for attr, value in QL_attrs.items():
    QL_var.setncattr(attr, value)
for attr, value in QI_attrs.items():
    QI_var.setncattr(attr, value)

# Write data
print("Writing QL...")
QL_var[:] = QL_new
print("Writing QI...")
QI_var[:] = QI_new

# Add global attributes
ds_out.setncattr('description', 'Tiled version of original data (3x3 repeated tiles)')
ds_out.setncattr('original_file', input_file)
ds_out.setncattr('tiling_method', 'repeated: original data duplicated in a 3x3 grid')

ds_out.close()
print(f"\nDone! Created {output_file}")
print(f"New domain size: {nx_new} x {ny_new} x {nz}")
