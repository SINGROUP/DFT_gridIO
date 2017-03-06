import numpy as np
cimport numpy as np

import conversion_factors as c_factor

DTYPE_FLOAT = np.float64
ctypedef np.float64_t DTYPE_FLOAT_t
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t

cdef double au_to_eV = c_factor.au_to_eV

def read_cube_nonorthogonal(filename, return_unstructured_data=False):
    cdef int ia, n_atoms, ix, iy, iz, idata
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] origin
    cdef np.ndarray[DTYPE_INT_t, ndim=1] n_voxels = np.zeros(3, dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] voxel_vectors = np.zeros((3, 3), dtype=DTYPE_FLOAT)
    cdef np.ndarray[DTYPE_INT_t, ndim=1] atom_types
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] atom_pos
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] volume_data
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=3] grid_data
    
    in_file = open(filename, 'r')
    
    # read comment lines
    title = in_file.readline()
    title += in_file.readline()
    
    # read metadata
    splitline = in_file.readline().split()
    n_atoms = int(splitline[0])
    origin = np.array([float(splitline[1]), float(splitline[2]), float(splitline[3])], dtype=DTYPE_FLOAT)
    
    for i in range(3):
        splitline = in_file.readline().split()
        n_voxels[i] = int(splitline[0])
        if n_voxels[i] > 0: #length unit is bohr
            conversion_factor = c_factor.bohr_to_m*1.0e10
        else:               #length unit is Ångström
            conversion_factor = 1
            n_voxels[i] = -n_voxels[i]
        voxel_vectors[i, 0] = float(splitline[1])*conversion_factor
        voxel_vectors[i, 1] = float(splitline[2])*conversion_factor
        voxel_vectors[i, 2] = float(splitline[3])*conversion_factor
    
    origin = origin*conversion_factor
    
    # read atom types and positions
    atom_types = np.zeros(n_atoms, dtype=DTYPE_INT)
    atom_pos = np.zeros((n_atoms, 3), dtype=DTYPE_FLOAT)
    for ia in range(n_atoms):
        splitline = in_file.readline().split()
        atom_types[ia] = int(splitline[0])
        atom_pos[ia, 0] = float(splitline[2])*conversion_factor
        atom_pos[ia, 1] = float(splitline[3])*conversion_factor
        atom_pos[ia, 2] = float(splitline[4])*conversion_factor
    
    # read the volumetric data (works for non-orthogonal unit cell)
    if return_unstructured_data:
        volume_data = np.zeros((n_voxels[0]*n_voxels[1]*n_voxels[2], 4), dtype=DTYPE_FLOAT)
        idata = 0
        for ix in range(n_voxels[0]):
            for iy in range(n_voxels[1]):
                iz = 0
                while iz < n_voxels[2]:
                    splitline = in_file.readline().split()
                    for data_entry in splitline:
                        volume_data[idata, 0] = origin[0] + ix*voxel_vectors[0, 0] + iy*voxel_vectors[1, 0] + iz*voxel_vectors[2, 0]
                        volume_data[idata, 1] = origin[1] + ix*voxel_vectors[0, 1] + iy*voxel_vectors[1, 1] + iz*voxel_vectors[2, 1]
                        volume_data[idata, 2] = origin[2] + ix*voxel_vectors[0, 2] + iy*voxel_vectors[1, 2] + iz*voxel_vectors[2, 2]
                        volume_data[idata, 3] = float(data_entry)*au_to_eV
                        idata = idata + 1
                        iz = iz + 1
        
        return atom_types, atom_pos, n_voxels, volume_data
    
    else:
        grid_data = np.zeros((n_voxels[0], n_voxels[1], n_voxels[2]), dtype=DTYPE_FLOAT)
        for ix in range(n_voxels[0]):
            for iy in range(n_voxels[1]):
                iz = 0
                while iz < n_voxels[2]:
                    splitline = in_file.readline().split()
                    for data_entry in splitline:
                        grid_data[ix, iy, iz] = float(data_entry)*au_to_eV
                        iz = iz + 1
        
        return atom_types, atom_pos, n_voxels, voxel_vectors, grid_data


def read_cube_orthogonal(filename):
    cdef int ia, n_atoms, ix, iy, iz, idata
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] origin
    cdef np.ndarray[DTYPE_INT_t, ndim=1] n_voxels = np.zeros(3, dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] voxel_diffs = np.zeros(3, dtype=DTYPE_FLOAT)
    cdef np.ndarray[DTYPE_INT_t, ndim=1] atom_types
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] atom_pos
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] xs, ys, zs
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=3] grid_data
    
    in_file = open(filename, 'r')
    
    # read comment lines
    title = in_file.readline()
    title += in_file.readline()
    
    # read metadata
    splitline = in_file.readline().split()
    n_atoms = int(splitline[0])
    origin = np.array([float(splitline[1]), float(splitline[2]), float(splitline[3])], dtype=DTYPE_FLOAT)
    
    for i in range(3):
        splitline = in_file.readline().split()
        n_voxels[i] = int(splitline[0])
        if n_voxels[i] > 0: #length unit is bohr
            conversion_factor = c_factor.bohr_to_m*1.0e10
        else:               #length unit is Ångström
            conversion_factor = 1
            n_voxels[i] = -n_voxels[i]
        voxel_diffs[i] = float(splitline[i+1])*conversion_factor
    
    origin = origin*conversion_factor
    
    # read atom types and positions
    atom_types = np.zeros(n_atoms, dtype=DTYPE_INT)
    atom_pos = np.zeros((n_atoms, 3), dtype=DTYPE_FLOAT)
    for ia in range(n_atoms):
        splitline = in_file.readline().split()
        atom_types[ia] = int(splitline[0])
        atom_pos[ia, 0] = float(splitline[2])*conversion_factor
        atom_pos[ia, 1] = float(splitline[3])*conversion_factor
        atom_pos[ia, 2] = float(splitline[4])*conversion_factor
    
    # read the volumetric data (works only for orthogonal unit cell)
    grid_data = np.zeros((n_voxels[0], n_voxels[1], n_voxels[2]), dtype=DTYPE_FLOAT)
    for ix in range(n_voxels[0]):
        for iy in range(n_voxels[1]):
            iz = 0
            while iz < n_voxels[2]:
                splitline = in_file.readline().split()
                for data_entry in splitline:
                    grid_data[ix, iy, iz] = float(data_entry)*au_to_eV
                    iz = iz + 1
    
    # create the grid points
    xs = origin[0] + np.arange(n_voxels[0], dtype=DTYPE_FLOAT)*voxel_diffs[0]
    ys = origin[1] + np.arange(n_voxels[1], dtype=DTYPE_FLOAT)*voxel_diffs[1]
    zs = origin[2] + np.arange(n_voxels[2], dtype=DTYPE_FLOAT)*voxel_diffs[2]
    
    return atom_types, atom_pos, xs, ys, zs, grid_data
