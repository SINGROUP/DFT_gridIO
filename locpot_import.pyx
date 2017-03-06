import numpy as np
cimport numpy as np

from atomic_symbols_and_numbers import atomic_symbol_to_number, atomic_number_to_symbol

DTYPE_FLOAT = np.float64
ctypedef np.float64_t DTYPE_FLOAT_t
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t


# Reads the voxel vectors from file and checks if they are orthogonal
def check_grid_orthogonality(filename):
    cdef int i
    cdef double scaling_factor
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] cell_vectors = np.zeros((3, 3), dtype=DTYPE_FLOAT)
    
    in_file = open(filename, 'r')
    is_orthogonal = False
    is_z_orthogonal = False
    
    # read comment line
    title = in_file.readline()
    
    # read unit cell
    splitline = in_file.readline().split()
    scaling_factor = float(splitline[0])
    
    for i in range(3):
        splitline = in_file.readline().split()
        cell_vectors[i, 0] = scaling_factor * float(splitline[0])
        cell_vectors[i, 1] = scaling_factor * float(splitline[1])
        cell_vectors[i, 2] = scaling_factor * float(splitline[2])
    
    if (np.dot(cell_vectors[0, :], cell_vectors[1, :]) == 0 and
        np.dot(cell_vectors[0, :], cell_vectors[2, :]) == 0 and
        np.dot(cell_vectors[1, :], cell_vectors[2, :]) == 0):
        is_orthogonal = True
    
    if (cell_vectors[2, 0] == 0 and cell_vectors[2, 1] == 0 and
        cell_vectors[0, 2] == 0 and cell_vectors[1, 2] == 0):
        is_z_orthogonal = True
    
    return is_orthogonal, is_z_orthogonal


def read_locpot(filename, orthogonal=True, return_unstructured_data=False):
    cdef int ia, n_atoms, ix, iy, iz, idata, i_atom_type, ia_per_type
    cdef double scaling_factor
    cdef np.ndarray[DTYPE_INT_t, ndim=1] n_voxels = np.zeros(3, dtype=DTYPE_INT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] voxel_diffs = np.zeros(3, dtype=DTYPE_FLOAT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] vector_coeffs = np.zeros(3, dtype=DTYPE_FLOAT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] cell_vectors = np.zeros((3, 3), dtype=DTYPE_FLOAT)
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] voxel_vectors = np.zeros((3, 3), dtype=DTYPE_FLOAT)
    cdef np.ndarray[DTYPE_INT_t, ndim=1] n_atoms_per_type
    cdef np.ndarray[DTYPE_INT_t, ndim=1] atom_types
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] atom_pos
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] xs, ys, zs
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=3] grid_data
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] volume_data
    
    in_file = open(filename, 'r')
    
    # read comment line
    title = in_file.readline()
    
    # read unit cell
    splitline = in_file.readline().split()
    scaling_factor = float(splitline[0])
    
    for ia in range(3):
        splitline = in_file.readline().split()
        cell_vectors[ia, 0] = scaling_factor * float(splitline[0])
        cell_vectors[ia, 1] = scaling_factor * float(splitline[1])
        cell_vectors[ia, 2] = scaling_factor * float(splitline[2])
    
    # read atom types and positions
    splitline = in_file.readline().split()
    atom_symbols = splitline
    splitline = in_file.readline().split()
    n_atoms_per_type = np.zeros(len(splitline), dtype=DTYPE_INT)
    for ia, atoms_per_symbol in enumerate(splitline):
        n_atoms_per_type[ia] = int(atoms_per_symbol)

    n_atoms = n_atoms_per_type.sum()
    atom_types = np.zeros(n_atoms, dtype=DTYPE_INT)
    atom_pos = np.zeros((n_atoms, 3), dtype=DTYPE_FLOAT)
    splitline = in_file.readline().split()
    coordinate_type = splitline[0]
    
    ia = 0
    if coordinate_type == 'Direct':
        for i_atom_type, atom_symbol in enumerate(atom_symbols):
            for ia_per_type in range(n_atoms_per_type[i_atom_type]):
                atom_types[ia] = atomic_symbol_to_number(atom_symbol)
                splitline = in_file.readline().split()
                vector_coeffs[0] = float(splitline[0])
                vector_coeffs[1] = float(splitline[1])
                vector_coeffs[2] = float(splitline[2])
                if orthogonal:
                    atom_pos[ia, 0] = vector_coeffs[0]*cell_vectors[0, 0]
                    atom_pos[ia, 1] = vector_coeffs[1]*cell_vectors[1, 1]
                    atom_pos[ia, 2] = vector_coeffs[2]*cell_vectors[2, 2]
                else:
                    atom_pos[ia, 0] = vector_coeffs[0]*cell_vectors[0, 0] + vector_coeffs[1]*cell_vectors[1, 0] + vector_coeffs[2]*cell_vectors[2, 0]
                    atom_pos[ia, 1] = vector_coeffs[0]*cell_vectors[0, 1] + vector_coeffs[1]*cell_vectors[1, 1] + vector_coeffs[2]*cell_vectors[2, 1]
                    atom_pos[ia, 2] = vector_coeffs[0]*cell_vectors[0, 2] + vector_coeffs[1]*cell_vectors[1, 2] + vector_coeffs[2]*cell_vectors[2, 2]
                ia = ia + 1
    elif coordinate_type == 'Cartesian':
        for i_atom_type, atom_symbol in enumerate(atom_symbols):
            for ia_per_type in range(n_atoms_per_type[i_atom_type]):
                atom_types[ia] = atomic_symbol_to_number(atom_symbol)
                splitline = in_file.readline().split()
                atom_pos[ia, 0] = scaling_factor * float(splitline[0])
                atom_pos[ia, 1] = scaling_factor * float(splitline[1])
                atom_pos[ia, 2] = scaling_factor * float(splitline[2])
                ia = ia +1
    else:
        raise Exception('Unknown coordinate type: {}'.format(coordinate_type))
    
    # read volumetric electrostatic potential data
    splitline = in_file.readline().split()
    if not splitline: # probably always empty line here
        splitline = in_file.readline().split()
    n_voxels[0] = int(splitline[0])
    n_voxels[1] = int(splitline[1])
    n_voxels[2] = int(splitline[2])
    voxel_vectors[0, :] = cell_vectors[0, :] / n_voxels[0]
    voxel_vectors[1, :] = cell_vectors[1, :] / n_voxels[1]
    voxel_vectors[2, :] = cell_vectors[2, :] / n_voxels[2]
    
    if orthogonal:
        grid_data = np.zeros((n_voxels[0], n_voxels[1], n_voxels[2]), dtype=DTYPE_FLOAT)
        idata = 0
        ix = 0
        iy = 0
        iz = 0
        while idata < n_voxels[0]*n_voxels[1]*n_voxels[2]:
            splitline = in_file.readline().split()
            for data_entry in splitline:
                grid_data[ix, iy, iz] = float(data_entry)
                idata = idata + 1
                ix = ix + 1
                if ix == n_voxels[0]:
                    iy = iy + 1
                    ix = 0
                if iy == n_voxels[1]:
                    iz = iz + 1
                    iy = 0
        
        # create the grid points
        voxel_diffs[0] = voxel_vectors[0, 0]
        voxel_diffs[1] = voxel_vectors[1, 1]
        voxel_diffs[2] = voxel_vectors[2, 2]
        xs = np.arange(n_voxels[0], dtype=DTYPE_FLOAT)*voxel_diffs[0]
        ys = np.arange(n_voxels[1], dtype=DTYPE_FLOAT)*voxel_diffs[1]
        zs = np.arange(n_voxels[2], dtype=DTYPE_FLOAT)*voxel_diffs[2]
        
        return atom_types, atom_pos, xs, ys, zs, grid_data
    
    elif not return_unstructured_data:
        grid_data = np.zeros((n_voxels[0], n_voxels[1], n_voxels[2]), dtype=DTYPE_FLOAT)
        idata = 0
        ix = 0
        iy = 0
        iz = 0
        while idata < n_voxels[0]*n_voxels[1]*n_voxels[2]:
            splitline = in_file.readline().split()
            for data_entry in splitline:
                grid_data[ix, iy, iz] = float(data_entry)
                idata = idata + 1
                ix = ix + 1
                if ix == n_voxels[0]:
                    iy = iy + 1
                    ix = 0
                if iy == n_voxels[1]:
                    iz = iz + 1
                    iy = 0
        
        return atom_types, atom_pos, n_voxels, voxel_vectors, grid_data
    
    else:
        volume_data = np.zeros((n_voxels[0]*n_voxels[1]*n_voxels[2], 4), dtype=DTYPE_FLOAT)
        idata = 0
        ix = 0
        iy = 0
        iz = 0
        while idata < n_voxels[0]*n_voxels[1]*n_voxels[2]:
            splitline = in_file.readline().split()
            for data_entry in splitline:
                volume_data[idata, 0] = ix*voxel_vectors[0, 0] + iy*voxel_vectors[1, 0] + iz*voxel_vectors[2, 0]
                volume_data[idata, 1] = ix*voxel_vectors[0, 1] + iy*voxel_vectors[1, 1] + iz*voxel_vectors[2, 1]
                volume_data[idata, 2] = ix*voxel_vectors[0, 2] + iy*voxel_vectors[1, 2] + iz*voxel_vectors[2, 2]
                volume_data[idata, 3] = float(data_entry)
                idata = idata + 1
                ix = ix + 1
                if ix == n_voxels[0]:
                    iy = iy + 1
                    ix = 0
                if iy == n_voxels[1]:
                    iz = iz + 1
                    iy = 0
        
        return atom_types, atom_pos, n_voxels, voxel_vectors, volume_data
