import numpy as np
cimport numpy as np

DTYPE_FLOAT = np.float64
ctypedef np.float64_t DTYPE_FLOAT_t
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t

cdef double angstrom_to_bohr = 1.88972613288564
cdef int row_width = 6

def write_cube_orthogonal(filename, np.ndarray[DTYPE_FLOAT_t, ndim=1] xgrid,
                    np.ndarray[DTYPE_FLOAT_t, ndim=1] ygrid, np.ndarray[DTYPE_FLOAT_t, ndim=1] zgrid,
                    np.ndarray[DTYPE_FLOAT_t, ndim=3] grid_data, comment_line=''):
    cdef int ia, ix, iy, iz
    cdef int nx = len(xgrid)
    cdef int ny = len(ygrid)
    cdef int nz = len(zgrid)
    cdef double dx = (xgrid[1]-xgrid[0])*angstrom_to_bohr
    cdef double dy = (ygrid[1]-ygrid[0])*angstrom_to_bohr
    cdef double dz = (zgrid[1]-zgrid[0])*angstrom_to_bohr
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] origin = np.array([xgrid[0]*angstrom_to_bohr,
                                                            ygrid[0]*angstrom_to_bohr,
                                                            zgrid[0]*angstrom_to_bohr])
    
    out_file = open(filename, 'w')
    out_file.write('Created by cube_file_writer\n')
    out_file.write(comment_line + '\n')
    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(0, origin[0], origin[1], origin[2]))

    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(nx,dx,0.0,0.0))
    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(ny,0.0,dy,0.0))
    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(nz,0.0,0.0,dz))
    
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                out_file.write('{0:14.6E}'.format(grid_data[ix,iy,iz]))
                if (iz+1)%row_width == 0:
                    out_file.write('\n')
            if nz%row_width > 0:
                out_file.write('\n')
    
    out_file.close()


def write_cube_nonorthogonal(filename, np.ndarray[DTYPE_INT_t, ndim=1] n_grid,
                            np.ndarray[DTYPE_FLOAT_t, ndim=2] voxel_vectors,
                            np.ndarray[DTYPE_FLOAT_t, ndim=3] grid_data,
                            np.ndarray[DTYPE_FLOAT_t, ndim=1] origin, comment_line=''):
    cdef ix, iy, iz, i_zrow, n_zrows, iz_begin
    
    
    out_file = open(filename, 'w')
    out_file.write('Created by cube_file_writer\n')
    out_file.write(comment_line + '\n')
    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(0, origin[0], origin[1], origin[2]))
    
    voxel_vectors = voxel_vectors*angstrom_to_bohr
    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(n_grid[0], voxel_vectors[0, 0], voxel_vectors[0, 1], voxel_vectors[0, 2]))
    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(n_grid[1], voxel_vectors[1, 0], voxel_vectors[1, 1], voxel_vectors[1, 2]))
    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(n_grid[2], voxel_vectors[2, 0], voxel_vectors[2, 1], voxel_vectors[2, 2]))
    
    n_zrows = int(n_grid[2]/row_width)+1
    for ix in range(n_grid[0]):
        for iy in range(n_grid[1]):
            z_section = []
            for i_zrow in range(n_zrows-1):
                iz_begin = i_zrow*row_width
                for iz in range(iz_begin, iz_begin+row_width):
                    z_section.append('{0:14.6E}'.format(grid_data[ix,iy,iz]))
                z_section.append('\n')
            for iz in range((n_zrows-1)*row_width, n_grid[2]):
                z_section.append('{0:14.6E}'.format(grid_data[ix,iy,iz]))
            z_section.append('\n')
            out_file.write(''.join(z_section))
    
    out_file.close()


def write_to_cube_with_atoms(filename, np.ndarray[DTYPE_FLOAT_t, ndim=1] xgrid,
                        np.ndarray[DTYPE_FLOAT_t, ndim=1] ygrid,
                        np.ndarray[DTYPE_FLOAT_t, ndim=1] zgrid,
                        np.ndarray[DTYPE_FLOAT_t, ndim=3] grid_data,
                        np.ndarray[DTYPE_INT_t, ndim=1] atom_types,
                        np.ndarray[DTYPE_FLOAT_t, ndim=2] atom_pos, comment_line=''):
    cdef int ia, ix, iy, iz
    cdef int natoms = len(atom_types)
    cdef int nx = len(xgrid)
    cdef int ny = len(ygrid)
    cdef int nz = len(zgrid)
    cdef double dx = (xgrid[1]-xgrid[0])*angstrom_to_bohr
    cdef double dy = (ygrid[1]-ygrid[0])*angstrom_to_bohr
    cdef double dz = (zgrid[1]-zgrid[0])*angstrom_to_bohr
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] origin = np.array([xgrid[0]*angstrom_to_bohr,
                                                            ygrid[0]*angstrom_to_bohr, 
                                                            zgrid[0]*angstrom_to_bohr])
    
    out_file = open(filename, 'w')
    out_file.write('Created by cube_file_writer\n')
    out_file.write(comment_line + '\n')
    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(natoms, origin[0], origin[1], origin[2]))

    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(nx,dx,0.0,0.0))
    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(ny,0.0,dy,0.0))
    out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}\n'.format(nz,0.0,0.0,dz))

    for ia in range(natoms):
        out_file.write('{0:5d}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n'.format(
                        atom_types[ia], 0.0, atom_pos[ia, 0]*angstrom_to_bohr,
                        atom_pos[ia, 1]*angstrom_to_bohr, atom_pos[ia, 2]*angstrom_to_bohr))

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                out_file.write('{0:14.6E}'.format(grid_data[ix,iy,iz]))
                if (iz+1)%row_width == 0:
                    out_file.write('\n')
            if nz%row_width > 0:
                out_file.write('\n')

    out_file.close()
