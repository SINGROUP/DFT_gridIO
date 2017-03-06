================
PyDFTFileIO
================
Description
-----------

Contains Python modules for efficient reading and writing of file formats containing values on a grid of points used by DFT codes. Modules for reading **cube** and **LOCPOT/CHGCAR** files are implemented as well as a module for writing **cube** files. cube format is used by CP2k and Gaussian, for example, and LOCPOT and CHGCAR formats by VASP.

Since the grids used by DFT codes contain a high number of points, the output files have a significant amount of data. High performance of the reading and writing of these files is thus essential. Therefore, these modules are written in Cython (Python with C-extensions) which is a compromise between speed of C and readability of Python. The modules can be easily used within Python scripts after they are compiled, or you can use automatic compilation which is described later in installition section.

Requirements
------------

- Python 2.7
- Cython (`http://cython.org/ <http://cython.org/>`_)
- NumPy

Installation
------------

Put this directory containing the Python and Cython (.pyx) modules to your ``PYTHONPATH`` environment variable. You can either run the setup scripts in the directory as

``python setup_x.py build_ext --inplace``

**or** you can use automatic compilation of Cython modules by adding line

``import pyximport; pyximport.install()``

to your Python scripts using the Cython modules. See `http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html <http://docs.cython.org/en/latest/src/tutorial/cython_tutorial.html>`_ for more information.

Usage
-----

cube_file_writer:
^^^^^^^^^^^^^^^^^
- ``write_cube_orthogonal`` can be used to write data on orthogonal grid to a cube file. ``xgrid``, ``ygrid`` and ``zgrid`` define the grid points. ``grid_data`` is a NumPy array of shape ``(len(xgrid), len(ygrid), len(zgrid))`` containing the values on the grid.
- ``write_cube_nonorthogonal`` can be used to write data on non-orthogonal grid to a cube file. In this case, ``n_grid`` defines the number of grid points along each voxel vector/incremental cell vector, and ``voxel_vectors`` is a 3x3 array containing the voxel vectors. ``grid_data`` is a NumPy array of shape ``(n_grid[0], n_grid[1], n_grid[2])`` containing the values on the grid.
- ``write_to_cube_with_atoms`` is the same function as ``write_cube_orthogonal`` except in that it allows to write an atomic geometry to the cube file. If the number of atoms is n, ``atom_types`` is expected to be a NumPy array of shape ``(n, 1)`` containing the atomic numbers and ``atom_pos`` is an array of shape ``(n, 3)`` containing the positions of each atom.

cube_import:
^^^^^^^^^^^^
- ``read_cube_nonorthogonal`` reads a cube file with non-orthogonal voxel vectors and returns the atomic geometry, voxel vectors and an array of shape ``(n_voxels[0], n_voxels[1], n_voxels[2])`` called ``grid_data`` containing the values on grid points defined by the vectors. If ``return_unstructured_data=True``, it returns ``volume_data`` which is an array of shape ``(n_voxels[0]*n_voxels[1]*n_voxels[2], 4)`` and each row contains the position of the grid point and the value at that point.
- ``read_cube_orthogonal`` reads a cube file with orthogonal voxel vectors and returns ``xs``, ``ys`` and ``zs`` which define the grid points and ``grid_data`` which is an array of shape ``(len(xs), len(ys), len(zs))`` containing the values on grid.

locpot_import:
^^^^^^^^^^^^^^
- Works for CHGCAR files as well because the file format is the same (only the values on grid have different meaning)
- ``check_grid_orthogonality`` can be used to check whether the voxel vectors/incremental cell vectors in the LOCPOT/CHGCAR file are orthogonal or not.
- ``read_locpot`` reads a LOCPOT/CHGCAR file and returns the atomic geometry, grid points and an array of values on the grid points. The structure of return values depends on ``orthogonal`` and ``return_unstructured_data`` arguments and follows the same scheme as in ``cube_import``

Examples
--------

You can find usage examples in ``DFT_grid_atomic_geo_post_proc`` package:
- ``plot_epot_cube``
- ``plot_locpot_efield_z``
- ``nonorthogonal_to_orthogonal_cube``

Author
------
Juha Ritala (2016)
`jritala@gmail.com <mailto:jritala@gmail.com>`_

