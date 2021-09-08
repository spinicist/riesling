Data Format
===========

RIESLING uses `HDF5 <https://www.hdfgroup.org/solutions/hdf5>`_ to store input and intermediate data. Output data is by default written to HDF5 (``.h5``), but optionally can also be output to NiFTI (``.nii``).

RIESLING mandates that the non-cartesian data is stored in "spokes", which could equally be called frames or traces. Here we will treat the data as being stored in `S` spokes, with `N` data-points per spoke, and `C` k-space channels.

Header
------

To be considered valid RIESLING input, the HDF5 file must contain the header information datastructure, stored as a compound data-type in a dataset with the name ``info``. We reserve the right to change these fields of the header structure between versions of RIESLING. For the canonical definition of the header, see ``src/info.h``. A pseudo-code version of the header is given here for clarity:

.. code-block:: c

  struct Info {
    long type;
    long channels;
    long matrix[3];

    long read_points;
    long read_gap;
    long spokes_hi;
    long spokes_lo;
    float lo_scale;

    long volumes;
    long echoes;

    float tr;
    float voxel_size[3];
    float origin[3];
    float direction[3][3];
  };

* ``type`` defines the kind of acquisition. Currently two values are supported - 1 means the acquisition is fully 3D, while 2 means the acquisition is a 3D stack-of-stars or stack-of-spirals type acquisition, with cartesian phase-encoding blips for the third axis.
* ``channels`` defines the number of k-space channels / coil-elements.
* ``matrix`` defines the nominal matrix size for the scan - i.e. it determines the matrix size of the final reconstructed image (unless the `--fov` option is used).
* ``read_points`` sets the number of data-points in the readout direction, i.e. how many readout points per spoke.
* ``read_gap`` sets the number of data-points in the dead-time gap for a ZTE acquisition. These points will not be used in the reconstruction.
* ``spokes_hi`` sets the number of spokes in the main, high-resolution k-space acquisition. For non-ZTE acquisitions, this should be the total number of spokes / read-outs.
* ``spokes_lo`` sets the number of spokes for an extra low-resolution k-space data (i.e. for WASPI acquisitions). Probably only useful for ZTE.
* ``lo_scale`` determines the scaling of the low-resolution k-space relative to the high-resolution k-space. It should be set to the ratio of the high-res k-space to the low-res k-space, i.e. for a WASPI acquisition that only reaches 1/4 of the radius in k-space of the main acquisition, ``lo_scale`` should be set to 4. The trajectory points for the lo-res k-space should NOT include this scaling factor, i.e. the maximum value of lo-res trajectory points should be 1 (this scaling will then be applied to the data to give the correct k-space locations).
* ``volumes`` indicates how many volumes or time-points were acquired in the acquisition.
* ``echoes`` specifies how many separate echoes were acquired per time-point.

The final four fields specify the TR and image orientation as required to build a valid NIfTI output file.

* ``tr`` The repetition time. Should be specified in milliseconds as per NIfTI convention.
* ``voxel_size`` The nominal voxel-size. Should be specified in millimeters as per NIfTI/ITK convention.
* ``origin`` The physical-space location of the center of the voxel at index 0,0,0, as per ITK convention.
* ``direction`` The physical-space axes directions, as per ITK convention.

Trajectory
----------

The trajectory should be stored as a float array in a dataset with the name ``trajectory`` with dimensions ``SxNx3``. HDF5 uses a row-major convention, if your software is column major (RIESLING is internally) then this will be ``3xNxS``. The 3 co-ordinates correspond to the x, y & z locations within the k-space volume. For a full 3D acquisition these should be scaled such that the nominal edge of k-space in each direction is 0.5. Hence, for radial spokes the k-space locations go between 0 and 0.5, and for diameter spokes between -0.5 and 0.5. For a 3D stack trajectory, the z co-ordinate should be the slice/stack position.

Non-cartesian Data
------------------

The non-cartesian data itself must be stored in a complex-valued float-precision dataset named ``noncartesian`` with dimensions ``VxSxNxC`` where V is the number of volumes. HDF5 does not have a native complex-valued datatype, hence a compound datatype with a ``r`` and ``i`` members corresponding to the real and imaginary parts must be used. In contrast to other toolkits RIESLING stores the channels as the fastest-varying index, i.e. the data for each k-space point across all channels is stored contiguously.

Cartesian Data
--------------

The ``riesling grid`` command will produce a complex-valued dataset named ``cartesian`` containing the gridded cartesian data for all channels. The dimensions will depend on the reconstruction settings (notably the oversampling factor).

Image Data
----------

The output of a reconstruction command will write a complex-valued dataset ``image`` with dimensions ``VxZxYxX`` where V is the number of volumes, and X, Y & Z are the matrix size, unless the ``--mag`` command is specified in which case the dataset will be real-valued.

Density Compensation
--------------------

``riesling sdc`` pre-calculates Sample Density Correction factors. It produces a real-valued dataset ``sdc`` of dimension ``SxN``.

Meta-Information
----------------

RIESLING is capable of storing additional meta-information and passing it through the processing chain. This should be stored in an HDF5 group named ``meta``, and consist of key-value pairs where the key is the dataset name and the value is a single floating-point number.
